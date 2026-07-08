"""
Command-line runner for a single prune + BN-recalibration experiment.

Given a model, a (registered) dataset, and a target sparsity, it measures top-1
accuracy at three stages -- baseline, after global L1 magnitude pruning, and
after BatchNorm recalibration -- and saves the numbers and plots into a
per-experiment folder.

All heavy lifting lives in util.py / stats.py; this script only orchestrates.

Example:
    python run_experiment.py --model ResNet50 --dataset imagenet --sparsity 0.7
"""

import argparse
import copy
import json
import os
from datetime import datetime, timezone

import torch

# Force a non-interactive backend BEFORE stats.py imports pyplot so the script
# can render plots to files on a headless machine.
import matplotlib
matplotlib.use("Agg")

from util import (
    SUPPORTED_MODELS,
    get_model,
    get_imagenet_loaders,
    global_pruning,
    model_update_bn,
    sparsity,
    test,
)
from stats import (
    sample_batches,
    collect_bn_activation_stats_from_batches,
    compare_stats_ratio,
    plot_stat_ratios,
    plot_accuracy_comparison,
)

# Named-dataset registry: name -> ImageFolder root (must contain train/ and val/).
DATASET_PATHS = {
    "imagenet": "/home/dw217/ImageNet/imagenet",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a prune + BN-recalibration experiment and save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS,
                        help="Model to load pretrained weights for.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PATHS),
                        help="Registered dataset name (see DATASET_PATHS).")
    parser.add_argument("--sparsity", required=True, type=float,
                        help="Target sparsity / global prune amount in (0, 1).")
    parser.add_argument("--experiment-name", default=None,
                        help="Output subfolder name. Default: <model>_<dataset>_sp<sparsity>.")
    parser.add_argument("--output-dir", default="results",
                        help="Root folder for experiment output subfolders.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--num-batches", type=int, default=100,
                        help="Batches used to collect BN activation statistics.")
    parser.add_argument("--bn-count", type=int, default=100,
                        help="Train batches used to recalibrate BN running stats.")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Cap evaluation to this many test batches (quick runs).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 < args.sparsity < 1.0:
        raise SystemExit(f"--sparsity must be in (0, 1), got {args.sparsity}")

    data_path = DATASET_PATHS[args.dataset]
    exp_name = args.experiment_name or f"{args.model}_{args.dataset}_sp{args.sparsity}"
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_experiment] device={device}  experiment={exp_name}")
    print(f"[run_experiment] output folder: {out_dir}")

    # --- data + model -------------------------------------------------------
    train_loader, test_loader = get_imagenet_loaders(
        data_path, args.batch_size, args.num_workers)
    model = get_model(args.model).to(device)

    # --- stage 1: baseline accuracy ----------------------------------------
    print("[run_experiment] evaluating baseline ...")
    acc_before = test(model, test_loader, device, max_batches=args.limit_batches)

    # --- stage 2: global L1 pruning ----------------------------------------
    print(f"[run_experiment] pruning (global L1, amount={args.sparsity}) ...")
    pruned_model = global_pruning(copy.deepcopy(model), args.sparsity)
    acc_pruned = test(pruned_model, test_loader, device, max_batches=args.limit_batches)
    measured_sparsity = sparsity(pruned_model)

    # --- stage 3: BN recalibration -----------------------------------------
    print(f"[run_experiment] recalibrating BN (count={args.bn_count}) ...")
    recal_model = copy.deepcopy(pruned_model)
    model_update_bn(recal_model, train_loader, device, count=args.bn_count)
    acc_recal = test(recal_model, test_loader, device, max_batches=args.limit_batches)

    # --- BN activation statistics + ratios ---------------------------------
    print(f"[run_experiment] collecting BN activation stats ({args.num_batches} batches) ...")
    batches = sample_batches(train_loader, device, num_batches=args.num_batches)
    orig_stats = collect_bn_activation_stats_from_batches(model, batches, device)
    pruned_stats = collect_bn_activation_stats_from_batches(pruned_model, batches, device)
    recal_stats = collect_bn_activation_stats_from_batches(recal_model, batches, device)
    ratio_res = compare_stats_ratio(
        orig_stats, {"pruned": pruned_stats, "bn_recal": recal_stats})

    # --- plots --------------------------------------------------------------
    accuracies = {"original": acc_before, "pruned": acc_pruned, "bn_recal": acc_recal}
    bn_plot_path = os.path.join(out_dir, "bn_ratios.png")
    acc_plot_path = os.path.join(out_dir, "accuracy.png")
    plot_stat_ratios(ratio_res, title=f"{exp_name} - BN output ratios",
                     save_path=bn_plot_path)
    plot_accuracy_comparison(accuracies, title=f"{exp_name} - accuracy",
                             save_path=acc_plot_path)

    # --- results.json -------------------------------------------------------
    results = {
        "experiment_name": exp_name,
        "model": args.model,
        "dataset": args.dataset,
        "data_path": data_path,
        "target_sparsity": args.sparsity,
        "measured_sparsity": measured_sparsity,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "limit_batches": args.limit_batches,
        "bn_count": args.bn_count,
        "num_batches": args.num_batches,
        "accuracy": accuracies,
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- summary ------------------------------------------------------------
    print("\n=== Results ===")
    print(f"  {'baseline':<24} {acc_before:6.2f}%")
    print(f"  {'pruned':<24} {acc_pruned:6.2f}%")
    print(f"  {'pruned + BN recal':<24} {acc_recal:6.2f}%")
    print(f"  measured sparsity: {measured_sparsity:.4f}")
    print(f"\nSaved to: {out_dir}")
    print(f"  - {os.path.basename(results_path)}")
    print(f"  - {os.path.basename(bn_plot_path)}")
    print(f"  - {os.path.basename(acc_plot_path)}")


if __name__ == "__main__":
    main()
