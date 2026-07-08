"""
Command-line runner for a single prune + BN-recalibration experiment on a YOLO
object-detection model (the detection analogue of run_experiment.py).

Given a YOLO model, a registered detection dataset, and a target sparsity, it
measures mAP at three stages -- baseline, after global L1 magnitude pruning, and
after BatchNorm recalibration -- collects per-BN-layer activation statistics for
each stage, and saves the numbers and plots into a per-experiment folder.

Detection logic lives in detection_util.py; pruning and the BN-stat machinery
are reused from util.py / stats.py. Because YOLO.val() fuses Conv+BN (deleting
BatchNorm layers), each stage collects its BN stats BEFORE its mAP is measured.

Example:
    python run_detection_experiment.py --model yolov8n --dataset coco128 --sparsity 0.5
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch

# Headless backend before stats.py imports pyplot.
import matplotlib
matplotlib.use("Agg")

from detection_util import (
    SUPPORTED_YOLO_MODELS,
    DETECTION_DATASETS,
    get_yolo_model,
    resolve_val_images,
    sample_yolo_batches,
    prune_yolo,
    recalibrate_bn_yolo,
    evaluate_map,
)
from util import sparsity
from stats import (
    collect_bn_activation_stats_from_batches,
    compare_stats_ratio,
    plot_stat_ratios,
    plot_accuracy_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLO prune + BN-recalibration experiment and save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=SUPPORTED_YOLO_MODELS,
                        help="YOLO model to load pretrained COCO weights for.")
    parser.add_argument("--dataset", required=True, choices=sorted(DETECTION_DATASETS),
                        help="Registered detection dataset (auto-downloaded).")
    parser.add_argument("--sparsity", required=True, type=float,
                        help="Target sparsity / global prune amount in (0, 1).")
    parser.add_argument("--experiment-name", default=None,
                        help="Output subfolder name. Default: <model>_<dataset>_sp<sparsity>.")
    parser.add_argument("--output-dir", default="results",
                        help="Root folder for experiment output subfolders.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference/eval image size.")
    parser.add_argument("--num-batches", type=int, default=16,
                        help="Image batches used to collect BN activation statistics.")
    parser.add_argument("--bn-count", type=int, default=16,
                        help="Image batches used to recalibrate BN running stats.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Images per batch for BN recalibration / stat collection.")
    return parser.parse_args()


def _stage_stats(model_nn, batches, device):
    """Collect BN activation stats for a DetectionModel (must run before val fusion)."""
    return collect_bn_activation_stats_from_batches(model_nn, batches, device)


def main():
    args = parse_args()

    if not 0.0 < args.sparsity < 1.0:
        raise SystemExit(f"--sparsity must be in (0, 1), got {args.sparsity}")

    exp_name = args.experiment_name or f"{args.model}_{args.dataset}_sp{args.sparsity}"
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_detection] device={device}  experiment={exp_name}")
    print(f"[run_detection] output folder: {out_dir}")

    # --- data (download if needed) + BN batches -----------------------------
    data_yaml, val_paths = resolve_val_images(args.dataset)
    print(f"[run_detection] dataset={args.dataset} ({data_yaml}), {len(val_paths)} val images")
    max_needed = max(args.num_batches, args.bn_count)
    batches = sample_yolo_batches(val_paths, device,
                                  num_batches=max_needed,
                                  batch_size=args.batch_size, imgsz=args.imgsz)
    stat_batches = batches[:args.num_batches]
    recal_batches = batches[:args.bn_count]
    print(f"[run_detection] built {len(batches)} image batches (bs={args.batch_size})")

    # --- stage 1: baseline --------------------------------------------------
    # NOTE: fresh YOLO instance per stage; pruning is deterministic so 'pruned'
    # and 'bn_recal' share identical masks. BN stats are collected BEFORE val()
    # fuses (and deletes) the BatchNorm layers.
    print("[run_detection] baseline: collecting BN stats + evaluating mAP ...")
    y_orig = get_yolo_model(args.model)
    y_orig.model.to(device)
    orig_stats = _stage_stats(y_orig.model, stat_batches, device)
    map_orig = evaluate_map(y_orig, data_yaml, imgsz=args.imgsz)

    # --- stage 2: global L1 pruning -----------------------------------------
    print(f"[run_detection] pruning (global L1, amount={args.sparsity}) ...")
    y_pruned = get_yolo_model(args.model)
    y_pruned.model.to(device)
    prune_yolo(y_pruned.model, args.sparsity)
    measured_sparsity = sparsity(y_pruned.model)
    pruned_stats = _stage_stats(y_pruned.model, stat_batches, device)
    map_pruned = evaluate_map(y_pruned, data_yaml, imgsz=args.imgsz)

    # --- stage 3: prune + BN recalibration ----------------------------------
    print(f"[run_detection] recalibrating BN ({len(recal_batches)} batches) ...")
    y_recal = get_yolo_model(args.model)
    y_recal.model.to(device)
    prune_yolo(y_recal.model, args.sparsity)
    recalibrate_bn_yolo(y_recal.model, recal_batches)
    recal_stats = _stage_stats(y_recal.model, stat_batches, device)
    map_recal = evaluate_map(y_recal, data_yaml, imgsz=args.imgsz)

    # --- BN activation ratios + plots ---------------------------------------
    ratio_res = compare_stats_ratio(
        orig_stats, {"pruned": pruned_stats, "bn_recal": recal_stats})

    map5095 = {"original": map_orig["map50_95"],
               "pruned": map_pruned["map50_95"],
               "bn_recal": map_recal["map50_95"]}

    bn_plot_path = os.path.join(out_dir, "bn_ratios.png")
    map_plot_path = os.path.join(out_dir, "map.png")
    plot_stat_ratios(ratio_res, title=f"{exp_name} - BN output ratios",
                     save_path=bn_plot_path)
    plot_accuracy_comparison(map5095, title=f"{exp_name} - mAP@0.5:0.95",
                             save_path=map_plot_path,
                             ylabel="mAP@0.5:0.95", ymax=1.0, value_fmt="{:.3f}")

    # --- results.json -------------------------------------------------------
    results = {
        "experiment_name": exp_name,
        "task": "detection",
        "model": args.model,
        "dataset": args.dataset,
        "data_yaml": data_yaml,
        "target_sparsity": args.sparsity,
        "measured_sparsity": measured_sparsity,
        "imgsz": args.imgsz,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "num_batches": args.num_batches,
        "bn_count": args.bn_count,
        "batch_size": args.batch_size,
        "map": {"original": map_orig, "pruned": map_pruned, "bn_recal": map_recal},
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- summary ------------------------------------------------------------
    print("\n=== Results (mAP@0.5:0.95 / mAP@0.5) ===")
    print(f"  {'baseline':<24} {map_orig['map50_95']:.4f} / {map_orig['map50']:.4f}")
    print(f"  {'pruned':<24} {map_pruned['map50_95']:.4f} / {map_pruned['map50']:.4f}")
    print(f"  {'pruned + BN recal':<24} {map_recal['map50_95']:.4f} / {map_recal['map50']:.4f}")
    print(f"  measured sparsity: {measured_sparsity:.4f}")
    print(f"\nSaved to: {out_dir}")
    print(f"  - {os.path.basename(results_path)}")
    print(f"  - {os.path.basename(bn_plot_path)}")
    print(f"  - {os.path.basename(map_plot_path)}")


if __name__ == "__main__":
    main()
