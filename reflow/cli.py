"""
Command-line entry point: ``reflow --model resnet50 --sparsity 0.8 --plot``.

Runs one prune-and-reflow experiment and writes ``results.json`` (plus the
figures, with ``--plot``) into ``<output-dir>/<experiment-name>/``.
"""

import argparse
import json
import os
import sys
from statistics import median

from .calibration import DEFAULT_CALIBRATION_BATCHES
from .data import SUPPORTED_DATASETS
from .models import MODELS, SUPPORTED_MODELS, get_spec, resolve_model_name
from .pipeline import DENSE, PRUNED, REFLOWED, run_experiment
from .pruning import GLOBAL, PRUNING_METHODS


def build_parser():
    parser = argparse.ArgumentParser(
        prog="reflow",
        description="Prune a pretrained model, restore its activation variance "
                    "with reflow, and report top-1 accuracy at each stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help=f"One of: {', '.join(SUPPORTED_MODELS)}")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None,
                        help="Defaults to the dataset the model was trained on.")
    parser.add_argument("--sparsity", type=float, default=0.8,
                        help="Fraction of prunable weights to remove, in (0, 1).")
    parser.add_argument("--pruning", choices=PRUNING_METHODS, default=GLOBAL,
                        help="Rank weight magnitudes across the whole network or per layer.")

    parser.add_argument("--plot", action="store_true",
                        help="Save the variance-ratio and accuracy figures "
                             "(implies --variance).")
    parser.add_argument("--variance", action="store_true",
                        help="Measure per-layer variance ratios and record them "
                             "in results.json.")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int,
                        default=DEFAULT_CALIBRATION_BATCHES,
                        help="Batches reflow recomputes BatchNorm statistics over.")
    parser.add_argument("--variance-batches", type=int, default=16,
                        help="Batches used to measure activation variance.")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Cap every accuracy measurement to this many batches, "
                             "drawn as a fixed random subset of the evaluation "
                             "split (smoke runs only).")
    parser.add_argument("--data-path", default=None,
                        help="Override the registered dataset location.")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment-name", default=None,
                        help="Output subfolder. Default: <model>_<dataset>_sp<sparsity>.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the supported models and exit.")
    return parser


def print_models():
    print(f"{'name':<16} {'dataset':<10} {'published top-1':>15}  description")
    for name, spec in MODELS.items():
        top1 = f"{spec.reference_top1:.2f}%" if spec.reference_top1 else "-"
        print(f"{name:<16} {spec.dataset:<10} {top1:>15}  {spec.description}")


def print_summary(result):
    sparsity_pct = 100 * result.measured_sparsity
    print("\n" + "=" * 52)
    print(f"{result.model}  |  {result.dataset}  |  {sparsity_pct:.1f}% sparse")
    print("-" * 52)
    for stage in (DENSE, PRUNED, REFLOWED):
        print(f"  {stage:<20} {result.accuracy[stage]:>8.2f}%")
    print("-" * 52)
    print(f"  {'recovered by reflow':<20} {result.recovered:>+8.2f} points")
    if result.variance_ratios:
        pruned_eta = result.variance_ratios[PRUNED]["output"]
        reflow_eta = result.variance_ratios[REFLOWED]["output"]
        # The median is the robust summary: a residual stage can inflate the very
        # last layers, so the final-layer value alone can misread the run.
        print(f"  {'median eta':<20} {median(pruned_eta):>8.3f} -> {median(reflow_eta):.3f}")
        print(f"  {'final-layer eta':<20} {pruned_eta[-1]:>8.3f} -> {reflow_eta[-1]:.3f}")
    print("=" * 52)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print_models()
        return 0
    if not args.model:
        parser.error("--model is required (see --list-models)")

    measure_variance = args.variance or args.plot

    try:
        model_name = resolve_model_name(args.model)   # so aliases name one folder
        spec = get_spec(model_name)
    except ValueError as exc:
        parser.error(str(exc))

    dataset = args.dataset or spec.dataset
    name = args.experiment_name or f"{model_name}_{dataset}_sp{args.sparsity}"
    out_dir = os.path.join(args.output_dir, name)

    def log(message):
        print(f"[reflow] {message}", flush=True)

    try:
        result = run_experiment(
            args.model,
            dataset=args.dataset,
            target_sparsity=args.sparsity,
            pruning_method=args.pruning,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            calibration_batches=args.calibration_batches,
            variance_batches=args.variance_batches,
            measure_variance=measure_variance,
            data_path=args.data_path,
            device=args.device,
            limit_eval_batches=args.limit_batches,
            seed=args.seed,
            log=log,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)   # only once there is something to write
    written = []
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    written.append(results_path)

    if args.plot:
        # Imported here so a run without --plot never touches matplotlib.
        from .plotting import plot_accuracy, plot_variance_ratios
        title = f"{result.model} - {100 * result.measured_sparsity:.0f}% sparsity"
        written.append(plot_variance_ratios(
            result.variance_ratios, os.path.join(out_dir, "variance_ratio.png"),
            title=f"{title} - activation variance vs. dense"))
        written.append(plot_accuracy(
            result.accuracy, os.path.join(out_dir, "accuracy.png"), title=title))

    print_summary(result)
    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
