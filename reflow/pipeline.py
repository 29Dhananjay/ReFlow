"""
The prune-and-reflow run, end to end.

:func:`run_experiment` is the library's headline entry point and the CLI's only
call: load a model and a dataset, measure top-1 accuracy dense, prune it once,
measure again, apply reflow, measure a third time -- and, on request, report the
per-layer variance ratio behind those numbers.
"""

import copy
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import torch

from .calibration import (
    DEFAULT_CALIBRATION_BATCHES,
    DEFAULT_LAYERNORM_BATCHES,
    DEFAULT_LAYERNORM_LR,
    LN_BACKPROP,
    LN_STRATEGIES,
    reflow,
)
from .data import cache_batches, load_dataset
from .evaluation import evaluate
from .models import NORM_BATCH, get_spec, load_model, norm_kind, resolve_model_name
from .pruning import GLOBAL, prune_model, sparsity
from .signal import collect_activation_variance, variance_ratios

# Stage labels, used as dict keys in the results and as plot/table labels.
DENSE = "dense"
PRUNED = "pruned"
REFLOWED = "pruned + reflow"


@dataclass
class ExperimentResult:
    """Everything one run produced."""

    model: str
    dataset: str
    target_sparsity: float
    measured_sparsity: float
    accuracy: Dict[str, float]                     # stage -> top-1 (%)
    reflow_info: Dict = field(default_factory=dict)
    variance_ratios: Optional[Dict[str, Dict[str, List[float]]]] = None
    layer_order: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def recovered(self):
        """Accuracy points reflow put back relative to the pruned model."""
        return self.accuracy[REFLOWED] - self.accuracy[PRUNED]

    def to_dict(self):
        return {
            "model": self.model,
            "dataset": self.dataset,
            "target_sparsity": self.target_sparsity,
            "measured_sparsity": self.measured_sparsity,
            "accuracy": self.accuracy,
            "accuracy_recovered_by_reflow": self.recovered,
            "reflow": self.reflow_info,
            "layer_order": self.layer_order,
            "variance_ratios": self.variance_ratios,
            "metadata": self.metadata,
        }


def resolve_device(device=None):
    """``device`` as given, else CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(model_name, dataset=None, target_sparsity=0.8,
                   pruning_method=GLOBAL, batch_size=128, num_workers=8,
                   calibration_batches=None, variance_batches=16,
                   measure_variance=False, data_path=None, device=None,
                   limit_eval_batches=None, lr=DEFAULT_LAYERNORM_LR,
                   ln_strategy=LN_BACKPROP, seed=0, log=print):
    """
    Run dense -> pruned -> reflowed and report accuracy at each stage.

    Parameters that shape the run:

    ``dataset``               registered dataset name; defaults to the one the
                              model was trained on. Passing an incompatible one
                              is an error rather than a silently wrong number.
    ``calibration_batches``   batches reflow consumes. Defaults to 50 for a
                              BatchNorm model and for the analytic LayerNorm
                              strategies (both estimate statistics), 500 for
                              backprop LayerNorm fitting, which needs the
                              larger budget to converge.
    ``ln_strategy``           how a LayerNorm model is reflowed: ``"backprop"``
                              fits gamma/beta on labeled batches; ``"moment"``
                              and ``"regression"`` correct them in closed form
                              against the dense model's statistics, forward
                              passes only. Ignored by BatchNorm models.
    ``measure_variance``      also measure the per-layer variance ratio of the
                              pruned and reflowed models against the dense one.
    ``variance_batches``      batches used for that measurement (a prefix of the
                              calibration batches, so no extra data is loaded).
    ``limit_eval_batches``    cap every accuracy measurement to this many
                              batches, drawn as a fixed random subset of the
                              evaluation split -- for smoke runs, not for
                              publishable numbers.

    Returns an :class:`ExperimentResult`.
    """
    if not 0.0 < target_sparsity < 1.0:
        raise ValueError(f"target sparsity must be in (0, 1), got {target_sparsity}")
    if ln_strategy not in LN_STRATEGIES:
        raise ValueError(
            f"unknown LayerNorm strategy {ln_strategy!r}, expected one of {LN_STRATEGIES}")

    model_name = resolve_model_name(model_name)
    spec = get_spec(model_name)
    dataset = dataset or spec.dataset
    if dataset != spec.dataset:
        raise ValueError(
            f"model {model_name!r} was trained on {spec.dataset!r} and cannot be "
            f"evaluated on {dataset!r} (different input size / class count).")

    t_start = time.perf_counter()
    device = resolve_device(device)
    torch.manual_seed(seed)

    log(f"model={model_name} ({spec.description})  dataset={dataset}  "
        f"sparsity={target_sparsity}  device={device}")

    # A capped evaluation is scored on a fixed random subset of the split, so
    # all three stages see the same images and the estimate is not confined to
    # whichever classes happen to sort first.
    eval_subset = limit_eval_batches * batch_size if limit_eval_batches else None
    calibration_loader, eval_loader = load_dataset(
        dataset, batch_size=batch_size, num_workers=num_workers,
        data_path=data_path, seed=seed, eval_subset=eval_subset)

    # --- stage 1: dense -----------------------------------------------------
    model, _ = load_model(model_name, device=device)

    # The strategy reflow will use, and therefore the calibration budget, comes
    # from the model itself: BatchNorm replays cached inputs, backprop
    # LayerNorm fits affine parameters from the labeled loader and needs far
    # more batches. The analytic LayerNorm strategies are estimators like
    # BatchNorm reflow, so they share its cached inputs and small budget.
    is_batchnorm = norm_kind(model) == NORM_BATCH
    ln_analytic = not is_batchnorm and ln_strategy != LN_BACKPROP
    if calibration_batches is None:
        calibration_batches = (DEFAULT_CALIBRATION_BATCHES
                               if is_batchnorm or ln_analytic
                               else DEFAULT_LAYERNORM_BATCHES)

    # Whatever the strategy, the variance measurement must replay *identical*
    # inputs through all three models, so those batches are drawn once and
    # cached on CPU. BatchNorm reflow and the analytic LayerNorm strategies
    # reuse them; backprop LayerNorm reflow does not, so there only the
    # measurement batches are worth caching.
    n_cached = max(calibration_batches if (is_batchnorm or ln_analytic) else 0,
                   variance_batches if measure_variance else 0)
    batches = []
    if n_cached:
        log(f"caching {n_cached} calibration batches "
            f"({n_cached * batch_size} images) ...")
        batches = cache_batches(calibration_loader, n_cached)
    variance_input = batches[:variance_batches]

    def variance_of(model):
        if not measure_variance:
            return None, None
        return collect_activation_variance(model, variance_input, device)

    log("evaluating dense model ...")
    accuracy = {DENSE: evaluate(model, eval_loader, device, desc="dense")}
    reference = spec.reference_top1
    log(f"  {DENSE}: {accuracy[DENSE]:.2f}%"
        + (f"  (published top-1 for these weights: {reference:.2f}%)" if reference else ""))
    dense_stats, layer_order = variance_of(model)

    # --- stage 2: one-shot magnitude pruning --------------------------------
    log(f"pruning ({pruning_method} magnitude, amount={target_sparsity}) ...")
    pruned = prune_model(copy.deepcopy(model), target_sparsity, method=pruning_method)
    measured_sparsity = sparsity(pruned)
    accuracy[PRUNED] = evaluate(pruned, eval_loader, device, desc="pruned")
    log(f"  {PRUNED}: {accuracy[PRUNED]:.2f}%  (measured sparsity {measured_sparsity:.4f})")
    pruned_stats, _ = variance_of(pruned)

    # The analytic LayerNorm strategies still need the dense model as their
    # statistical reference; every other path is done with it.
    if not ln_analytic:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- stage 3: reflow ----------------------------------------------------
    log(f"applying reflow ({calibration_batches} calibration batches"
        + (f", {ln_strategy} strategy" if not is_batchnorm else "") + ") ...")
    reflowed = copy.deepcopy(pruned)
    t_reflow = time.perf_counter()
    reflow_info = reflow(reflowed, batches=batches[:calibration_batches],
                         device=device, loader=calibration_loader,
                         num_batches=calibration_batches, lr=lr,
                         ln_strategy=ln_strategy,
                         dense_model=model if ln_analytic else None)
    # The wall-clock cost of the calibration itself is a headline number for
    # reflow (the whole point is being cheap), so it is measured, not estimated.
    reflow_info["seconds"] = round(time.perf_counter() - t_reflow, 2)
    log(f"  reflow calibration took {reflow_info['seconds']:.1f}s")
    if ln_analytic:
        del model                      # reference served; free it before evaluation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    accuracy[REFLOWED] = evaluate(reflowed, eval_loader, device, desc="reflow")
    log(f"  {REFLOWED}: {accuracy[REFLOWED]:.2f}%")
    reflowed_stats, _ = variance_of(reflowed)

    # --- variance ratios ----------------------------------------------------
    ratios = None
    if measure_variance:
        ratios = {
            PRUNED: variance_ratios(dense_stats, pruned_stats, layer_order),
            REFLOWED: variance_ratios(dense_stats, reflowed_stats, layer_order),
        }
        log(f"  variance ratio at the final layer: "
            f"{ratios[PRUNED]['output'][-1]:.3f} pruned -> "
            f"{ratios[REFLOWED]['output'][-1]:.3f} after reflow")

    return ExperimentResult(
        model=model_name,
        dataset=dataset,
        target_sparsity=target_sparsity,
        measured_sparsity=measured_sparsity,
        accuracy=accuracy,
        reflow_info=reflow_info,
        variance_ratios=ratios,
        layer_order=layer_order,
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "pruning_method": pruning_method,
            "ln_strategy": None if is_batchnorm else ln_strategy,
            "batch_size": batch_size,
            "seed": seed,
            "limit_eval_batches": limit_eval_batches,
            "variance_batches": variance_batches if measure_variance else None,
            "reference_top1": reference,
            "total_seconds": round(time.perf_counter() - t_start, 2),
        },
    )
