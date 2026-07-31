"""
reflow -- restoring signal variance in one-shot pruned vision models.

One-shot magnitude pruning leaves every BatchNorm layer dividing by the dense
model's statistics, so activations are over-normalized and their variance decays
multiplicatively with depth until the network can no longer tell inputs apart.
Reflow repairs that by correcting the *normalizer* instead of retraining the
weights, which takes a few dozen forward passes.

The library is six operations:

    load_model(name)                  a pretrained model, ready to prune
    load_dataset(name)                calibration + evaluation loaders
    evaluate(model, loader, device)   top-1 accuracy, before or after pruning
    prune_model(model, amount)        one-shot magnitude pruning
    reflow(model, batches, device)    restore the activation variance
    variance_ratios(...)              per-layer eta_l, dense vs. pruned

and :func:`run_experiment`, which strings them together into one run.

    >>> from reflow import run_experiment
    >>> result = run_experiment("resnet50", target_sparsity=0.8, measure_variance=True)
    >>> result.accuracy
    {'dense': 76.13, 'pruned': 4.28, 'pruned + reflow': 64.03}
"""

from .calibration import DEFAULT_CALIBRATION_BATCHES, reflow
from .data import DATASETS, SUPPORTED_DATASETS, cache_batches, load_dataset
from .evaluation import evaluate
from .models import (
    MODELS,
    SUPPORTED_MODELS,
    ModelSpec,
    batchnorm_layers,
    get_spec,
    load_model,
)
from .pipeline import (
    DENSE,
    PRUNED,
    REFLOWED,
    ExperimentResult,
    resolve_device,
    run_experiment,
)
from .pruning import (
    finalize_pruning,
    get_prunable_layers,
    get_prune_masks,
    layerwise_sparsity,
    prune_model,
    sparsity,
)
from .signal import collect_activation_variance, final_layer_ratio, variance_ratios

__version__ = "0.1.0"

__all__ = [
    # models & data
    "load_model", "load_dataset", "MODELS", "DATASETS", "SUPPORTED_MODELS",
    "SUPPORTED_DATASETS", "ModelSpec", "get_spec", "batchnorm_layers", "cache_batches",
    # pruning
    "prune_model", "sparsity", "layerwise_sparsity", "finalize_pruning",
    "get_prunable_layers", "get_prune_masks",
    # reflow
    "reflow", "DEFAULT_CALIBRATION_BATCHES",
    # measurement
    "evaluate", "collect_activation_variance", "variance_ratios", "final_layer_ratio",
    # pipeline
    "run_experiment", "ExperimentResult", "resolve_device",
    "DENSE", "PRUNED", "REFLOWED",
    "__version__",
]
