"""
One-shot magnitude pruning over a model's Conv2d / Linear weights.

``prune_model`` is the only entry point the pipeline uses; the rest are
diagnostics. Pruning is applied through ``torch.nn.utils.prune``, so a pruned
layer keeps its dense ``weight_orig`` plus a ``weight_mask`` and recomputes
``weight`` on every forward. Call :func:`finalize_pruning` to bake the mask in
and drop that reparametrization (needed before exporting a checkpoint that
should load without the pruning hooks).
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

GLOBAL = "global"
LAYERWISE = "layerwise"
PRUNING_METHODS = (GLOBAL, LAYERWISE)


def get_prunable_layers(model):
    """
    ``(name, module)`` for every layer pruning applies to.

    Conv2d and Linear weights only -- biases and BatchNorm parameters are never
    pruned.
    """
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]


def _detach_masked_weights(model):
    """
    Drop the autograd history from the masked weights pruning just computed.

    ``prune`` sets ``module.weight = weight_mask * weight_orig`` at prune time,
    outside any no-grad context, so the attribute carries a ``grad_fn`` until
    the next forward pass overwrites it. A module holding a non-leaf tensor
    cannot be deepcopied -- which is how the pipeline derives the reflowed model
    from the pruned one -- so a prune-then-copy with no forward in between would
    otherwise fail.

    Detaching is safe because the mask never changes and these weights are never
    trained: reflow updates BatchNorm statistics, not weights.
    """
    for _, module in get_prunable_layers(model):
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.grad_fn is not None:
            module.weight = weight.detach()
    return model


def prune_model(model, amount, method=GLOBAL):
    """
    Magnitude-prune ``amount`` (a fraction in (0, 1)) of the weights in place.

    ``GLOBAL`` ranks every prunable weight in the network against every other
    one, so the sparsity budget is spent where magnitudes are smallest overall;
    ``LAYERWISE`` removes the same fraction from each layer independently.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError(f"prune amount must be in (0, 1), got {amount}")
    if method not in PRUNING_METHODS:
        raise ValueError(
            f"unknown pruning method {method!r}, expected one of {PRUNING_METHODS}")

    layers = get_prunable_layers(model)
    if method == GLOBAL:
        prune.global_unstructured(
            [(module, "weight") for _, module in layers],
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    else:
        for _, module in layers:
            prune.l1_unstructured(module, name="weight", amount=amount)

    return _detach_masked_weights(model)


def finalize_pruning(model):
    """Bake ``weight_mask`` into ``weight`` and remove the prune reparametrization."""
    for _, module in get_prunable_layers(model):
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return model


def get_prune_masks(model):
    """
    Per-layer boolean keep-masks over the prunable weights.

    Read from ``weight_mask`` while the reparametrization is attached, which is
    exact. After :func:`finalize_pruning` the fallback (``weight != 0``) cannot
    distinguish a pruned weight from one that was already exactly zero.
    """
    masks = {}
    for name, module in get_prunable_layers(model):
        mask = getattr(module, "weight_mask", None)
        if mask is not None:
            masks[name] = mask.detach().cpu().bool()
        else:
            masks[name] = module.weight.detach().cpu() != 0
    return masks


def sparsity(model):
    """Fraction of prunable weights that are zero."""
    total = 0
    nonzero = 0
    for _, module in get_prunable_layers(model):
        total += module.weight.numel()
        nonzero += torch.count_nonzero(module.weight).item()
    return (total - nonzero) / total if total else 0.0


def layerwise_sparsity(model):
    """``{layer name: sparsity fraction}``, for inspecting how the budget landed."""
    report = {}
    for name, module in get_prunable_layers(model):
        total = module.weight.numel()
        nonzero = torch.count_nonzero(module.weight).item()
        report[name] = (total - nonzero) / total if total else 0.0
    return report
