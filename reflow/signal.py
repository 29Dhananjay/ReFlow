"""
Per-layer activation variance, and the variance ratio between two models.

The quantity of interest is

    eta_l = Var_pruned(Z_l) / Var_dense(Z_l)

the ratio of post-normalization activation variance at layer ``l`` between a
pruned model and the dense one it came from. ``eta_l ~ 1`` means the layer
passes its signal through intact; ``eta_l -> 0`` down the depth of the network
is signal collapse -- activations converging on a constant, so distinct inputs
produce indistinguishable representations.

Both models must see *identical* inputs for the ratio to mean anything, which
is why the pipeline caches its calibration batches and replays them
(:func:`reflow.data.cache_batches`).
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .models import NORM_BATCH, _BN_TYPES, norm_kind


def _channel_variance(tensor, channel_dim):
    """
    Per-channel variance, reducing over every dimension except ``channel_dim``.

    Covers both layouts in play: (N, C, H, W) for BatchNorm, where channels are
    dim 1, and (N, tokens, D) for LayerNorm, where features are the last.
    """
    dims = [d for d in range(tensor.dim()) if d != channel_dim % tensor.dim()]
    return tensor.var(dim=dims, unbiased=False)


@torch.no_grad()
def collect_activation_variance(model, batches, device, progress=True):
    """
    Measure per-channel activation variance at every normalization layer.

    Returns ``(stats, order)`` where ``stats[name]`` holds the ``"input"``
    (pre-normalization) and ``"output"`` (post-normalization) variance as a
    per-channel tensor averaged over batches, and ``order`` lists the layer
    names in true forward-execution order -- which is what the plots index over
    and is not always the order ``named_modules`` reports.
    """
    is_batchnorm = norm_kind(model) == NORM_BATCH
    layer_types = _BN_TYPES if is_batchnorm else (nn.LayerNorm,)
    channel_dim = 1 if is_batchnorm else -1

    totals, counts, order = {}, {}, []

    def _hook(name):
        def fn(_module, inputs, output):
            if name not in totals:
                order.append(name)
                totals[name] = {"input": 0.0, "output": 0.0}
                counts[name] = 0
            totals[name]["input"] += _channel_variance(inputs[0], channel_dim).cpu()
            totals[name]["output"] += _channel_variance(output, channel_dim).cpu()
            counts[name] += 1
        return fn

    handles = [module.register_forward_hook(_hook(name))
               for name, module in model.named_modules()
               if isinstance(module, layer_types)]
    if not handles:
        raise ValueError("model has no normalization layers to measure")

    model.eval()
    try:
        for images in tqdm(batches, desc="variance", leave=False, disable=not progress):
            model(images.to(device, non_blocking=True))
    finally:
        for h in handles:
            h.remove()

    stats = {name: {key: value / counts[name] for key, value in totals[name].items()}
             for name in order}
    return stats, order


def variance_ratios(dense_stats, other_stats, order):
    """
    Layer-wise variance ratios of ``other`` against ``dense``, in forward order.

    Two reductions over channels are reported, because they answer different
    questions:

    ``output`` / ``input``  ratio of the channel-averaged variances. A handful
                            of loud channels can dominate this.
    ``*_median``            median over channels of the per-channel ratio, so it
                            reflects what a typical channel does.

    ``output`` is eta_l as defined above; ``input`` is the pre-normalization
    variance the normalization layer is fed.
    """
    result = {key: [] for key in ("output", "input", "output_median", "input_median")}
    for name in order:
        for key in ("output", "input"):
            dense, other = dense_stats[name][key], other_stats[name][key]
            result[key].append((other.mean() / (dense.mean() + 1e-12)).item())
            result[f"{key}_median"].append((other / (dense + 1e-12)).median().item())
    return result


def final_layer_ratio(ratios):
    """The deepest layer's eta -- the single number signal collapse shows up in."""
    return ratios["output"][-1] if ratios["output"] else float("nan")
