"""
Reflow: restore a pruned network's activation variance without touching its weights.

One-shot pruning shrinks the variance of every layer's pre-normalization
activations, but the BatchNorm layers still divide by the dense model's stored
statistics. The activations are therefore over-normalized, and because each
layer feeds the next the shortfall compounds with depth until the deepest
activations carry almost no variance and the network stops discriminating
between inputs.

Reflow fixes the normalizer rather than the weights: it replaces each
BatchNorm's stored running mean/var with the statistics the *pruned* network
actually produces, measured over a small calibration set. Dividing activations
by their own standard deviation returns each layer to unit variance. Forward
passes only -- no gradients, no weight updates, tens of seconds.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

# BatchNorm statistics converge in a few tens of forward passes; past this the
# accuracy curve is flat.
DEFAULT_CALIBRATION_BATCHES = 50


@torch.no_grad()
def reflow(model, batches, device, progress=True):
    """
    Recompute every BatchNorm's running mean/var from the pruned model's own
    activations over ``batches`` (an iterable of input tensors).

    Three details matter and are easy to get wrong:

    * statistics are **reset** first and ``momentum`` is set to ``None``, so
      PyTorch accumulates a cumulative average -- the exact mean over the
      calibration set, independent of batch order. Leaving the default momentum
      would instead keep an exponential moving average still contaminated by the
      dense model's statistics;
    * only the BatchNorm layers are switched to train mode, not the whole model,
      so dropout stays off and does not perturb the statistics being measured;
    * the layers are restored to eval mode afterwards, ready for evaluation.

    Returns a dict describing what was done, for the run's results.json.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, _BN_TYPES)]
    if not bn_layers:
        raise ValueError(
            "model has no BatchNorm layers, so there are no statistics to "
            "recalibrate")

    model.eval()
    saved_momentum = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None              # cumulative average over the calibration set
        m.train()

    try:
        count = 0
        for images in tqdm(batches, desc="reflow", leave=False, disable=not progress):
            model(images.to(device, non_blocking=True))
            count += 1
    finally:
        for m, momentum in zip(bn_layers, saved_momentum):
            m.momentum = momentum
            m.eval()

    return {"layers": len(bn_layers), "calibration_batches": count}
