"""
Reflow: restore a pruned network's activation variance without touching its weights.

One-shot pruning shrinks the variance of every layer's pre-normalization
activations, but the normalization layers still work off the dense model's
scale. The activations are over-normalized, and because each layer feeds the
next the shortfall compounds with depth until the deepest activations carry
almost no variance and the network stops discriminating between inputs.

Reflow fixes the normalizer rather than the weights. What that means depends on
whether the normalizer *stores* anything correctable:

* :func:`reflow_batchnorm` -- BatchNorm keeps ``running_mean``/``running_var``
  buffers inherited from the dense model, and they are simply wrong for the
  pruned network. Overwrite them with statistics measured from the pruned
  network's own activations and post-BN variance returns to gamma^2 by
  construction. Forward passes only: no gradients, no labels, seconds.

* :func:`reflow_layernorm` -- LayerNorm computes its statistics per sample at
  run time, so there are no stale buffers to overwrite. The only persistent
  handles on scale are the affine parameters, and there is no closed form for
  what they should become, so they are *fitted* on a small labeled calibration
  set with every other parameter frozen. Gradients, labels, minutes.

:func:`reflow` dispatches on the model and is what the pipeline calls. Neither
path updates a single model weight.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .models import NORM_BATCH, _BN_TYPES, norm_kind

# BatchNorm statistics converge in a few tens of forward passes; past this the
# accuracy curve is flat. Fitting LayerNorm affine parameters is an
# optimization, so it needs an order of magnitude more.
DEFAULT_CALIBRATION_BATCHES = 50
DEFAULT_LAYERNORM_BATCHES = 500
DEFAULT_LAYERNORM_LR = 1e-3


@torch.no_grad()
def reflow_batchnorm(model, batches, device, progress=True):
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
        raise ValueError("model has no BatchNorm layers to recalibrate")

    model.eval()
    saved_momentum = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None              # cumulative average over the calibration set
        m.train()

    try:
        count = 0
        for images in tqdm(batches, desc="reflow (BN)", leave=False,
                           disable=not progress):
            model(images.to(device, non_blocking=True))
            count += 1
    finally:
        for m, momentum in zip(bn_layers, saved_momentum):
            m.momentum = momentum
            m.eval()

    if count == 0:
        # The statistics were reset above, so leaving now would ship a model
        # normalizing by mean 0 / variance 1 -- far worse than not reflowing.
        raise ValueError(
            "no calibration batches were supplied, so BatchNorm statistics were "
            "reset but never re-estimated")

    return {"strategy": "batchnorm_statistics", "layers": len(bn_layers),
            "calibration_batches": count}


def reflow_layernorm(model, loader, device, num_batches=DEFAULT_LAYERNORM_BATCHES,
                     lr=DEFAULT_LAYERNORM_LR, progress=True):
    """
    Fit the LayerNorm affine parameters (gamma, beta) of a pruned model.

    Every other parameter -- attention, MLP, embeddings, classifier head -- is
    frozen, so the pruned weights themselves are never updated; the affine
    parameters are the only route back to a healthy activation scale when the
    normalizer keeps no running statistics.

    Unlike :func:`reflow_batchnorm` this is an optimization and consumes
    *labeled* batches, so it takes a loader rather than cached inputs. The model
    stays in eval mode (dropout off) while gradients flow only to the affine
    parameters. Because it converges gradually rather than exactly, an
    undersized ``num_batches`` underperforms silently instead of failing.

    Returns a dict describing what was done, for the run's results.json.
    """
    ln_layers = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    if not ln_layers:
        raise ValueError("model has no LayerNorm layers to calibrate")

    original_requires_grad = [(p, p.requires_grad) for p in model.parameters()]
    for parameter, _ in original_requires_grad:
        parameter.requires_grad_(False)

    affine = []
    for m in ln_layers:
        if m.elementwise_affine:
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            affine += [m.weight, m.bias]
    if not affine:
        raise ValueError("LayerNorm layers have no affine parameters to calibrate")

    optimizer = torch.optim.Adam(affine, lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.eval()                       # keep dropout off; gradients still flow
    first_loss = None
    total_loss = 0.0
    seen = 0
    bar = tqdm(total=num_batches, desc="reflow (LN)", leave=False, disable=not progress)
    for images, labels in loader:
        if seen >= num_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        total_loss += loss.item()
        seen += 1
        bar.update(1)
    bar.close()

    for parameter, required in original_requires_grad:
        parameter.requires_grad_(required)

    if seen == 0:
        raise ValueError(
            "no calibration batches were consumed, so the LayerNorm affine "
            "parameters were never updated")

    return {"strategy": "layernorm_affine", "layers": len(ln_layers),
            "calibration_batches": seen, "lr": lr,
            "first_loss": first_loss,
            "mean_loss": total_loss / seen if seen else float("nan")}


def reflow(model, batches=None, device=None, loader=None, num_batches=None,
           lr=DEFAULT_LAYERNORM_LR, progress=True):
    """
    Apply whichever reflow strategy fits ``model``'s normalization layers.

    BatchNorm models consume ``batches`` (cached input tensors); LayerNorm
    models consume ``loader`` directly, because fitting affine parameters needs
    labels. Passing both is fine -- only the relevant one is used.
    """
    if norm_kind(model) == NORM_BATCH:
        if batches is None:
            raise ValueError("reflow of a BatchNorm model needs cached calibration batches")
        return reflow_batchnorm(model, batches, device, progress=progress)

    if loader is None:
        raise ValueError(
            "reflow of a LayerNorm model needs a labeled calibration loader, "
            "not cached inputs")
    return reflow_layernorm(
        model, loader, device,
        num_batches=num_batches if num_batches is not None else DEFAULT_LAYERNORM_BATCHES,
        lr=lr, progress=progress)
