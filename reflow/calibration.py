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
  what the *end-to-end optimal* ones are, so they are *fitted* on a small
  labeled calibration set with every other parameter frozen. Gradients,
  labels, minutes.

* :func:`reflow_layernorm_analytic` -- closed-form alternatives to the fitted
  route. Per-token normalization means pruning cannot change a LayerNorm
  output's scale, but it does shift the *per-channel* distribution of the
  normalized activations -- exactly the subspace gamma/beta control. Measuring
  that shift against the dense model gives a direct correction: forward passes
  only, no labels, no search. Needs the dense model as statistical reference.

:func:`reflow` dispatches on the model and is what the pipeline calls
(``ln_strategy`` picks the fitted or closed-form route within LayerNorm).
No path updates a single model weight.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .models import NORM_BATCH, _BN_TYPES, norm_kind

# BatchNorm statistics converge in a few tens of forward passes; past this the
# accuracy curve is flat. Fitting LayerNorm affine parameters is an
# optimization, so it needs an order of magnitude more. The analytic LayerNorm
# strategies are estimators like BatchNorm reflow and share its small budget.
DEFAULT_CALIBRATION_BATCHES = 50
DEFAULT_LAYERNORM_BATCHES = 500
DEFAULT_LAYERNORM_LR = 1e-3

# How a LayerNorm model is reflowed. BACKPROP is the fitted route (labeled
# gradient steps on gamma/beta -- the paper's REFLOW-LN, kept as the baseline).
# MOMENT and REGRESSION are closed-form, forward-pass-only corrections that
# take the dense model as a statistical reference instead of labels.
LN_BACKPROP = "backprop"
LN_MOMENT = "moment"
LN_REGRESSION = "regression"
LN_STRATEGIES = (LN_BACKPROP, LN_MOMENT, LN_REGRESSION)

# A channel whose normalized activation is (numerically) constant in the pruned
# model carries no signal to rescale; its affine parameters are left at the
# dense values rather than divided by a vanishing variance.
_DEGENERATE_VAR = 1e-10


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


def _ln_normalized(x, ln):
    """The pre-affine LayerNorm output of ``ln`` for input ``x``."""
    dims = tuple(range(-len(ln.normalized_shape), 0))
    mu = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + ln.eps)


def _ln_layers_in_forward_order(model, sample):
    """
    ``(name, module)`` for every affine LayerNorm, in forward-execution order.

    The order matters because the analytic correction is sequential: a layer's
    statistics are only meaningful once every layer before it is repaired, and
    ``named_modules`` order is not guaranteed to be execution order.
    """
    order, seen = [], set()

    def _record(name):
        def fn(_module, _inputs, _output):
            if name not in seen:
                seen.add(name)
                order.append(name)
        return fn

    handles = [m.register_forward_hook(_record(n))
               for n, m in model.named_modules()
               if isinstance(m, nn.LayerNorm) and m.elementwise_affine]
    if not handles:
        raise ValueError("model has no affine LayerNorm layers to calibrate")
    try:
        with torch.no_grad():
            model(sample)
    finally:
        for h in handles:
            h.remove()
    if not order:
        raise ValueError("no LayerNorm layer was executed by the model's forward pass")
    return [(n, model.get_submodule(n)) for n in order]


class _MomentAccumulator:
    """Streaming per-channel mean/variance over batches of normalized activations."""

    def __init__(self, shape, device):
        self.n = 0
        self.s = torch.zeros(shape, dtype=torch.float64, device=device)
        self.ss = torch.zeros(shape, dtype=torch.float64, device=device)

    def add(self, x):
        # Reduce each batch in fp32 (cheap, well-conditioned at this size),
        # accumulate across batches in fp64 so E[x^2] - E[x]^2 stays exact.
        flat = x.reshape(-1, *self.s.shape)
        self.n += flat.shape[0]
        self.s += flat.sum(0).to(torch.float64)
        self.ss += flat.square().sum(0).to(torch.float64)

    def moments(self):
        mean = self.s / self.n
        var = (self.ss / self.n - mean.square()).clamp_min_(0.0)
        return mean, var


@torch.no_grad()
def _collect_normalized_moments(model, layers, batches, device, progress=True):
    """Per-channel moments of every listed layer's pre-affine normalized activation."""
    accs = {name: _MomentAccumulator(tuple(m.normalized_shape), device)
            for name, m in layers}
    handles = [m.register_forward_hook(
                   lambda mod, inputs, _out, name=name:
                       accs[name].add(_ln_normalized(inputs[0].detach(), mod)))
               for name, m in layers]
    model.eval()
    try:
        for images in tqdm(batches, desc="reflow (LN) dense stats", leave=False,
                           disable=not progress):
            model(images.to(device, non_blocking=True))
    finally:
        for h in handles:
            h.remove()
    return {name: acc.moments() for name, acc in accs.items()}


@torch.no_grad()
def reflow_layernorm_analytic(model, dense_model, batches, device,
                              mode=LN_MOMENT, progress=True):
    """
    Closed-form correction of a pruned model's LayerNorm affine parameters.

    LayerNorm re-normalizes every token at run time, so pruning cannot leave a
    stale scale the way it does with BatchNorm -- but it *does* shift the
    per-channel distribution of the normalized activations x_hat, and gamma /
    beta are per-channel, so the drift lives exactly in the subspace the affine
    parameters control. Both modes measure that drift on calibration inputs and
    repair it in closed form -- forward passes only, no labels, no search:

    * ``moment``     -- match each LayerNorm output's per-channel mean and
                        variance to the dense model's. The LN analogue of
                        overwriting BN statistics: exact by construction,
                        energy-preserving regardless of how the activations
                        decorrelate.
    * ``regression`` -- per-channel least squares of the pruned normalized
                        activation onto the dense layer's output (run on the
                        same inputs). MSE-optimal, but being a regression it
                        shrinks gamma wherever dense and pruned activations
                        decorrelate -- conservative where ``moment`` amplifies.

    Layers are corrected sequentially in forward-execution order, because the
    variance decay compounds with depth: layer l's pruned statistics are only
    meaningful once every layer before it has been repaired. Correcting a layer
    never invalidates earlier corrections (it only affects what is downstream),
    so a single sequential pass suffices. Cost is one sweep of ``batches`` per
    LayerNorm (plus one dense sweep for ``moment``, or a paired dense forward
    per batch for ``regression``).

    ``dense_model`` must be the unpruned twin of ``model`` (same architecture
    and module names); both are expected to be on ``device`` and are run in
    eval mode. Neither model's weights are touched -- only ``model``'s
    LayerNorm gamma/beta are rewritten.

    Returns a dict describing what was done, for the run's results.json.
    ``median_affine_gain`` is the per-layer median of |gamma_new / gamma| -- the
    amplification the correction applied, i.e. the LN-side estimate of how much
    signal pruning removed at that depth.
    """
    if mode not in (LN_MOMENT, LN_REGRESSION):
        raise ValueError(
            f"unknown analytic LayerNorm mode {mode!r}, "
            f"expected {LN_MOMENT!r} or {LN_REGRESSION!r}")
    if not batches:
        raise ValueError("analytic LayerNorm reflow needs cached calibration batches")

    model.eval()
    dense_model.eval()
    layers = _ln_layers_in_forward_order(model, batches[0].to(device))

    dense_moments = None
    if mode == LN_MOMENT:
        # Dense statistics never change, so one sweep covers every layer; only
        # the regression's cross-moment needs a paired dense forward per batch.
        dense_moments = _collect_normalized_moments(
            dense_model, [(n, dense_model.get_submodule(n)) for n, _ in layers],
            batches, device, progress=progress)

    gains = []
    holder = {}

    def _grab(key):
        def fn(module, inputs, _output):
            holder[key] = _ln_normalized(inputs[0].detach(), module)
        return fn

    for name, ln in tqdm(layers, desc=f"reflow (LN-{mode})", leave=False,
                         disable=not progress):
        dense_ln = dense_model.get_submodule(name)
        shape = tuple(ln.normalized_shape)
        acc_p = _MomentAccumulator(shape, device)
        acc_d = _MomentAccumulator(shape, device) if mode == LN_REGRESSION else None
        cross = torch.zeros(shape, dtype=torch.float64, device=device)

        handles = [ln.register_forward_hook(_grab("pruned"))]
        if mode == LN_REGRESSION:
            handles.append(dense_ln.register_forward_hook(_grab("dense")))
        try:
            for images in batches:
                images = images.to(device, non_blocking=True)
                model(images)
                acc_p.add(holder["pruned"])
                if mode == LN_REGRESSION:
                    dense_model(images)
                    acc_d.add(holder["dense"])
                    cross += ((holder["pruned"] * holder["dense"])
                              .reshape(-1, *shape).sum(0).to(torch.float64))
        finally:
            for h in handles:
                h.remove()

        m_p, v_p = acc_p.moments()
        if mode == LN_MOMENT:
            m_d, v_d = dense_moments[name]
            scale = (v_d / v_p.clamp_min(_DEGENERATE_VAR)).sqrt()
        else:
            m_d, _ = acc_d.moments()
            cov = cross / acc_p.n - m_p * m_d
            scale = cov / v_p.clamp_min(_DEGENERATE_VAR)

        # Both modes reduce to an affine remap of the dense parameters: match
        # the output mean exactly, and scale gamma by the per-channel factor.
        gamma = dense_ln.weight.detach().to(torch.float64)
        beta = dense_ln.bias.detach().to(torch.float64)
        new_gamma = gamma * scale
        new_beta = gamma * m_d + beta - new_gamma * m_p

        degenerate = v_p < _DEGENERATE_VAR
        if degenerate.any():
            new_gamma[degenerate] = gamma[degenerate]
            new_beta[degenerate] = beta[degenerate]

        ln.weight.copy_(new_gamma.to(ln.weight.dtype))
        ln.bias.copy_(new_beta.to(ln.bias.dtype))
        gains.append((new_gamma.abs()
                      / gamma.abs().clamp_min(1e-12)).median().item())

    return {"strategy": f"layernorm_{mode}", "layers": len(layers),
            "calibration_batches": len(batches),
            "median_affine_gain": gains}


def reflow(model, batches=None, device=None, loader=None, num_batches=None,
           lr=DEFAULT_LAYERNORM_LR, ln_strategy=LN_BACKPROP, dense_model=None,
           progress=True):
    """
    Apply whichever reflow strategy fits ``model``'s normalization layers.

    BatchNorm models consume ``batches`` (cached input tensors). LayerNorm
    models are steered by ``ln_strategy``: ``"backprop"`` (the default) fits
    the affine parameters on the labeled ``loader``, while ``"moment"`` and
    ``"regression"`` correct them in closed form from ``batches`` and need
    ``dense_model`` as the statistical reference. Passing more than is needed
    is fine -- only the relevant inputs are used.
    """
    if norm_kind(model) == NORM_BATCH:
        if batches is None:
            raise ValueError("reflow of a BatchNorm model needs cached calibration batches")
        return reflow_batchnorm(model, batches, device, progress=progress)

    if ln_strategy not in LN_STRATEGIES:
        raise ValueError(
            f"unknown LayerNorm strategy {ln_strategy!r}, expected one of {LN_STRATEGIES}")

    if ln_strategy == LN_BACKPROP:
        if loader is None:
            raise ValueError(
                "backprop reflow of a LayerNorm model needs a labeled "
                "calibration loader, not cached inputs")
        return reflow_layernorm(
            model, loader, device,
            num_batches=num_batches if num_batches is not None else DEFAULT_LAYERNORM_BATCHES,
            lr=lr, progress=progress)

    if batches is None:
        raise ValueError(
            "analytic reflow of a LayerNorm model consumes cached calibration "
            "batches (no labels needed)")
    if dense_model is None:
        raise ValueError(
            "analytic reflow of a LayerNorm model needs the dense model as its "
            "statistical reference")
    return reflow_layernorm_analytic(model, dense_model, batches, device,
                                     mode=ln_strategy, progress=progress)
