"""
Model registry: ``load_model(name)`` returns a pretrained network ready to prune.

``MODELS`` is the single source of truth for what ``--model`` accepts. Each
entry is a :class:`ModelSpec` recording not just how to build the network but
also which dataset it was trained on and the top-1 accuracy the weights are
published with -- the CLI prints that number next to the measured baseline so a
broken data path or transform shows up immediately.

Every model here is a BatchNorm CNN, which is what reflow rewrites the running
statistics of. To add a model, add one ``ModelSpec`` to ``MODELS``; nothing else
needs editing.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm

# Where the local (non-torchvision) checkpoints live. These are research
# artifacts that are not distributed with the source, so the repo root is only
# the default -- $REFLOW_CHECKPOINTS points elsewhere.
_CHECKPOINT_DIR = os.environ.get(
    "REFLOW_CHECKPOINTS",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class ModelSpec:
    """How to build one supported model, and what to expect from it."""

    builder: Callable[..., nn.Module]
    dataset: str                       # registered dataset name (see reflow.data)
    weights: Optional[str] = None      # torchvision weight enum; None = local checkpoint
    reference_top1: Optional[float] = None   # published top-1 of these weights (%)
    description: str = ""

    def build(self) -> nn.Module:
        if self.weights is None:
            return self.builder()
        return self.builder(weights=self.weights)


# ---------------------------------------------------------------------------
# local checkpoints
# ---------------------------------------------------------------------------
def _load_pickled_module(filename, cls, *ctor_args):
    """
    Load a *pickled nn.Module* (not a state_dict) from the repo root.

    These checkpoints were saved from sessions where the model class lived in
    ``__main__``, so the unpickler resolves ``__main__.<class name>``. Binding
    the class there keeps the load working whatever the entry point is.
    """
    path = os.path.join(_CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint for {cls.__name__} not found at {path}. It is a local "
            "artifact, not a download and not distributed with this repo -- put "
            f"{filename} in the repo root, or set $REFLOW_CHECKPOINTS to the "
            "directory holding it. The torchvision-backed models need no such file.")

    setattr(sys.modules["__main__"], cls.__name__, cls)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):          # tolerate a state_dict re-save
        model = cls(*ctor_args)
        model.load_state_dict(obj)
        return model
    return obj


def _mobilenet_v1():
    """MobileNetV1 trained on ImageNet (27 BN layers)."""
    from .architectures import MobileNet
    return _load_pickled_module("trained_MobileNetExplicit.pt", MobileNet)


def _resnet20_cifar():
    """ResNet-20 trained on CIFAR-10 (21 BN layers)."""
    from .architectures import ResNet20
    return _load_pickled_module("chita_trained_resnet20.pt", ResNet20, 10)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
# Weight enums are pinned per model so the dense baseline is reproducible: where
# torchvision offers several, the one whose published top-1 matches the number
# in `reference_top1` is selected explicitly rather than taking DEFAULT (which
# tracks the best available and changes between torchvision releases).
MODELS = {
    "resnet20": ModelSpec(
        builder=_resnet20_cifar, dataset="cifar10",
        description="ResNet-20 (CIFAR-10, local checkpoint)"),
    "mobilenet": ModelSpec(
        builder=_mobilenet_v1, dataset="imagenet",
        description="MobileNetV1 (ImageNet, local checkpoint)"),
    "resnet50": ModelSpec(
        builder=tvm.resnet50, dataset="imagenet",
        weights="IMAGENET1K_V1", reference_top1=76.13,
        description="ResNet-50"),
    "resnet101": ModelSpec(
        builder=tvm.resnet101, dataset="imagenet",
        weights="IMAGENET1K_V1", reference_top1=77.37,
        description="ResNet-101"),
    "resnet152": ModelSpec(
        builder=tvm.resnet152, dataset="imagenet",
        weights="IMAGENET1K_V1", reference_top1=78.31,
        description="ResNet-152"),
    "regnet_x_32gf": ModelSpec(
        builder=tvm.regnet_x_32gf, dataset="imagenet",
        weights="IMAGENET1K_V1", reference_top1=80.62,
        description="RegNetX-32GF"),
    "resnext101": ModelSpec(
        builder=tvm.resnext101_32x8d, dataset="imagenet",
        weights="IMAGENET1K_V2", reference_top1=82.83,
        description="ResNeXt-101 32x8d"),
}

SUPPORTED_MODELS = list(MODELS)

# Convenience spellings accepted on the command line, so the capitalised /
# torchvision names people already type keep working.
_ALIASES = {
    "resnet-20": "resnet20",
    "mobilenet_v1": "mobilenet",
    "mobilenetv1": "mobilenet",
    "resnet-50": "resnet50",
    "resnet-101": "resnet101",
    "resnet-152": "resnet152",
    "regnetx": "regnet_x_32gf",
    "regnet_x": "regnet_x_32gf",
    "resnext101_32x8d": "resnext101",
    "resnext-101": "resnext101",
}


def resolve_model_name(name):
    """Canonical registry key for a user-supplied model name (case-insensitive)."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Supported: {', '.join(SUPPORTED_MODELS)}")
    return key


def get_spec(name):
    """:class:`ModelSpec` for a model name (aliases resolved)."""
    return MODELS[resolve_model_name(name)]


def load_model(name, device=None):
    """
    Build a supported model with its pretrained weights, in eval mode.

    Returns ``(model, spec)`` so callers keep access to the dataset / norm-type
    metadata that drives the rest of the pipeline.
    """
    key = resolve_model_name(name)
    spec = MODELS[key]
    model = spec.build()
    model.eval()
    if device is not None:
        model = model.to(device)
    return model, spec


def batchnorm_layers(model):
    """``(name, module)`` for every BatchNorm2d, in ``named_modules`` order."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.BatchNorm2d)]
