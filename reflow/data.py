"""
Dataset registry: ``load_dataset(name)`` returns the two loaders the pipeline needs.

* a **calibration** loader, drawn from the training split -- feeds reflow's
  forward passes and the activation-variance measurement;
* an **evaluation** loader, the held-out split top-1 accuracy is measured on.

``DATASETS`` maps a registered *name* (not a path) to where the data lives and
how to build those splits. Neither dataset is downloaded -- both are local
filesystem copies. Point the library at yours in whichever way suits:

    --data-path /data/imagenet          per run, wins over everything
    export REFLOW_IMAGENET=/data/...    per machine (REFLOW_<NAME> for any dataset)
    edit the DatasetSpec below          per checkout

The registry defaults are the paths on the machine this was developed on, so on
any other machine one of the first two is required.
"""

import os
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 statistics the local ResNet-20 checkpoint was trained with. Note the
# (0.2023, 0.1994, 0.2010) std rather than torchvision's more common
# (0.2470, 0.2435, 0.2616): changing it shifts the reported accuracy.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _imagenet_splits(root):
    """Train / val datasets over an ImageFolder tree."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    for split in ("train", "val"):
        if not os.path.isdir(os.path.join(root, split)):
            raise FileNotFoundError(
                f"ImageNet root {root!r} has no {split}/ subfolder. Expected a "
                "standard ImageFolder layout (<root>/train, <root>/val).")

    return (datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf),
            datasets.ImageFolder(os.path.join(root, "val"), transform=eval_tf))


def _cifar10_splits(root):
    """Train / test datasets from a torchvision CIFAR-10 download root."""
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    return (datasets.CIFAR10(root=root, train=True, download=False, transform=train_tf),
            datasets.CIFAR10(root=root, train=False, download=False, transform=eval_tf))


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    splits: Callable[[str], tuple]
    num_classes: int
    input_size: int


DATASETS = {
    # imagenet: an ImageFolder tree with train/ and val/ subfolders.
    # cifar10:  a torchvision download root holding cifar-10-batches-py/.
    "imagenet": DatasetSpec(
        path="/home/dw217/ImageNet/imagenet",
        splits=_imagenet_splits, num_classes=1000, input_size=224),
    "cifar10": DatasetSpec(
        path="/home/ds304/Desktop/CIFAR10_DATA/",
        splits=_cifar10_splits, num_classes=10, input_size=32),
}

SUPPORTED_DATASETS = list(DATASETS)


def dataset_root(name, data_path=None):
    """
    Where dataset ``name`` lives: explicit argument, else ``$REFLOW_<NAME>``,
    else the registry default.
    """
    if data_path:
        return data_path
    return os.environ.get(f"REFLOW_{name.upper()}") or DATASETS[name].path


def load_dataset(name, batch_size=128, num_workers=8, data_path=None, seed=0,
                 eval_subset=None):
    """
    Build ``(calibration_loader, evaluation_loader)`` for a registered dataset.

    ``seed`` pins the calibration loader's shuffling, so a run is reproducible.

    ``eval_subset`` caps evaluation at that many *samples*. It draws a fixed
    random subset rather than a prefix, which matters because a validation set
    laid out by class (ImageFolder sorts them) would otherwise cap you to the
    first few classes. The subset is seeded, so every stage of a run is scored
    on exactly the same images.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported: {', '.join(SUPPORTED_DATASETS)}")
    spec = DATASETS[name]
    root = dataset_root(name, data_path)
    if not os.path.exists(root):
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {root!r}. This is a local copy, not "
            f"a download -- pass --data-path, or set REFLOW_{name.upper()}.")

    train_set, eval_set = spec.splits(root)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if eval_subset is not None and eval_subset < len(eval_set):
        picked = torch.randperm(len(eval_set), generator=torch.Generator().manual_seed(seed))
        eval_set = Subset(eval_set, picked[:eval_subset].tolist())

    calibration = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, generator=generator)
    evaluation = DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return calibration, evaluation


def cache_batches(loader, num_batches, device=None):
    """
    Materialise the first ``num_batches`` input tensors of ``loader``.

    The variance-ratio measurement has to push *identical* inputs through the
    dense, pruned and reflowed models, so the batches are drawn once and
    replayed. They stay on CPU unless ``device`` is given -- at ImageNet
    resolution the default 50 batches of 128 is already ~3.8 GB, so passing a
    device here pins that much GPU memory for the whole run.
    """
    batches = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        batches.append(images.to(device) if device is not None else images)
    return batches
