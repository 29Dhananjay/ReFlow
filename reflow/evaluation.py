"""Top-1 accuracy measurement."""

import torch
from tqdm import tqdm


def evaluate(model, loader, device, max_batches=None, desc="eval"):
    """
    Top-1 accuracy (percent) of ``model`` over ``loader``.

    ``max_batches`` caps how many batches are consumed, which turns a full
    ImageNet pass into a quick smoke run. ``None`` evaluates the whole split.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0:
        raise RuntimeError("evaluation loader yielded no batches")
    return 100.0 * correct / total
