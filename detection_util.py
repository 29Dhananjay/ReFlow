"""
Detection-side helpers for the prune + BN-recalibration study, built on
Ultralytics YOLO (which uses real ``nn.BatchNorm2d`` throughout, unlike
torchvision's detection backbones that use FrozenBatchNorm2d).

Kept separate from util.py so the classification pipeline (notebook +
run_experiment.py) does not take on an ultralytics dependency. Pruning
(``global_pruning``/``sparsity``) and the BN activation-statistics machinery are
reused from util.py / stats.py unchanged.

Two YOLO-specific facts drive the design:
  * ``YOLO.val()`` FUSES Conv+BN into single Conv layers, which deletes every
    BatchNorm2d module. So all BN work (recalibration, stat collection) must
    happen BEFORE calling ``evaluate_map`` on a given model instance.
  * A DetectionModel called on a plain image tensor runs its inference forward
    (no targets/loss needed), so BN running stats can be refreshed simply by
    forwarding image batches in train() mode.
"""

import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import check_det_dataset

from util import get_prunable_layers, global_pruning, sparsity  # reuse pruning helpers

# name -> ultralytics checkpoint. Both are small, COCO-pretrained, BN-based.
YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
}
SUPPORTED_YOLO_MODELS = list(YOLO_MODELS)

# registered detection datasets -> ultralytics data YAML (auto-downloaded).
DETECTION_DATASETS = {
    "coco128": "coco128.yaml",  # 128 images, tiny/fast
    "coco": "coco.yaml",        # full COCO val2017 (~5k images)
}


def get_yolo_model(name):
    """Load a pretrained YOLO model wrapper by registered name."""
    if name not in YOLO_MODELS:
        raise ValueError(f"Model {name} not supported (choose from {SUPPORTED_YOLO_MODELS})")
    return YOLO(YOLO_MODELS[name])


def resolve_val_images(data_name):
    """Return (data_yaml, list_of_val_image_paths), downloading the dataset if needed."""
    if data_name not in DETECTION_DATASETS:
        raise ValueError(f"Dataset {data_name} not supported (choose from {list(DETECTION_DATASETS)})")
    data_yaml = DETECTION_DATASETS[data_name]
    info = check_det_dataset(data_yaml)  # downloads if missing
    val = info["val"]
    val = val[0] if isinstance(val, list) else val
    paths = sorted(
        glob.glob(os.path.join(val, "*.jpg"))
        + glob.glob(os.path.join(val, "*.png"))
    )
    return data_yaml, paths


def sample_yolo_batches(image_paths, device, num_batches=16, batch_size=4, imgsz=640):
    """
    Preprocess images the way YOLO expects (letterbox -> BGR2RGB -> CHW -> /255)
    and return a list of input tensors, for BN recalibration and BN-stat
    collection. Mirrors ``stats.sample_batches`` but for on-disk image files.
    """
    lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
    batches = []
    for b in range(num_batches):
        chunk = image_paths[b * batch_size:(b + 1) * batch_size]
        if not chunk:
            break
        ims = []
        for p in chunk:
            im = cv2.imread(p)
            if im is None:
                continue
            im = lb(image=im)[..., ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            ims.append(torch.from_numpy(np.ascontiguousarray(im)).float() / 255.0)
        if ims:
            batches.append(torch.stack(ims).to(device))
    return batches


def finalize_pruning(model):
    """
    Make pruning permanent: bake weight_mask into weight and drop the prune
    reparametrization on every Conv2d/Linear. Needed so YOLO's Conv+BN fusion
    (triggered by val()) sees plain zeroed weights rather than prune hooks.
    """
    for _, layer in get_prunable_layers(model):
        if hasattr(layer, "weight_orig"):
            prune.remove(layer, "weight")
    return model


def prune_yolo(model, prune_amount):
    """Global L1 magnitude prune of a YOLO DetectionModel, made permanent."""
    global_pruning(model, prune_amount)
    finalize_pruning(model)
    return model


def recalibrate_bn_yolo(model, batches):
    """
    Refresh BatchNorm running stats by forwarding image batches in train() mode.
    The DetectionModel's tensor-forward is inference-style, so no targets/loss
    are required. Analogue of util.model_update_bn for detection.
    """
    model.train()
    with torch.no_grad():
        for b in batches:
            model(b)
    return model


def evaluate_map(yolo, data_yaml, imgsz=640):
    """
    Run ultralytics validation and return {'map50_95', 'map50'}.

    NOTE: this FUSES the model in place (Conv+BN folded), removing BatchNorm2d
    layers -- call it only after any BN recalibration / stat collection on this
    model instance.
    """
    res = yolo.val(data=data_yaml, imgsz=imgsz, verbose=False, plots=False)
    return {"map50_95": float(res.box.map), "map50": float(res.box.map50)}
