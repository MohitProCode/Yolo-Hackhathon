from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_logger(output_dir: str, name: str = "seg_hardfix") -> logging.Logger:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / "train.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    if ignore_index >= 0:
        valid = targets != ignore_index
        preds = preds[valid]
        targets = targets[valid]
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]
    idx = num_classes * targets + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor) -> dict:
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(dim=1)
    pred = cm.sum(dim=0)
    fp = pred - tp
    fn = gt - tp

    denom = (tp + fp + fn).clamp(min=1.0)
    iou = tp / denom
    dice = (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1.0)
    valid = gt > 0

    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=cm.device)
    mdice = dice[valid].mean() if valid.any() else torch.tensor(0.0, device=cm.device)
    map50 = (iou[valid] >= 0.5).float().mean() if valid.any() else torch.tensor(0.0, device=cm.device)
    pix_acc = tp.sum() / cm.sum().clamp(min=1.0)

    gt_dist = (gt / gt.sum().clamp(min=1.0)).tolist()
    pred_dist = (pred / pred.sum().clamp(min=1.0)).tolist()
    return {
        "miou": float(miou.item()),
        "dice": float(mdice.item()),
        "map50": float(map50.item()),
        "pixel_acc": float(pix_acc.item()),
        "per_class_iou": iou.tolist(),
        "gt_dist": gt_dist,
        "pred_dist": pred_dist,
    }


def inverse_log_class_weights(pixel_counts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    counts = pixel_counts.astype(np.float64).copy()
    counts[counts <= 0] = np.nan
    freq = counts / np.nansum(counts)
    # Inverse-log weighting for imbalance robustness.
    weights = 1.0 / np.log(1.02 + freq + eps)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if weights.mean() > 0:
        weights = weights / weights.mean()
    return weights

