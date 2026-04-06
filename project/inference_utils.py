from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.config import load_config
from project.model import build_model
from project.utils import confusion_matrix, metrics_from_confusion


def default_palette(num_classes: int) -> np.ndarray:
    base = np.array(
        [
            [20, 20, 20],
            [226, 88, 34],
            [61, 133, 198],
            [86, 181, 92],
            [236, 179, 52],
            [171, 71, 188],
            [0, 150, 136],
            [255, 112, 67],
            [92, 107, 192],
            [141, 110, 99],
            [66, 165, 245],
            [102, 187, 106],
        ],
        dtype=np.uint8,
    )
    if num_classes <= len(base):
        return base[:num_classes]

    palette = [c for c in base]
    rng = np.random.default_rng(42)
    while len(palette) < num_classes:
        palette.append(rng.integers(0, 255, size=3, dtype=np.uint8))
    return np.stack(palette, axis=0)


def _build_eval_transform(cfg: dict):
    h, w = map(int, cfg["data"]["image_size"])
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    return A.Compose([A.Resize(h, w), A.Normalize(mean=mean, std=std), ToTensorV2()])


def _normalize_label_map(label_map):
    if label_map is None:
        return None
    if isinstance(label_map, dict):
        return {int(k): int(v) for k, v in label_map.items()}
    if isinstance(label_map, list):
        return {int(v): i for i, v in enumerate(label_map)}
    raise ValueError("label_map must be list/dict/null")


def map_mask_values(raw_mask: np.ndarray, cfg: dict) -> np.ndarray:
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., 0]
    raw_mask = raw_mask.astype(np.int64)
    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"]["ignore_index"])
    label_map = _normalize_label_map(cfg["data"].get("label_map"))

    if label_map is None:
        mapped = raw_mask
    else:
        mapped = np.full(raw_mask.shape, ignore_index, dtype=np.int64)
        for raw_value, cls in label_map.items():
            mapped[raw_mask == raw_value] = cls

    invalid = (mapped != ignore_index) & ((mapped < 0) | (mapped >= num_classes))
    if np.any(invalid):
        bad_values = np.unique(mapped[invalid])[:10].tolist()
        raise ValueError(f"Invalid class indices in mask: {bad_values}")
    return mapped


def load_model_bundle(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
):
    cfg = load_config(config_path)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = build_model(cfg).to(dev)

    try:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    except TypeError:
        # Backward compatibility with older torch releases.
        ckpt = torch.load(checkpoint_path, map_location=dev)
    if "model" not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'model' weights: {checkpoint_path}")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    transform = _build_eval_transform(cfg)
    return {"cfg": cfg, "device": dev, "model": model, "transform": transform}


@torch.no_grad()
def predict_mask(bundle: dict, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    cfg = bundle["cfg"]
    dev = bundle["device"]
    transform = bundle["transform"]

    h0, w0 = image_rgb.shape[:2]
    x = transform(image=image_rgb)["image"].unsqueeze(0).to(dev)
    logits, _ = model(x)
    logits = torch.nn.functional.interpolate(
        logits,
        size=(h0, w0),
        mode="bilinear",
        align_corners=False,
    )
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [C,H,W]
    pred = np.argmax(probs, axis=0).astype(np.int64)
    return pred, probs


def mask_to_color(mask: np.ndarray, num_classes: int, palette: np.ndarray | None = None) -> np.ndarray:
    pal = palette if palette is not None else default_palette(num_classes)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(num_classes):
        out[mask == cls] = pal[cls]
    return out


def overlay_mask(image_rgb: np.ndarray, mask_color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return np.clip(image_rgb.astype(np.float32) * (1.0 - alpha) + mask_color.astype(np.float32) * alpha, 0, 255).astype(
        np.uint8
    )


def compute_seg_metrics(pred_mask: np.ndarray, gt_mask_raw: np.ndarray, cfg: dict) -> dict:
    gt_mask = map_mask_values(gt_mask_raw, cfg)
    if pred_mask.shape[:2] != gt_mask.shape[:2]:
        gh, gw = gt_mask.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.int32), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"]["ignore_index"])
    cm = confusion_matrix(
        torch.from_numpy(pred_mask).long(),
        torch.from_numpy(gt_mask).long(),
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return metrics_from_confusion(cm)


def class_distribution(mask: np.ndarray, num_classes: int) -> list[float]:
    binc = np.bincount(mask.reshape(-1), minlength=num_classes).astype(np.float64)
    total = max(1.0, float(binc.sum()))
    return (binc / total).tolist()


def read_image_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask_raw(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask
