"""
Blur Evaluation: Before vs After mAP50 comparison

Given paired blurred images + ground truth masks, this script:
  1. Runs the model on the blurred image  → computes mAP50 / mIoU (BEFORE)
  2. Deblurs the image via Wiener filter
  3. Runs the model on the deblurred image → computes mAP50 / mIoU (AFTER)
  4. Saves per-image and aggregate results to outputs/blur_eval_results.json
  5. Saves RGB segmentation overlays for the dashboard

Usage:
    python scripts/blur_evaluate.py \
        --config  configs/blur_robust.yaml \
        --checkpoint configs/outputs/checkpoints/best.pth \
        --images  data/blurred_input \
        --masks   data/blurred_masks \
        --output  outputs/blur_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model
from src.data.transforms import build_eval_transforms
from src.training.metrics import confusion_matrix, metrics_from_confusion
from src.utils.config import load_config, resolve_paths

CLASS_NAMES = [
    "Road", "Low Vegetation", "Unpaved Road", "Obstacle",
    "High Vegetation", "Sky", "Vehicle", "Rough Trail",
    "Smooth Trail", "Water",
]
CLASS_COLORS = np.array([
    [128,  64, 128], [107, 142,  35], [190, 153, 153], [220,  20,  60],
    [ 34, 139,  34], [ 70, 130, 180], [  0,   0, 142], [244, 164,  96],
    [255, 255,   0], [  0, 191, 255],
], dtype=np.uint8)


# ── Deblur ─────────────────────────────────────────────────────────────────────
def wiener_deblur(img_rgb: np.ndarray, kernel_size: int = 21, noise_var: float = 0.02) -> np.ndarray:
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=np.float32)
    sigma = kernel_size / 6.0
    g = np.exp(-ax ** 2 / (2 * sigma ** 2))
    kernel = np.outer(g, g)
    kernel /= kernel.sum()

    img_f = img_rgb.astype(np.float32) / 255.0
    result = np.empty_like(img_f)
    h, w = img_f.shape[:2]
    kp = np.zeros((h, w), dtype=np.float32)
    kp[:kernel.shape[0], :kernel.shape[1]] = kernel
    K = np.fft.fft2(kp)
    denom = np.abs(K) ** 2 + noise_var
    K_conj = np.conj(K)
    for c in range(3):
        restored = np.real(np.fft.ifft2((K_conj / denom) * np.fft.fft2(img_f[:, :, c])))
        result[:, :, c] = np.clip(restored, 0.0, 1.0)
    out = (result * 255).astype(np.uint8)
    return cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model, img_rgb: np.ndarray, transforms, device) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    tensor = transforms(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.argmax(logits, dim=1).squeeze(0).cpu()


def load_mask(mask_path: Path, label_map: dict) -> torch.Tensor:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[2] == 1 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ids = np.full(mask.shape, 255, dtype=np.int64)
    for value, idx in label_map.items():
        ids[mask == value] = idx
    return torch.from_numpy(ids)


def colorize(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.numpy().astype(np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        rgb[mask == i] = c
    return rgb


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, num_classes: int,
                    ignore_classes: list) -> dict:
    cm = confusion_matrix(pred, gt, num_classes)
    return metrics_from_confusion(cm, ignore_classes)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--images",      required=True,  help="Folder of blurred images")
    parser.add_argument("--masks",       required=True,  help="Folder of ground truth masks")
    parser.add_argument("--output",      default="outputs/blur_eval")
    parser.add_argument("--kernel",      type=int,   default=21)
    parser.add_argument("--noise",       type=float, default=0.02)
    parser.add_argument("--ext",         default=".png,.jpg,.jpeg")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"]["num_classes"]
    ignore_classes = cfg.get("eval", {}).get("ignore_classes") or []

    # Build label_map
    dataset_cfg = cfg.get("dataset", {})
    raw_lm = dataset_cfg.get("label_map")
    label_map = {int(v): i for i, v in enumerate(raw_lm)} if raw_lm else {}

    # Load model
    model = build_model(cfg).to(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    model.eval()
    print(f"Loaded: {ckpt}")

    transforms = build_eval_transforms(
        tuple(cfg["train"]["image_size"]),
        mean=dataset_cfg.get("mean"),
        std=dataset_cfg.get("std"),
    )

    out_dir = Path(args.output)
    (out_dir / "seg_blurred").mkdir(parents=True, exist_ok=True)
    (out_dir / "seg_deblurred").mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.ext.split(",")}
    img_paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in exts)
    mask_dir = Path(args.masks)

    # Accumulators for aggregate confusion matrices
    cm_blurred   = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    cm_deblurred = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    per_image = []
    print(f"\nEvaluating {len(img_paths)} images...\n")

    for img_path in img_paths:
        # Find matching mask
        mask_path = None
        for ext in exts:
            candidate = mask_dir / f"{img_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            print(f"  [skip] no mask for {img_path.name}")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = load_mask(mask_path, label_map)

        # BEFORE: segment blurred
        pred_blurred = predict(model, img_rgb, transforms, device)
        m_b = compute_metrics(pred_blurred, gt, num_classes, ignore_classes)
        cm_blurred += confusion_matrix(pred_blurred, gt, num_classes)

        # AFTER: deblur then segment
        deblurred_rgb = wiener_deblur(img_rgb, args.kernel, args.noise)
        pred_deblurred = predict(model, deblurred_rgb, transforms, device)
        m_d = compute_metrics(pred_deblurred, gt, num_classes, ignore_classes)
        cm_deblurred += confusion_matrix(pred_deblurred, gt, num_classes)

        # Save RGB segmentation images
        cv2.imwrite(str(out_dir / "seg_blurred"   / f"{img_path.stem}.png"),
                    cv2.cvtColor(colorize(pred_blurred),   cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "seg_deblurred" / f"{img_path.stem}.png"),
                    cv2.cvtColor(colorize(pred_deblurred), cv2.COLOR_RGB2BGR))

        per_image.append({
            "image": img_path.name,
            "blurred":   {"map50": round(m_b["map50"], 4), "miou": round(m_b["miou"], 4),
                          "pixel_acc": round(m_b["pixel_acc"], 4)},
            "deblurred": {"map50": round(m_d["map50"], 4), "miou": round(m_d["miou"], 4),
                          "pixel_acc": round(m_d["pixel_acc"], 4)},
            "map50_gain": round(m_d["map50"] - m_b["map50"], 4),
        })
        print(f"  {img_path.name:30s}  blurred mAP50={m_b['map50']:.3f}  →  deblurred mAP50={m_d['map50']:.3f}  gain={m_d['map50']-m_b['map50']:+.3f}")

    # Aggregate metrics
    agg_b = metrics_from_confusion(cm_blurred,   ignore_classes)
    agg_d = metrics_from_confusion(cm_deblurred, ignore_classes)

    results = {
        "aggregate": {
            "blurred":   {"map50": round(agg_b["map50"], 4), "miou": round(agg_b["miou"], 4),
                          "pixel_acc": round(agg_b["pixel_acc"], 4),
                          "per_class_iou": [round(v, 4) for v in agg_b["per_class_iou"]]},
            "deblurred": {"map50": round(agg_d["map50"], 4), "miou": round(agg_d["miou"], 4),
                          "pixel_acc": round(agg_d["pixel_acc"], 4),
                          "per_class_iou": [round(v, 4) for v in agg_d["per_class_iou"]]},
            "map50_gain": round(agg_d["map50"] - agg_b["map50"], 4),
            "miou_gain":  round(agg_d["miou"]  - agg_b["miou"],  4),
            "class_names": CLASS_NAMES,
        },
        "per_image": per_image,
    }

    out_json = out_dir / "blur_eval_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*55}")
    print(f"  Aggregate  BLURRED   mAP50={agg_b['map50']:.4f}  mIoU={agg_b['miou']:.4f}")
    print(f"  Aggregate  DEBLURRED mAP50={agg_d['map50']:.4f}  mIoU={agg_d['miou']:.4f}")
    print(f"  mAP50 gain: {agg_d['map50']-agg_b['map50']:+.4f}")
    print(f"{'='*55}")
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
