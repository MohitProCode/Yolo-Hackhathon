"""
Before vs After mAP50 Evaluation (No Ground Truth Required)

Strategy:
  - Run model on AFTER (clear/high-res) images  -> treat as pseudo ground-truth
  - Run model on BEFORE (blurred) images         -> evaluate against pseudo-GT
  - Compute mAP50, mIoU, per-class IoU
  - Save RGB segmentation maps + side-by-side panel
  - Write results to outputs/before_after_eval/results.json

Usage:
    python scripts/before_after_eval.py \
        --config     configs/deeplabv3plus_resnet50.yaml \
        --checkpoint outputs/deeplabv3plus/checkpoints/best.pth

Folder layout expected (auto-detected from data/):
    data/before/   1.jpg  2.jpg  ...
    data/before/after/   IMG-xxx.jpg  ...

Images are matched by sorted order (1st before <-> 1st after, etc.)
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


def set_inference_model_defaults(cfg: dict) -> dict:
    model_cfg = cfg.setdefault("model", {})
    name = str(model_cfg.get("name", "")).lower()
    if name in {"unet", "attention_unet", "attunet", "deeplabv3plus", "deeplabv3+"}:
        if model_cfg.get("pretrained", False):
            model_cfg["use_timm_backbone"] = True
            model_cfg["pretrained"] = False
    return cfg


# -- Class metadata -------------------------------------------------------------
CLASS_NAMES = [
    "Road", "Low Veg.", "Unpaved Rd.", "Obstacle",
    "High Veg.", "Sky", "Vehicle", "Rough Trail",
    "Smooth Trail", "Water",
]
CLASS_COLORS = np.array([
    [128,  64, 128], [107, 142,  35], [190, 153, 153], [220,  20,  60],
    [ 34, 139,  34], [ 70, 130, 180], [  0,   0, 142], [244, 164,  96],
    [255, 255,   0], [  0, 191, 255],
], dtype=np.uint8)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _metrics_from_confusion_with_ignored(cm: torch.Tensor, ignore_classes: list[int] | None) -> dict:
    if not ignore_classes:
        return metrics_from_confusion(cm)
    cm = cm.clone()
    for idx in ignore_classes:
        if 0 <= idx < cm.shape[0]:
            # Ignore pixels whose GT is in these classes (row-wise).
            cm[idx, :] = 0
    return metrics_from_confusion(cm)


# -- Inference ------------------------------------------------------------------
def predict(model, img_rgb: np.ndarray, transforms, device, out_hw: tuple) -> torch.Tensor:
    tensor = transforms(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return torch.argmax(logits, dim=1).squeeze(0).cpu()


def colorize(mask: torch.Tensor) -> np.ndarray:
    arr = mask.numpy().astype(np.uint8)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        rgb[arr == i] = c
    return rgb


# -- Panel builder --------------------------------------------------------------
def draw_iou_bars(per_class_iou: list, width: int = 900) -> np.ndarray:
    """Draw per-class IoU bars for blurred-image predictions vs pseudo-GT."""
    n       = len(CLASS_NAMES)
    bar_h   = 24
    gap     = 8
    pad     = 10
    label_w = 110
    total_h = pad + 24 + n * (bar_h + gap) + pad
    canvas  = np.ones((total_h, width, 3), dtype=np.uint8) * 30

    cv2.putText(canvas, "Per-Class IoU (blurred vs clear pseudo-GT)",
                (pad, 20), FONT_BOLD, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    bar_w = width - label_w - pad * 2
    y = 32
    for i, name in enumerate(CLASS_NAMES):
        iou = per_class_iou[i] if i < len(per_class_iou) else 0.0
        cv2.putText(canvas, name, (pad, y + bar_h - 6),
                    FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        x0 = pad + label_w
        bw = int(bar_w * max(0.0, min(1.0, iou)))
        color = (60, 200, 60) if iou >= 0.5 else (60, 140, 200) if iou >= 0.25 else (60, 60, 200)
        cv2.rectangle(canvas, (x0, y), (x0 + bw, y + bar_h - 2), color, -1)
        cv2.putText(canvas, f"{iou:.3f}", (x0 + bw + 4, y + bar_h - 6),
                    FONT, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y += bar_h + gap
    return canvas


def build_panel(before_bgr, after_bgr, seg_before_rgb, seg_after_rgb,
                metrics_before: dict, stem: str) -> np.ndarray:
    H = 420
    def resize(img): return cv2.resize(img, (H, H))
    def titled(img, title, sub):
        out = resize(img).copy()
        cv2.rectangle(out, (0, 0), (H, 44), (0, 0, 0), -1)
        cv2.putText(out, title, (6, 20), FONT_BOLD, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, sub,   (6, 38), FONT,      0.42, (200, 255, 200), 1, cv2.LINE_AA)
        return out

    mb = metrics_before
    top = np.hstack([
        titled(before_bgr,
               "BEFORE (blurred)", f"mAP50={mb['map50']:.3f}  mIoU={mb['miou']:.3f}"),
        titled(after_bgr,
               "AFTER (clear) - pseudo-GT", "reference only"),
        titled(cv2.cvtColor(seg_before_rgb, cv2.COLOR_RGB2BGR),
               "Seg BEFORE", f"pixAcc={mb['pixel_acc']:.3f}"),
        titled(cv2.cvtColor(seg_after_rgb,  cv2.COLOR_RGB2BGR),
               "Seg AFTER (GT)", "pseudo ground truth"),
    ])

    bars   = draw_iou_bars(mb["per_class_iou"], width=top.shape[1])
    banner = np.ones((52, top.shape[1], 3), dtype=np.uint8) * 20
    txt    = (f"  {stem}   |   mAP50={mb['map50']:.3f}  "
              f"mIoU={mb['miou']:.3f}  pixAcc={mb['pixel_acc']:.3f}  "
              f"(blurred vs clear pseudo-GT)")
    cv2.putText(banner, txt, (8, 34), FONT_BOLD, 0.52, (0, 255, 180), 1, cv2.LINE_AA)
    return np.vstack([banner, top, bars])


def draw_legend(width: int) -> np.ndarray:
    h = 28
    leg = np.ones((h, width, 3), dtype=np.uint8) * 30
    cw  = width // len(CLASS_NAMES)
    for i, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        x = i * cw
        cv2.rectangle(leg, (x+2, 4), (x+16, h-4), col.tolist(), -1)
        cv2.putText(leg, name, (x+19, h-7), FONT, 0.3, (210,210,210), 1, cv2.LINE_AA)
    return leg


# -- Main -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Before vs After mAP50 evaluation")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--before",      default="data/before",
                        help="Folder of blurred/before images")
    parser.add_argument("--after",       default="data/before/after",
                        help="Folder of clear/after images")
    parser.add_argument("--output",      default="outputs/before_after_eval")
    parser.add_argument("--ext",         default=".jpg,.png,.jpeg")
    args = parser.parse_args()

    cfg    = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    cfg    = set_inference_model_defaults(cfg)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes    = cfg["model"]["num_classes"]
    ignore_classes = cfg.get("eval", {}).get("ignore_classes") or []

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    model.eval()
    print(f"Loaded: {ckpt}\n")

    dataset_cfg = cfg.get("dataset", {})
    transforms  = build_eval_transforms(
        tuple(cfg["train"]["image_size"]),
        mean=dataset_cfg.get("mean"),
        std=dataset_cfg.get("std"),
    )

    exts         = {e.strip().lower() for e in args.ext.split(",")}
    before_imgs  = sorted(p for p in Path(args.before).iterdir()
                          if p.suffix.lower() in exts and p.is_file())
    after_imgs   = sorted(p for p in Path(args.after).iterdir()
                          if p.suffix.lower() in exts and p.is_file())

    if not before_imgs:
        raise FileNotFoundError(f"No images found in {args.before}")
    if not after_imgs:
        raise FileNotFoundError(f"No images found in {args.after}")

    # Match by sorted order
    pairs = list(zip(before_imgs, after_imgs))
    print(f"Found {len(pairs)} before/after pairs\n")

    out_dir = Path(args.output)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "seg_before").mkdir(exist_ok=True)
    (out_dir / "seg_after").mkdir(exist_ok=True)

    # Aggregate confusion matrices
    cm_before = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    cm_after  = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    per_image_results = []

    for b_path, a_path in pairs:
        print(f"  {b_path.name}  <->  {a_path.name}")

        b_bgr = cv2.imread(str(b_path))
        a_bgr = cv2.imread(str(a_path))
        if b_bgr is None or a_bgr is None:
            print("    [skip] cannot read")
            continue

        b_rgb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB)
        a_rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)

        # Use after-image dimensions as reference
        H, W  = a_bgr.shape[:2]

        # Predict on both
        pred_after  = predict(model, a_rgb, transforms, device, (H, W))
        pred_before = predict(model, b_rgb, transforms, device, (H, W))

        # After prediction = pseudo ground truth
        cm_pair = confusion_matrix(pred_before, pred_after, num_classes)
        cm_before += cm_pair

        cm_after_pair = confusion_matrix(pred_after, pred_after, num_classes)
        cm_after += cm_after_pair

        m_b = _metrics_from_confusion_with_ignored(cm_pair, ignore_classes)
        m_a = _metrics_from_confusion_with_ignored(cm_after_pair, ignore_classes)

        # Colorize
        seg_b_rgb = colorize(pred_before)
        seg_a_rgb = colorize(pred_after)

        stem = b_path.stem
        cv2.imwrite(str(out_dir / "seg_before" / f"{stem}.png"),
                    cv2.cvtColor(seg_b_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "seg_after"  / f"{stem}.png"),
                    cv2.cvtColor(seg_a_rgb, cv2.COLOR_RGB2BGR))

        # Build panel
        panel = build_panel(b_bgr, a_bgr, seg_b_rgb, seg_a_rgb, m_b, stem)
        legend = draw_legend(panel.shape[1])
        cv2.imwrite(str(out_dir / "panels" / f"{stem}_panel.png"),
                    np.vstack([panel, legend]))

        per_image_results.append({
            "before_file": b_path.name,
            "after_file":  a_path.name,
            "before": {
                "map50":      round(m_b["map50"], 4),
                "miou":       round(m_b["miou"],  4),
                "pixel_acc":  round(m_b["pixel_acc"], 4),
                "per_class_iou": [round(v, 4) for v in m_b["per_class_iou"]],
            },
            "after": {
                "map50":      round(m_a["map50"], 4),
                "miou":       round(m_a["miou"],  4),
                "pixel_acc":  round(m_a["pixel_acc"], 4),
                "per_class_iou": [round(v, 4) for v in m_a["per_class_iou"]],
            },
            "map50_gain": round(m_a["map50"] - m_b["map50"], 4),
            "miou_gain":  round(m_a["miou"] - m_b["miou"], 4),
        })

        print(f"    Before mAP50={m_b['map50']:.3f}  mIoU={m_b['miou']:.3f}  "
              f"(vs clear-image pseudo-GT)")

    # Aggregate metrics - only before (vs pseudo-GT)
    agg_b = _metrics_from_confusion_with_ignored(cm_before, ignore_classes)
    agg_a = _metrics_from_confusion_with_ignored(cm_after, ignore_classes)

    results = {
        "aggregate": {
            "before":     {"map50": round(agg_b["map50"], 4),
                           "miou":  round(agg_b["miou"],  4),
                           "pixel_acc": round(agg_b["pixel_acc"], 4),
                           "per_class_iou": [round(v,4) for v in agg_b["per_class_iou"]]},
            "after":      {"map50": round(agg_a["map50"], 4),
                           "miou":  round(agg_a["miou"],  4),
                           "pixel_acc": round(agg_a["pixel_acc"], 4),
                           "per_class_iou": [round(v,4) for v in agg_a["per_class_iou"]]},
            "map50_gain": round(agg_a["map50"] - agg_b["map50"], 4),
            "miou_gain":  round(agg_a["miou"] - agg_b["miou"], 4),
            "class_names": CLASS_NAMES,
            "num_pairs":   len(per_image_results),
            "note": "Scores measure how closely blurred-image segmentation matches clear-image segmentation (pseudo-GT)",
        },
        "per_image": per_image_results,
    }

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"  AGGREGATE  mAP50={agg_b['map50']:.4f}  mIoU={agg_b['miou']:.4f}")
    print(f"  (blurred predictions vs clear-image pseudo-GT)")
    print(f"{'='*60}")
    print(f"\nPanels  -> {out_dir}/panels/")
    print(f"Results -> {out_json}")


if __name__ == "__main__":
    main()
