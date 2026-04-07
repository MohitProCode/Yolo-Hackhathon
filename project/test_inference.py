"""
Batch inference on test images.
Usage:
    python project/test_inference.py \
        --images "C:/Users/ADMIN/Downloads/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images" \
        --masks  "C:/Users/ADMIN/Downloads/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Segmentation" \
        --config  project/scratch_hardfix.yaml \
        --checkpoint project/outputs/scratch_hardfix/checkpoints/best.pth \
        --output  project/outputs/test_results \
        --device  cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.inference_utils import (
    compute_seg_metrics,
    default_palette,
    load_model_bundle,
    mask_to_color,
    overlay_mask,
    predict_mask,
    read_image_rgb,
    read_mask_raw,
)


CLASS_NAMES = ["dirt_road", "gravel", "grass", "rock", "water",
               "mud", "sand", "vegetation", "obstacle", "sky"]


def run(args):
    img_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)

    bundle = load_model_bundle(args.config, args.checkpoint, args.device)
    cfg = bundle["cfg"]
    num_classes = int(cfg["data"]["num_classes"])
    palette = default_palette(num_classes)

    image_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    mask_dir = Path(args.masks) if args.masks else None
    all_metrics = []

    for img_path in image_paths:
        image_rgb = read_image_rgb(img_path)
        pred_mask, _ = predict_mask(bundle, image_rgb)
        pred_color = mask_to_color(pred_mask, num_classes, palette)
        overlay = overlay_mask(image_rgb, pred_color, alpha=0.55)

        # Save overlay
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "overlays" / img_path.name), overlay_bgr)

        row = {"image": img_path.name}

        if mask_dir is not None:
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                gt_raw = read_mask_raw(mask_path)
                m = compute_seg_metrics(pred_mask, gt_raw, cfg)
                row.update({
                    "miou": round(float(m["miou"]), 4),
                    "map50": round(float(m["map50"]), 4),
                    "dice": round(float(m["dice"]), 4),
                    "pixel_acc": round(float(m["pixel_acc"]), 4),
                })
                print(f"{img_path.name}  mIoU={row['miou']:.4f}  mAP50={row['map50']:.4f}  Dice={row['dice']:.4f}")
            else:
                print(f"{img_path.name}  (no GT mask found)")
        else:
            print(f"{img_path.name}  predicted")

        all_metrics.append(row)

    results_path = out_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary if metrics exist
    metric_rows = [r for r in all_metrics if "miou" in r]
    if metric_rows:
        avg_miou = np.mean([r["miou"] for r in metric_rows])
        avg_map50 = np.mean([r["map50"] for r in metric_rows])
        avg_dice = np.mean([r["dice"] for r in metric_rows])
        avg_acc = np.mean([r["pixel_acc"] for r in metric_rows])
        print(f"\n{'='*50}")
        print(f"Test Summary  ({len(metric_rows)} images)")
        print(f"  Avg mIoU      : {avg_miou:.4f}")
        print(f"  Avg mAP50     : {avg_map50:.4f}")
        print(f"  Avg Dice      : {avg_dice:.4f}")
        print(f"  Avg Pixel Acc : {avg_acc:.4f}")
        print(f"{'='*50}")
        print(f"\nResults saved to: {results_path}")
        print(f"Overlays saved to: {out_dir / 'overlays'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to test Color_Images folder")
    parser.add_argument("--masks", default=None, help="Path to test Segmentation folder (optional for metrics)")
    parser.add_argument("--config", default="project/scratch_hardfix.yaml")
    parser.add_argument("--checkpoint", default="project/outputs/scratch_hardfix/checkpoints/best.pth")
    parser.add_argument("--output", default="project/outputs/test_results")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    run(args)
