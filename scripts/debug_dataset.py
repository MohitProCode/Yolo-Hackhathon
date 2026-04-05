import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import SegmentationDataset
from src.utils.config import load_config, resolve_paths
from src.utils.visualization import colorize_mask, overlay_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--save-dir", default="outputs/debug")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    dataset_cfg = cfg.get("dataset", {})
    mask_mode = dataset_cfg.get("mask_mode", "auto")
    color_map = dataset_cfg.get("color_map")
    label_map = dataset_cfg.get("label_map")
    image_exts = tuple(dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]))
    mask_exts = tuple(dataset_cfg.get("mask_exts", [".png", ".jpg", ".jpeg"]))

    split_key = "train" if args.split == "train" else "val" if args.split == "val" else "test"
    images_dir = cfg["paths"][f"{split_key}_images"]
    masks_dir = cfg["paths"][f"{split_key}_masks"]

    ds = SegmentationDataset(
        images_dir,
        masks_dir,
        transforms=None,
        image_exts=image_exts,
        mask_exts=mask_exts,
        mask_mode=mask_mode,
        color_map=color_map,
        label_map=label_map,
    )

    max_samples = min(args.max_samples, len(ds))
    counts = None
    unique_values = set()
    for i in range(max_samples):
        mask = ds.get_mask(i)
        unique_values.update(np.unique(mask).tolist())
        num_classes = cfg["model"]["num_classes"]
        counts_i = np.bincount(mask.reshape(-1), minlength=num_classes)
        counts = counts_i if counts is None else counts + counts_i

    print(f"Checked {max_samples} samples from {args.split} split")
    print(f"Unique mask values: {sorted(int(v) for v in unique_values)}")
    if counts is not None:
        total = counts.sum()
        dist = (counts / max(1, total)).tolist()
        print(f"Class distribution: {dist}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if max_samples > 0:
        img = ds._load_image(ds.image_paths[0])
        mask = ds.get_mask(0)
        mask_color = colorize_mask(mask, cfg["model"]["num_classes"])
        overlay = overlay_image(img, mask_color, alpha=0.5)
        out = save_dir / f"sample_{args.split}.png"
        import cv2

        cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay to {out}")


if __name__ == "__main__":
    main()
