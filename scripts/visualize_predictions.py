import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import SegmentationDataset
from src.data.transforms import build_eval_transforms
from src.models.factory import build_model
from src.utils.config import load_config, resolve_paths
from src.utils.visualization import colorize_mask, overlay_image


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--out-dir", default="outputs/vis")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    dataset_cfg = cfg.get("dataset", {})
    mask_mode = dataset_cfg.get("mask_mode", "auto")
    color_map = dataset_cfg.get("color_map")
    label_map = dataset_cfg.get("label_map")
    image_exts = tuple(dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]))
    mask_exts = tuple(dataset_cfg.get("mask_exts", [".png", ".jpg", ".jpeg"]))
    mean = dataset_cfg.get("mean")
    std = dataset_cfg.get("std")

    split_key = "train" if args.split == "train" else "val" if args.split == "val" else "test"
    images_dir = cfg["paths"][f"{split_key}_images"]
    masks_dir = cfg["paths"][f"{split_key}_masks"]

    image_size = tuple(cfg["train"]["image_size"])
    ds = SegmentationDataset(
        images_dir,
        masks_dir,
        transforms=build_eval_transforms(image_size, mean=mean, std=std),
        image_exts=image_exts,
        mask_exts=mask_exts,
        mask_mode=mask_mode,
        color_map=color_map,
        label_map=label_map,
    )

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = cfg["model"]["num_classes"]

    for i in range(min(args.num_samples, len(ds))):
        image_path = ds.image_paths[i]
        raw_image = ds._load_image(image_path)
        image, mask = ds[i]
        image = image.unsqueeze(0).to(device)
        logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        mask = mask.cpu().numpy()

        pred_color = colorize_mask(pred, num_classes)
        gt_color = colorize_mask(mask, num_classes)
        overlay_pred = overlay_image(raw_image, pred_color, alpha=0.5)
        overlay_gt = overlay_image(raw_image, gt_color, alpha=0.5)

        import cv2

        cv2.imwrite(str(out_dir / f"{i:03d}_pred.png"), cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{i:03d}_gt.png"), cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))
        print(f"Saved {i:03d}_pred.png and {i:03d}_gt.png")


if __name__ == "__main__":
    main()
