import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import SegmentationDataset
from src.data.transforms import build_eval_transforms
from src.models.factory import build_model
from src.training.metrics import confusion_matrix, metrics_from_confusion
from src.utils.config import load_config, resolve_paths
from src.utils.logging import create_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    args = parser.parse_args()

    if len(args.config) != len(args.checkpoint):
        raise ValueError("Provide equal number of --config and --checkpoint arguments.")

    cfgs = []
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg_path, ckpt_path in zip(args.config, args.checkpoint):
        cfg = resolve_paths(load_config(cfg_path), os.path.dirname(cfg_path))
        cfgs.append(cfg)
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        models.append(model)

    base_cfg = cfgs[0]
    num_classes = base_cfg["model"]["num_classes"]
    for cfg in cfgs[1:]:
        if cfg["model"]["num_classes"] != num_classes:
            raise ValueError("All models must have the same num_classes for ensembling.")

    dataset_cfg = base_cfg.get("dataset", {})
    mask_mode = dataset_cfg.get("mask_mode", "auto")
    color_map = dataset_cfg.get("color_map")
    label_map = dataset_cfg.get("label_map")
    image_exts = tuple(dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]))
    mask_exts = tuple(dataset_cfg.get("mask_exts", [".png", ".jpg", ".jpeg"]))
    mean = dataset_cfg.get("mean")
    std = dataset_cfg.get("std")

    image_size = tuple(base_cfg["train"]["image_size"])
    test_ds = SegmentationDataset(
        base_cfg["paths"]["test_images"],
        base_cfg["paths"]["test_masks"],
        transforms=build_eval_transforms(image_size, mean=mean, std=std),
        image_exts=image_exts,
        mask_exts=mask_exts,
        mask_mode=mask_mode,
        color_map=color_map,
        label_map=label_map,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=base_cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=base_cfg["train"]["num_workers"],
        pin_memory=True,
    )

    logger = create_logger()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        logits_sum = None
        for model in models:
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = logits_sum / len(models)
        preds = torch.argmax(logits_avg, dim=1)
        cm += confusion_matrix(preds.cpu(), masks.cpu(), num_classes)

    metrics = metrics_from_confusion(cm)
    logger.info(
        "Ensemble "
        f"| mIoU={metrics['miou']:.4f} | mAP50={metrics['map50']:.4f} "
        f"| pixAcc={metrics['pixel_acc']:.4f} | meanAcc={metrics['mean_acc']:.4f} "
        f"| fwIoU={metrics['fw_iou']:.4f} | dice={metrics['dice']:.4f}"
    )


if __name__ == "__main__":
    main()
