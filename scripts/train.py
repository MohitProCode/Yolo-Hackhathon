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
from src.data.transforms import build_train_transforms, build_eval_transforms
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config, resolve_paths
from src.utils.logging import create_logger
from src.utils.seed import set_seed
from src.utils.weights import compute_class_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    set_seed(cfg["train"]["seed"])
    logger = create_logger()

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    image_size = tuple(cfg["train"]["image_size"])
    dataset_cfg = cfg.get("dataset", {})
    mask_mode = dataset_cfg.get("mask_mode", "auto")
    color_map = dataset_cfg.get("color_map")
    label_map = dataset_cfg.get("label_map")
    max_samples = dataset_cfg.get("max_samples")
    train_profile = dataset_cfg.get("train_profile", "strong")
    patch_size = dataset_cfg.get("patch_size")
    if isinstance(patch_size, (list, tuple)):
        patch_size = tuple(patch_size)
    crop_scale = tuple(dataset_cfg.get("crop_scale", (0.6, 1.0)))
    mean = dataset_cfg.get("mean")
    std = dataset_cfg.get("std")
    multi_scale_sizes = dataset_cfg.get("multi_scale_sizes")
    if isinstance(multi_scale_sizes, list):
        multi_scale_sizes = [tuple(size) for size in multi_scale_sizes]
    image_exts = tuple(dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]))
    mask_exts = tuple(dataset_cfg.get("mask_exts", [".png", ".jpg", ".jpeg"]))
    cache = bool(dataset_cfg.get("cache", False))
    train_ds = SegmentationDataset(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        transforms=build_train_transforms(
            image_size, train_profile, patch_size=patch_size,
            crop_scale=crop_scale, mean=mean, std=std,
            multi_scale_sizes=multi_scale_sizes,
        ),
        image_exts=image_exts, mask_exts=mask_exts,
        mask_mode=mask_mode, color_map=color_map, label_map=label_map,
        max_samples=max_samples, cache=cache,
    )
    val_ds = SegmentationDataset(
        cfg["paths"]["val_images"],
        cfg["paths"]["val_masks"],
        transforms=build_eval_transforms(image_size, mean=mean, std=std),
        image_exts=image_exts, mask_exts=mask_exts,
        mask_mode=mask_mode, color_map=color_map, label_map=label_map,
        max_samples=max_samples, cache=cache,
    )

    loss_cfg = cfg.get("loss", {})
    if loss_cfg.get("auto_class_weights", False):
        weight_samples = loss_cfg.get("weight_samples")
        weight_ds = SegmentationDataset(
            cfg["paths"]["train_images"],
            cfg["paths"]["train_masks"],
            transforms=None,
            image_exts=image_exts,
            mask_exts=mask_exts,
            mask_mode=mask_mode,
            color_map=color_map,
            label_map=label_map,
            max_samples=max_samples,
        )
        weights = compute_class_weights(
            weight_ds,
            cfg["model"]["num_classes"],
            mode=loss_cfg.get("class_weighting", "median_freq"),
            ignore_index=loss_cfg.get("ignore_index"),
            max_samples=weight_samples,
            min_weight=loss_cfg.get("min_class_weight"),
            max_weight=loss_cfg.get("max_class_weight"),
        )
        cfg.setdefault("loss", {})["class_weights"] = weights
        logger.info(f"Computed class weights: {weights}")

    num_workers = cfg["train"]["num_workers"]
    # On Windows, num_workers>0 with cache=True causes each worker to hold
    # its own copy of the cache — use num_workers=0 when cache is enabled
    effective_workers = 0 if (cache and num_workers > 0) else num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=effective_workers == 0,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"] * 2,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=effective_workers == 0,
        persistent_workers=False,
    )

    scheduler = None
    sched_name = cfg["train"].get("lr_scheduler", "none")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["num_epochs"])
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    trainer = Trainer(model, optimizer, device, cfg, logger, scheduler=scheduler)
    trainer.fit(train_loader, val_loader, cfg["model"]["num_classes"], cfg["paths"]["output_dir"])


if __name__ == "__main__":
    main()
