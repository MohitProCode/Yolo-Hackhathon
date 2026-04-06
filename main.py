from src.utils.weights import compute_class_weights
from src.utils.seed import set_seed
from src.utils.logging import create_logger
from src.utils.config import load_config, resolve_paths
from src.utils.validation import validate_config
from src.training.trainer import Trainer
from src.training.metrics import confusion_matrix, metrics_from_confusion
from src.models.factory import build_model
from src.data.transforms import build_train_transforms, build_eval_transforms
from src.data.datasets import SegmentationDataset
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dataset_cfg(cfg: dict) -> dict:
    dataset_cfg = cfg.get("dataset", {})
    return {
        "mask_mode": dataset_cfg.get("mask_mode", "auto"),
        "color_map": dataset_cfg.get("color_map"),
        "label_map": dataset_cfg.get("label_map"),
        "max_samples": dataset_cfg.get("max_samples"),
        "train_profile": dataset_cfg.get("train_profile", "strong"),
        "patch_size": dataset_cfg.get("patch_size"),
        "crop_scale": dataset_cfg.get("crop_scale", (0.6, 1.0)),
        "mean": dataset_cfg.get("mean"),
        "std": dataset_cfg.get("std"),
        "multi_scale_sizes": dataset_cfg.get("multi_scale_sizes"),
        "image_exts": tuple(dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"])),
        "mask_exts": tuple(dataset_cfg.get("mask_exts", [".png", ".jpg", ".jpeg"])),
    }


def _build_loader(
    images_dir: str,
    masks_dir: str,
    transforms,
    dataset_cfg: dict,
    cfg: dict,
    shuffle: bool,
) -> DataLoader:
    dataset = SegmentationDataset(
        images_dir,
        masks_dir,
        transforms=transforms,
        image_exts=dataset_cfg["image_exts"],
        mask_exts=dataset_cfg["mask_exts"],
        mask_mode=dataset_cfg["mask_mode"],
        color_map=dataset_cfg["color_map"],
        label_map=dataset_cfg["label_map"],
        max_samples=dataset_cfg["max_samples"],
    )
    num_workers = int(cfg["train"]["num_workers"])
    pin_memory = bool(torch.cuda.is_available() and str(cfg["train"].get("device", "cpu")).startswith("cuda"))
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _forward_logits(model, images: torch.Tensor) -> torch.Tensor:
    logits = model(images)
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    return logits


def _tta_logits(model, images: torch.Tensor, mode: str = "flip") -> torch.Tensor:
    mode = (mode or "flip").lower()
    logits_list = []
    logits_list.append(_forward_logits(model, images))

    if mode in {"flip", "flip_rotate"}:
        imgs = torch.flip(images, dims=[3])
        logits = _forward_logits(model, imgs)
        logits_list.append(torch.flip(logits, dims=[3]))

        imgs = torch.flip(images, dims=[2])
        logits = _forward_logits(model, imgs)
        logits_list.append(torch.flip(logits, dims=[2]))

    if mode == "flip_rotate":
        imgs = torch.rot90(images, 1, dims=[2, 3])
        logits = _forward_logits(model, imgs)
        logits_list.append(torch.rot90(logits, -1, dims=[2, 3]))

        imgs = torch.rot90(images, 3, dims=[2, 3])
        logits = _forward_logits(model, imgs)
        logits_list.append(torch.rot90(logits, 1, dims=[2, 3]))

    return torch.mean(torch.stack(logits_list, dim=0), dim=0)


def train(cfg: dict, logger) -> Path:
    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"]
                          if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    image_size = tuple(cfg["train"]["image_size"])
    dataset_cfg = _dataset_cfg(cfg)
    patch_size = dataset_cfg.get("patch_size")
    if isinstance(patch_size, (list, tuple)):
        patch_size = tuple(patch_size)
    crop_scale = tuple(dataset_cfg.get("crop_scale", (0.6, 1.0)))
    mean = dataset_cfg.get("mean")
    std = dataset_cfg.get("std")
    multi_scale_sizes = dataset_cfg.get("multi_scale_sizes")
    if isinstance(multi_scale_sizes, list):
        multi_scale_sizes = [tuple(size) for size in multi_scale_sizes]
    train_loader = _build_loader(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        build_train_transforms(
            image_size,
            dataset_cfg["train_profile"],
            patch_size=patch_size,
            crop_scale=crop_scale,
            mean=mean,
            std=std,
            multi_scale_sizes=multi_scale_sizes,
        ),
        dataset_cfg,
        cfg,
        shuffle=True,
    )
    val_loader = _build_loader(
        cfg["paths"]["val_images"],
        cfg["paths"]["val_masks"],
        build_eval_transforms(image_size, mean=mean, std=std),
        dataset_cfg,
        cfg,
        shuffle=False,
    )

    loss_cfg = cfg.get("loss", {})
    if loss_cfg.get("auto_class_weights", False):
        weight_samples = loss_cfg.get("weight_samples")
        weight_ds = SegmentationDataset(
            cfg["paths"]["train_images"],
            cfg["paths"]["train_masks"],
            transforms=None,
            image_exts=dataset_cfg["image_exts"],
            mask_exts=dataset_cfg["mask_exts"],
            mask_mode=dataset_cfg["mask_mode"],
            color_map=dataset_cfg["color_map"],
            label_map=dataset_cfg["label_map"],
            max_samples=dataset_cfg["max_samples"],
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

    scheduler = None
    sched_name = cfg["train"].get("lr_scheduler", "none")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["num_epochs"])
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5)

    trainer = Trainer(model, optimizer, device, cfg,
                      logger, scheduler=scheduler)
    trainer.fit(train_loader, val_loader,
                cfg["model"]["num_classes"], cfg["paths"]["output_dir"])

    output_dir = Path(cfg["paths"]["output_dir"])
    ckpt_path = output_dir / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        fallback = output_dir / "checkpoints" / "last.pth"
        torch.save({"model": model.state_dict()}, fallback)
        logger.info(
            f"No best checkpoint found; saved last checkpoint to: {fallback}")
        ckpt_path = fallback
    return ckpt_path


@torch.no_grad()
def test(cfg: dict, checkpoint_path: Path, logger) -> float:
    device = torch.device(cfg["train"]["device"]
                          if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    image_size = tuple(cfg["train"]["image_size"])
    dataset_cfg = _dataset_cfg(cfg)
    mean = dataset_cfg.get("mean")
    std = dataset_cfg.get("std")
    test_loader = _build_loader(
        cfg["paths"]["test_images"],
        cfg["paths"]["test_masks"],
        build_eval_transforms(image_size, mean=mean, std=std),
        dataset_cfg,
        cfg,
        shuffle=False,
    )

    cm = torch.zeros((cfg["model"]["num_classes"],
                     cfg["model"]["num_classes"]), dtype=torch.int64)
    ignore_index = cfg.get("loss", {}).get("ignore_index")
    tta_cfg = cfg.get("tta", {})
    tta_enabled = tta_cfg.get("enabled", False)
    tta_mode = tta_cfg.get("mode", "flip")
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        if tta_enabled:
            logits = _tta_logits(model, images, mode=tta_mode)
        else:
            logits = _forward_logits(model, images)
        preds = torch.argmax(logits, dim=1)
        cm += confusion_matrix(
            preds.cpu(),
            masks.cpu(),
            cfg["model"]["num_classes"],
            ignore_index=ignore_index,
        )

    metrics = metrics_from_confusion(cm)
    logger.info(
        "Test "
        f"| mIoU={metrics['miou']:.4f} | mAP50={metrics['map50']:.4f} "
        f"| pixAcc={metrics['pixel_acc']:.4f} | meanAcc={metrics['mean_acc']:.4f} "
        f"| fwIoU={metrics['fw_iou']:.4f} | dice={metrics['dice']:.4f}"
    )
    test_cfg = cfg.get("test", {})
    if test_cfg.get("log_class_iou", False):
        logger.info(f"Class IoU: {metrics['per_class_iou']}")
    if test_cfg.get("log_class_dist", False):
        logger.info(f"GT dist: {metrics['gt_dist']}")
        logger.info(f"Pred dist: {metrics['pred_dist']}")
        if metrics["pred_dist"] and max(metrics["pred_dist"]) > 0.95:
            logger.warning(
                "Model predictions are collapsing to a single class.")
    return metrics["miou"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run evaluation")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to evaluate")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. cpu, cuda, cuda:0)")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), ROOT)
    if args.device:
        cfg["train"]["device"] = args.device
    warnings = validate_config(cfg, require_data_paths=True)

    output_dir = Path(cfg["paths"]["output_dir"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"
    logger = create_logger(log_file=str(log_file))
    logger.info(f"Log file: {log_file}")
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")

    if not args.train and not args.test:
        args.train = True
        args.test = True

    ckpt_path = None
    if args.train:
        ckpt_path = train(cfg, logger)
        logger.info(f"Saved best checkpoint to: {ckpt_path}")

    if args.test:
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
        if ckpt_path is None:
            ckpt_path = Path(cfg["paths"]["output_dir"]) / \
                "checkpoints" / "best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        test(cfg, ckpt_path, logger)


if __name__ == "__main__":
    main()
