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
from src.utils.config import load_config, resolve_paths
from src.utils.logging import create_logger


def set_inference_model_defaults(cfg: dict) -> dict:
    model_cfg = cfg.setdefault("model", {})
    name = str(model_cfg.get("name", "")).lower()
    if name in {"unet", "attention_unet", "attunet", "deeplabv3plus", "deeplabv3+"}:
        if model_cfg.get("pretrained", False):
            model_cfg["use_timm_backbone"] = True
            model_cfg["pretrained"] = False
    return cfg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    cfg = set_inference_model_defaults(cfg)
    logger = create_logger()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    image_size = tuple(cfg["train"]["image_size"])
    dataset_cfg = cfg.get("dataset", {})
    mask_mode = dataset_cfg.get("mask_mode", "auto")
    color_map = dataset_cfg.get("color_map")
    label_map = dataset_cfg.get("label_map")
    max_samples = dataset_cfg.get("max_samples")
    image_exts = tuple(dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]))
    mask_exts = tuple(dataset_cfg.get("mask_exts", [".png", ".jpg", ".jpeg"]))
    mean = dataset_cfg.get("mean")
    std = dataset_cfg.get("std")
    test_ds = SegmentationDataset(
        cfg["paths"]["test_images"],
        cfg["paths"]["test_masks"],
        transforms=build_eval_transforms(image_size, mean=mean, std=std),
        image_exts=image_exts,
        mask_exts=mask_exts,
        mask_mode=mask_mode,
        color_map=color_map,
        label_map=label_map,
        max_samples=max_samples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    ignore_classes = cfg.get("eval", {}).get("ignore_classes") or []
    cm = torch.zeros((cfg["model"]["num_classes"], cfg["model"]["num_classes"]), dtype=torch.int64)
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
        cm += confusion_matrix(preds.cpu(), masks.cpu(), cfg["model"]["num_classes"])

    metrics = metrics_from_confusion(cm, ignore_classes)
    logger.info(
        "Test "
        f"| mIoU={metrics['miou']:.4f} | mAP50={metrics['map50']:.4f} "
        f"| pixAcc={metrics['pixel_acc']:.4f} | meanAcc={metrics['mean_acc']:.4f} "
        f"| fwIoU={metrics['fw_iou']:.4f} | dice={metrics['dice']:.4f}"
    )
    if cfg.get("test", {}).get("log_class_iou", False):
        logger.info(f"Class IoU: {metrics['per_class_iou']}")
    if cfg.get("test", {}).get("log_class_dist", False):
        logger.info(f"GT dist: {metrics['gt_dist']}")
        logger.info(f"Pred dist: {metrics['pred_dist']}")


if __name__ == "__main__":
    main()
