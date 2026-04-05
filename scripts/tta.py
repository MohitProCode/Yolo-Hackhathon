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
from src.tta.adapt import adapt_model
from src.training.metrics import mean_iou
from src.utils.config import load_config, resolve_paths
from src.utils.logging import create_logger


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    miou_scores = []
    for images, masks in loader:
        images = images.to(next(model.parameters()).device)
        masks = masks.to(images.device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        miou_scores.append(mean_iou(preds.cpu(), masks.cpu(), num_classes))
    return float(sum(miou_scores) / max(1, len(miou_scores)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    logger = create_logger()

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])

    image_size = tuple(cfg["train"]["image_size"])
    test_ds = SegmentationDataset(
        cfg["paths"]["test_images"],
        cfg["paths"]["test_masks"],
        transforms=build_eval_transforms(image_size),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    adapt_model(model, test_loader, cfg, device)
    miou = evaluate(model, test_loader, cfg["model"]["num_classes"])
    logger.info(f"TTA mIoU={miou:.4f}")


if __name__ == "__main__":
    main()
