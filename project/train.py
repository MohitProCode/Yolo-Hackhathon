from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from dataset import MultiClassSegDataset
from loss import HybridSegLoss
from model import build_model
from sampler import ClassAwareBatchSampler, build_weighted_random_sampler
from utils import (
    confusion_matrix,
    create_logger,
    inverse_log_class_weights,
    metrics_from_confusion,
    seed_everything,
)


def build_train_loader(dataset: MultiClassSegDataset, cfg: dict) -> DataLoader:
    train_cfg = cfg["train"]
    sampler_cfg = cfg["sampler"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])

    strategy = str(sampler_cfg["strategy"]).lower()
    steps_per_epoch = sampler_cfg["steps_per_epoch"]
    if steps_per_epoch is None:
        steps_per_epoch = int(math.ceil(len(dataset) / batch_size))

    if strategy == "class_aware":
        batch_sampler = ClassAwareBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            rare_per_batch=int(sampler_cfg["rare_per_batch"]),
            class_sample_alpha=float(sampler_cfg["class_sample_alpha"]),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    if strategy == "weighted":
        sampler = build_weighted_random_sampler(
            dataset=dataset,
            batch_size=batch_size,
            weighted_alpha=float(sampler_cfg["weighted_alpha"]),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    cfg: dict,
    logger,
    epoch: int,
    rare_classes: list[int],
    required_classes: set[int],
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    seen_classes_epoch: set[int] = set()
    gt_pixel_counts = np.zeros(cfg["data"]["num_classes"], dtype=np.int64)

    for step, (images, masks) in enumerate(tqdm(loader, desc=f"train-{epoch}", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=bool(cfg["train"]["amp"] and device.type == "cuda"),
        ):
            main_logits, aux_logits = model(images)
            loss, loss_stats = criterion(main_logits, masks, aux_logits=aux_logits)

        scaler.scale(loss).backward()
        if cfg["train"]["grad_clip_norm"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(cfg["train"]["grad_clip_norm"])
            )
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_steps += 1

        mask_np = masks.detach().cpu().numpy()
        ignore_index = int(cfg["data"]["ignore_index"])
        valid = mask_np != ignore_index
        valid_vals = mask_np[valid]
        if valid_vals.size > 0:
            seen_classes_epoch.update(np.unique(valid_vals).tolist())
            gt_pixel_counts += np.bincount(
                valid_vals.reshape(-1),
                minlength=int(cfg["data"]["num_classes"]),
            )

        # Mandatory debugging assertion: each batch must include rare classes.
        if cfg["debug"]["assert_rare_in_every_batch"] and rare_classes:
            batch_cls = set(np.unique(valid_vals).tolist()) if valid_vals.size > 0 else set()
            if not (batch_cls & set(rare_classes)):
                raise RuntimeError(
                    f"Rare-class coverage failed at epoch={epoch}, step={step}. "
                    f"Batch classes={sorted(batch_cls)} rare={rare_classes}"
                )

        if step % int(cfg["train"]["log_interval"]) == 0:
            logger.info(
                "Epoch %03d Step %04d | loss=%.4f ce=%.4f dice=%.4f focal=%.4f kl=%.4f",
                epoch,
                step,
                float(loss.item()),
                loss_stats["ce"],
                loss_stats["dice"],
                loss_stats["focal"],
                loss_stats["kl"],
            )

    if cfg["debug"]["assert_all_dataset_classes_seen_per_epoch"]:
        missing = sorted(list(required_classes - seen_classes_epoch))
        if missing:
            raise RuntimeError(
                f"Epoch {epoch}: missing dataset classes in training batches: {missing}. "
                "Sampling/cropping is not covering all classes."
            )

    gt_dist = gt_pixel_counts / max(1, gt_pixel_counts.sum())
    logger.info(
        "Epoch %03d TRAIN class distribution (GT): %s",
        epoch,
        [round(float(x), 6) for x in gt_dist.tolist()],
    )
    return total_loss / max(1, total_steps)


@torch.no_grad()
def validate(model, loader, device, cfg: dict):
    model.eval()
    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"]["ignore_index"])
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        main_logits, _ = model(images)
        preds = torch.argmax(main_logits, dim=1)
        cm += confusion_matrix(preds.cpu(), masks.cpu(), num_classes, ignore_index=ignore_index)
    return metrics_from_confusion(cm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(str(output_dir))
    seed_everything(int(cfg["seed"]))

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() and "cuda" in cfg["train"]["device"] else "cpu"
    )
    logger.info("Using device: %s", device)

    train_ds = MultiClassSegDataset(
        cfg["paths"]["train_images"],
        cfg["paths"]["train_masks"],
        cfg=cfg,
        train=True,
    )
    val_ds = MultiClassSegDataset(
        cfg["paths"]["val_images"],
        cfg["paths"]["val_masks"],
        cfg=cfg,
        train=False,
    )

    logger.info("Train size=%d | Val size=%d", len(train_ds), len(val_ds))
    logger.info("Present classes in train set: %s", sorted(list(train_ds.present_classes)))
    logger.info("Rare classes (oversampled): %s", train_ds.rare_classes)
    logger.info(
        "Global train class pixel distribution: %s",
        [
            round(x, 6)
            for x in (train_ds.pixel_counts / max(1, train_ds.pixel_counts.sum())).tolist()
        ],
    )

    class_weights_np = inverse_log_class_weights(train_ds.pixel_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    logger.info("Class weights (inverse-log): %s", [round(float(w), 4) for w in class_weights_np.tolist()])

    train_loader = build_train_loader(train_ds, cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["train"]["num_workers"]) > 0,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"] and device.type == "cuda"))
    criterion = HybridSegLoss(cfg=cfg, class_weights=class_weights).to(device)

    best_miou = -1.0
    epochs = int(cfg["train"]["epochs"])
    required_classes = set(train_ds.present_classes)
    rare_classes = train_ds.rare_classes

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            cfg=cfg,
            logger=logger,
            epoch=epoch,
            rare_classes=rare_classes,
            required_classes=required_classes,
        )
        val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step()

        logger.info(
            "Epoch %03d | loss=%.4f | mIoU=%.4f | mAP50=%.4f | dice=%.4f | pixAcc=%.4f",
            epoch,
            train_loss,
            val_metrics["miou"],
            val_metrics["map50"],
            val_metrics["dice"],
            val_metrics["pixel_acc"],
        )
        logger.info("Epoch %03d Per-class IoU: %s", epoch, [round(float(v), 4) for v in val_metrics["per_class_iou"]])
        logger.info(
            "Epoch %03d GT dist: %s",
            epoch,
            [round(float(v), 6) for v in val_metrics["gt_dist"]],
        )
        logger.info(
            "Epoch %03d Pred dist: %s",
            epoch,
            [round(float(v), 6) for v in val_metrics["pred_dist"]],
        )

        # Mandatory debugging: detect predictions on classes absent in GT.
        gt_dist = np.array(val_metrics["gt_dist"], dtype=np.float64)
        pred_dist = np.array(val_metrics["pred_dist"], dtype=np.float64)
        pred_only = np.where(
            (gt_dist <= 1e-12)
            & (pred_dist >= float(cfg["debug"]["pred_only_warn_threshold"]))
        )[0].tolist()
        if pred_only:
            logger.warning(
                "Epoch %03d predicted classes absent in GT (collapse risk): %s",
                epoch,
                pred_only,
            )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": val_metrics,
            "config": cfg,
        }
        torch.save(state, ckpt_dir / "last.pth")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(state, ckpt_dir / "best.pth")
            logger.info("Epoch %03d new best checkpoint: mIoU=%.4f", epoch, best_miou)

    logger.info("Training complete. Best mIoU=%.4f", best_miou)


if __name__ == "__main__":
    main()
