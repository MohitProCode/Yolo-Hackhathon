from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from .losses import compute_loss
from .consistency import consistency_loss
from .metrics import confusion_matrix, metrics_from_confusion


class Trainer:
    def __init__(self, model, optimizer, device, cfg, logger, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.scheduler = scheduler
        self.best_miou = 0.0
        self.consistency_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        train_cfg = cfg.get("train", {})
        self.use_amp = bool(train_cfg.get("mixed_precision", False) and self.device.type == "cuda")
        self.grad_accum = int(train_cfg.get("grad_accum", 1))
        if self.grad_accum < 1:
            self.grad_accum = 1
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _apply_consistency_aug(self, images: torch.Tensor) -> torch.Tensor:
        # Denormalize → augment → renormalize to avoid corrupting normalized tensors
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        imgs_01 = (images * std + mean).clamp(0, 1)
        augmented = torch.stack([self.consistency_aug(img) for img in imgs_01])
        return (augmented - mean) / std

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        for step, (images, masks) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss, _ = compute_loss(logits, masks, self.cfg)
                logits_main = logits[-1] if isinstance(logits, (list, tuple)) else logits

                cons_cfg = self.cfg.get("consistency", {})
                if cons_cfg.get("enabled", False):
                    images_aug = self._apply_consistency_aug(images)
                    logits_aug = self.model(images_aug)
                    logits_aug_main = logits_aug[-1] if isinstance(logits_aug, (list, tuple)) else logits_aug
                    cons_loss = consistency_loss(logits_main, logits_aug_main)
                    loss = loss + cons_cfg.get("weight", 0.2) * cons_loss

            total_loss += loss.item()
            loss = loss / self.grad_accum

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # Handle leftover grads if dataset size not divisible by grad_accum
        if len(loader) % self.grad_accum != 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader: DataLoader, num_classes: int) -> dict:
        self.model.eval()
        ignore_classes = self.cfg.get("eval", {}).get("ignore_classes") or []
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for images, masks in tqdm(loader, desc="val", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)
            logits = self.model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]
            preds = torch.argmax(logits, dim=1)
            cm += confusion_matrix(preds.cpu(), masks.cpu(), num_classes)
        return metrics_from_confusion(cm, ignore_classes)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_classes: int, output_dir: str) -> None:
        output_path = Path(output_dir) / "checkpoints"
        output_path.mkdir(parents=True, exist_ok=True)

        epochs = self.cfg["train"]["num_epochs"]
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            metrics = self.validate(val_loader, num_classes)
            self.logger.info(
                "Epoch "
                f"{epoch:03d} | loss={train_loss:.4f} | mIoU={metrics['miou']:.4f} "
                f"| mAP50={metrics['map50']:.4f} | pixAcc={metrics['pixel_acc']:.4f} "
                f"| meanAcc={metrics['mean_acc']:.4f} | fwIoU={metrics['fw_iou']:.4f} "
                f"| dice={metrics['dice']:.4f} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )
            train_cfg = self.cfg.get("train", {})
            if train_cfg.get("log_class_iou", False):
                self.logger.info(f"Class IoU: {metrics['per_class_iou']}")
            if train_cfg.get("log_class_dist", False):
                self.logger.info(f"GT dist: {metrics['gt_dist']}")
                self.logger.info(f"Pred dist: {metrics['pred_dist']}")
                if metrics["pred_dist"] and max(metrics["pred_dist"]) > 0.95:
                    self.logger.warning("Model predictions are collapsing to a single class.")

            if metrics["miou"] > self.best_miou:
                self.best_miou = metrics["miou"]
                ckpt_path = output_path / "best.pth"
                torch.save({"model": self.model.state_dict()}, ckpt_path)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(metrics["miou"])
                    else:
                        self.scheduler.step()
