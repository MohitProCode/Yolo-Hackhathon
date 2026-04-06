from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        ignore_index=ignore_index,
    )


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    targets = targets.clone()
    valid = targets != ignore_index
    targets[~valid] = 0

    onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1).float()

    probs = probs * valid_mask
    onehot = onehot * valid_mask

    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)

    # Average only classes present in GT.
    gt_pixels = onehot.sum(dims)
    present = gt_pixels > 0
    if present.any():
        return 1.0 - dice[present].mean()
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    ignore_index: int,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def distribution_alignment_kl(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    temp: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    valid = targets != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    probs = F.softmax(logits / temp, dim=1)
    pred_hist = probs.permute(0, 2, 3, 1)[valid].mean(dim=0)  # [C]
    pred_hist = pred_hist / pred_hist.sum().clamp(min=eps)

    gt = targets.clone()
    gt[~valid] = 0
    gt_onehot = F.one_hot(gt, num_classes=num_classes).float()
    gt_hist = gt_onehot[valid].mean(dim=0)
    gt_hist = gt_hist / gt_hist.sum().clamp(min=eps)

    return F.kl_div((pred_hist + eps).log(), gt_hist + eps, reduction="batchmean")


class HybridSegLoss(nn.Module):
    """
    Loss = 0.5 * WeightedCE + 0.3 * Dice + 0.2 * Focal + lambda * KL(pred_dist || gt_dist)
    """

    def __init__(self, cfg: dict, class_weights: torch.Tensor):
        super().__init__()
        self.ce_w = float(cfg["loss"]["ce_weight"])
        self.dice_w = float(cfg["loss"]["dice_weight"])
        self.focal_w = float(cfg["loss"]["focal_weight"])
        self.kl_w = float(cfg["loss"]["dist_kl_weight"])
        self.gamma = float(cfg["loss"]["focal_gamma"])
        self.temp = float(cfg["loss"]["dist_temp"])
        self.num_classes = int(cfg["data"]["num_classes"])
        self.ignore_index = int(cfg["data"]["ignore_index"])
        self.aux_weight = float(cfg["model"].get("aux_weight", 0.4))
        self.register_buffer("class_weights", class_weights)

    def _single(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        ce = weighted_ce_loss(logits, targets, self.class_weights, self.ignore_index)
        dice = multiclass_dice_loss(logits, targets, self.num_classes, self.ignore_index)
        focal = focal_loss(logits, targets, self.class_weights, self.gamma, self.ignore_index)
        kl = distribution_alignment_kl(
            logits, targets, self.num_classes, self.ignore_index, temp=self.temp
        )
        total = self.ce_w * ce + self.dice_w * dice + self.focal_w * focal + self.kl_w * kl
        return {"total": total, "ce": ce, "dice": dice, "focal": focal, "kl": kl}

    def forward(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        main = self._single(main_logits, targets)
        total = main["total"]
        if aux_logits is not None:
            aux = self._single(aux_logits, targets)
            total = total + self.aux_weight * aux["total"]
        return total, {k: float(v.item()) for k, v in main.items()}

