import torch
import torch.nn.functional as F
from .hard_mining import hard_pixel_mining


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    if ignore_index is not None and ignore_index >= 0:
        valid = targets != ignore_index
        targets = targets.clone()
        targets[~valid] = 0
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        targets_onehot = targets_onehot * valid
    else:
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * targets_onehot).sum(dims)
    denom = probs.sum(dims) + targets_onehot.sum(dims)
    loss = 1 - (2 * intersection + eps) / (denom + eps)
    return loss.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=weight,
        ignore_index=ignore_index if ignore_index is not None else -100,
    )
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def compute_loss(
    logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    loss_cfg = cfg.get("loss", {})
    class_weights = loss_cfg.get("class_weights")
    ignore_index = loss_cfg.get("ignore_index", -100)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, device=targets.device, dtype=torch.float32)

    outputs = logits if isinstance(logits, (list, tuple)) else [logits]
    ds_weights = loss_cfg.get("deep_supervision_weights")
    if ds_weights is None:
        ds_weights = [1.0 / len(outputs)] * len(outputs)
    if len(ds_weights) != len(outputs):
        raise ValueError("deep_supervision_weights length must match number of outputs.")

    total = 0.0
    ce_val = 0.0
    dice_val = 0.0
    focal_val = 0.0
    hard_cfg = cfg.get("hard_mining", {})
    focal_gamma = loss_cfg.get("focal_gamma", 2.0)

    for idx, (out, ds_w) in enumerate(zip(outputs, ds_weights)):
        ce_map = F.cross_entropy(
            out,
            targets,
            reduction="none",
            weight=weight_tensor,
            ignore_index=ignore_index,
        )
        if hard_cfg.get("enabled", False):
            ce = hard_pixel_mining(ce_map, hard_cfg.get("top_k", 0.2))
        else:
            ce = ce_map.mean()
        dice = dice_loss(out, targets, ignore_index=ignore_index)
        focal_w_check = loss_cfg.get("focal_weight", 0.0)
        focal = focal_loss(out, targets, gamma=focal_gamma, weight=weight_tensor, ignore_index=ignore_index) if focal_w_check > 0.0 else torch.tensor(0.0, device=out.device)

        ce_w = loss_cfg.get("ce_weight", 1.0)
        dice_w = loss_cfg.get("dice_weight", 1.0)
        focal_w = loss_cfg.get("focal_weight", 0.0)
        weighted = ce_w * ce + dice_w * dice
        if focal_w > 0.0:
            weighted = weighted + focal_w * focal
        total = total + ds_w * weighted

        # Log only from the last output (highest resolution)
        if idx == len(outputs) - 1:
            ce_val = ce
            dice_val = dice
            focal_val = focal

    return total, {"ce": float(ce_val.item()), "dice": float(dice_val.item()), "focal": float(focal_val.item())}
