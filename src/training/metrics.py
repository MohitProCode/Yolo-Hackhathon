import torch


def _confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]
    indices = num_classes * targets + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def _metrics_from_confusion(cm: torch.Tensor, ignore_classes: list[int] | None = None) -> dict:
    cm = cm.float()
    tp = torch.diag(cm)
    row_sum = cm.sum(dim=1)
    col_sum = cm.sum(dim=0)
    fp = col_sum - tp
    fn = row_sum - tp
    denom = tp + fp + fn

    # Exclude classes with fewer than 10 GT pixels or explicitly ignored
    valid = row_sum >= 10
    if ignore_classes:
        for c in ignore_classes:
            if 0 <= c < valid.shape[0]:
                valid[c] = False
    iou = torch.zeros_like(tp)
    dice = torch.zeros_like(tp)
    acc = torch.zeros_like(tp)

    iou[valid] = tp[valid] / torch.clamp(denom[valid], min=1.0)
    dice[valid] = (2 * tp[valid]) / torch.clamp(2 * tp[valid] + fp[valid] + fn[valid], min=1.0)
    acc[valid] = tp[valid] / torch.clamp(row_sum[valid], min=1.0)

    total = cm.sum().clamp(min=1.0)
    pixel_acc = tp.sum() / total
    mean_acc = acc[valid].mean() if valid.any() else torch.tensor(0.0, device=cm.device)
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=cm.device)
    fw_iou = (row_sum[valid] / total * iou[valid]).sum() if valid.any() else torch.tensor(0.0, device=cm.device)
    # mAP50: fraction of classes with IoU >= 0.5, counting only classes present in GT
    map50 = (iou[valid] >= 0.5).float().mean() if valid.any() else torch.tensor(0.0, device=cm.device)
    mean_dice = dice[valid].mean() if valid.any() else torch.tensor(0.0, device=cm.device)

    gt_dist = (row_sum / total).tolist()
    pred_dist = (col_sum / total).tolist()

    return {
        "pixel_acc": float(pixel_acc.item()),
        "mean_acc": float(mean_acc.item()),
        "miou": float(miou.item()),
        "fw_iou": float(fw_iou.item()),
        "dice": float(mean_dice.item()),
        "map50": float(map50.item()),
        "per_class_iou": iou.tolist(),
        "per_class_dice": dice.tolist(),
        "per_class_acc": acc.tolist(),
        "gt_dist": gt_dist,
        "pred_dist": pred_dist,
    }


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    return _confusion_matrix(preds, targets, num_classes, ignore_index)


def metrics_from_confusion(cm: torch.Tensor, ignore_classes: list[int] | None = None) -> dict:
    return _metrics_from_confusion(cm, ignore_classes)


def segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    ignore_classes: list[int] | None = None,
) -> dict:
    cm = _confusion_matrix(preds, targets, num_classes, ignore_index)
    return _metrics_from_confusion(cm, ignore_classes)


def mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    return float(segmentation_metrics(preds, targets, num_classes, ignore_index)["miou"])
