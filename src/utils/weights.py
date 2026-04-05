from __future__ import annotations

import numpy as np


def compute_class_weights(
    dataset,
    num_classes: int,
    mode: str = "median_freq",
    ignore_index: int | None = None,
    max_samples: int | None = None,
    min_weight: float | None = None,
    max_weight: float | None = None,
) -> list[float]:
    mode = (mode or "median_freq").lower()
    counts = np.zeros(num_classes, dtype=np.int64)

    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    for idx in range(total):
        mask = dataset.get_mask(idx)
        if ignore_index is not None and ignore_index >= 0:
            mask = mask[mask != ignore_index]
        if mask.size == 0:
            continue
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)

    counts = counts.astype(np.float64)
    counts[counts == 0] = np.nan

    if mode == "inverse_freq":
        total_count = np.nansum(counts)
        weights = total_count / (num_classes * counts)
    else:
        median = np.nanmedian(counts)
        weights = median / counts

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if min_weight is not None or max_weight is not None:
        low = min_weight if min_weight is not None else -np.inf
        high = max_weight if max_weight is not None else np.inf
        weights = np.clip(weights, low, high)
    return weights.tolist()
