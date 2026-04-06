from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class ClassAwareBatchSampler(Sampler[list[int]]):
    """
    Class-first batch sampler:
      1) Pick rare classes for guaranteed coverage in every batch.
      2) Fill remaining slots by sampling classes (inverse-frequency) then images.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        steps_per_epoch: int | None = None,
        rare_per_batch: int = 2,
        class_sample_alpha: float = 1.7,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch or math.ceil(len(dataset) / batch_size))
        self.rare_per_batch = max(0, int(rare_per_batch))
        self.class_sample_alpha = float(class_sample_alpha)

        self.class_to_indices = dataset.class_to_indices
        self.rare_classes = [c for c in dataset.rare_classes if len(self.class_to_indices.get(c, [])) > 0]

        class_counts = np.array([len(self.class_to_indices[c]) for c in range(dataset.num_classes)], dtype=np.float64)
        class_counts[class_counts <= 0] = np.nan
        probs = np.power(1.0 / class_counts, self.class_sample_alpha)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        self.class_probs = probs

    def __len__(self):
        return self.steps_per_epoch

    def _sample_from_class(self, cls: int) -> int:
        ids = self.class_to_indices.get(int(cls), [])
        if not ids:
            return random.randrange(len(self.dataset))
        return random.choice(ids)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch: list[int] = []

            # Guarantee rare classes in every batch when available.
            if self.rare_classes and self.rare_per_batch > 0:
                k = min(self.rare_per_batch, self.batch_size)
                chosen = np.random.choice(
                    self.rare_classes,
                    size=k,
                    replace=(len(self.rare_classes) < k),
                )
                for cls in chosen:
                    batch.append(self._sample_from_class(int(cls)))

            # Fill the remaining slots using class-aware sampling.
            while len(batch) < self.batch_size:
                sampled_cls = int(np.random.choice(np.arange(self.dataset.num_classes), p=self.class_probs))
                batch.append(self._sample_from_class(sampled_cls))

            yield batch


def build_weighted_random_sampler(dataset, batch_size: int, weighted_alpha: float = 1.5):
    """
    Fallback sampler: image weights are driven by rare classes present in each image.
    """
    class_counts = np.array([len(dataset.class_to_indices[c]) for c in range(dataset.num_classes)], dtype=np.float64)
    class_counts[class_counts <= 0] = np.nan
    cls_w = np.power(1.0 / class_counts, weighted_alpha)
    cls_w = np.nan_to_num(cls_w, nan=0.0, posinf=0.0, neginf=0.0)
    if cls_w.mean() > 0:
        cls_w = cls_w / cls_w.mean()

    image_weights = []
    for idx in range(len(dataset)):
        classes = list(dataset.classes_in_index(idx))
        if not classes:
            image_weights.append(0.05)
            continue
        # Aggressive oversampling of rare-class images.
        w = max(float(cls_w[c]) for c in classes)
        if classes == [dataset.background_index]:
            w *= 0.2
        image_weights.append(max(0.01, w))

    num_samples = int(math.ceil(len(dataset) / batch_size) * batch_size)
    weights_t = torch.tensor(image_weights, dtype=torch.float32)
    return WeightedRandomSampler(weights=weights_t, num_samples=num_samples, replacement=True)

