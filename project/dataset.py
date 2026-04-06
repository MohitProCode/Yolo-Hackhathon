from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


@dataclass
class DatasetStats:
    pixel_counts: np.ndarray
    image_class_sets: list[set[int]]
    class_to_indices: dict[int, list[int]]
    present_classes: set[int]
    rare_classes: list[int]


def _build_label_map(label_map: list[int] | dict[int, int] | None) -> dict[int, int] | None:
    if label_map is None:
        return None
    if isinstance(label_map, dict):
        return {int(k): int(v) for k, v in label_map.items()}
    if isinstance(label_map, list):
        return {int(v): i for i, v in enumerate(label_map)}
    raise ValueError("label_map must be list, dict, or null.")


class MultiClassSegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, cfg: dict, train: bool = True):
        self.cfg = cfg
        self.train = train
        data_cfg = cfg["data"]
        self.num_classes = int(data_cfg["num_classes"])
        self.ignore_index = int(data_cfg["ignore_index"])
        self.background_index = int(data_cfg["background_index"])
        self.label_map = _build_label_map(data_cfg.get("label_map"))
        self.object_crop_prob = float(data_cfg["object_crop_prob"])
        self.object_crop_size = tuple(int(x) for x in data_cfg["object_crop_size"])

        self.images = self._list_files(images_dir, tuple(data_cfg["image_exts"]))
        self.masks = self._align_masks(masks_dir, tuple(data_cfg["mask_exts"]))
        if not self.images:
            raise RuntimeError(f"No images found in {images_dir}")

        self.transforms = self._build_transforms(train=train)
        self.stats = self._scan_class_statistics(cfg["sampler"]["rare_quantile"])

    def _list_files(self, directory: str, exts: tuple[str, ...]) -> list[Path]:
        out: list[Path] = []
        root = Path(directory)
        for ext in exts:
            out.extend(root.glob(f"*{ext}"))
        return sorted(out)

    def _align_masks(self, masks_dir: str, exts: tuple[str, ...]) -> list[Path]:
        root = Path(masks_dir)
        mask_map: dict[str, Path] = {}
        for ext in exts:
            for p in root.glob(f"*{ext}"):
                mask_map[p.stem] = p
        aligned = []
        for img in self.images:
            m = mask_map.get(img.stem)
            if m is None:
                raise FileNotFoundError(f"Mask missing for image stem '{img.stem}'")
            aligned.append(m)
        return aligned

    def _build_transforms(self, train: bool):
        h, w = map(int, self.cfg["data"]["image_size"])
        mean = self.cfg["data"]["mean"]
        std = self.cfg["data"]["std"]
        if train:
            return A.Compose(
                [
                    # Required augmentation: RandomResizedCrop + flips/jitter/blur.
                    A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.6),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        return A.Compose([A.Resize(h, w), A.Normalize(mean=mean, std=std), ToTensorV2()])

    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_raw_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def _map_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.label_map is None:
            mapped = mask.astype(np.int64)
        else:
            mapped = np.full(mask.shape, self.ignore_index, dtype=np.int64)
            for raw_value, cls in self.label_map.items():
                mapped[mask == raw_value] = cls
        self._validate_labels(mapped)
        return mapped

    def _validate_labels(self, mask: np.ndarray) -> None:
        invalid = (mask != self.ignore_index) & ((mask < 0) | (mask >= self.num_classes))
        if np.any(invalid):
            bad_values = np.unique(mask[invalid])[:10].tolist()
            raise ValueError(f"Invalid class indices found in mask: {bad_values}")

    def _scan_class_statistics(self, rare_quantile: float) -> DatasetStats:
        pixel_counts = np.zeros(self.num_classes, dtype=np.int64)
        image_class_sets: list[set[int]] = []
        class_to_indices = {c: [] for c in range(self.num_classes)}

        for idx, mask_path in enumerate(self.masks):
            raw_mask = self._read_raw_mask(mask_path)
            mask = self._map_mask(raw_mask)
            valid = mask != self.ignore_index
            cls_pixels = mask[valid]
            binc = np.bincount(cls_pixels, minlength=self.num_classes)
            pixel_counts += binc

            cls_set = set(np.where(binc > 0)[0].tolist())
            image_class_sets.append(cls_set)
            for c in cls_set:
                class_to_indices[c].append(idx)

        present_classes = set(np.where(pixel_counts > 0)[0].tolist())
        present_counts = pixel_counts[list(present_classes)] if present_classes else np.array([], dtype=np.int64)
        if len(present_counts) > 0:
            thresh = float(np.quantile(present_counts, rare_quantile))
            rare_classes = [c for c in sorted(present_classes) if pixel_counts[c] <= thresh]
        else:
            rare_classes = []

        if self.background_index in rare_classes and len(rare_classes) > 1:
            rare_classes = [c for c in rare_classes if c != self.background_index]

        return DatasetStats(
            pixel_counts=pixel_counts,
            image_class_sets=image_class_sets,
            class_to_indices=class_to_indices,
            present_classes=present_classes,
            rare_classes=rare_classes,
        )

    def _safe_crop(self, image: np.ndarray, mask: np.ndarray, cy: int, cx: int, h: int, w: int):
        ih, iw = mask.shape[:2]
        y1 = max(0, min(cy - h // 2, ih - h))
        x1 = max(0, min(cx - w // 2, iw - w))
        y2 = min(ih, y1 + h)
        x2 = min(iw, x1 + w)
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _object_focused_crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ch, cw = self.object_crop_size
        valid = mask != self.ignore_index
        rare_candidates = np.isin(mask, self.stats.rare_classes) & valid
        fg_candidates = (mask != self.background_index) & valid
        if np.any(rare_candidates):
            ys, xs = np.where(rare_candidates)
        elif np.any(fg_candidates):
            ys, xs = np.where(fg_candidates)
        elif np.any(valid):
            ys, xs = np.where(valid)
        else:
            return image, mask

        pick = np.random.randint(0, len(ys))
        cy, cx = int(ys[pick]), int(xs[pick])
        return self._safe_crop(image, mask, cy, cx, ch, cw)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self._read_image(self.images[idx])
        raw_mask = self._read_raw_mask(self.masks[idx])
        mask = self._map_mask(raw_mask)

        if self.train and np.random.rand() < self.object_crop_prob:
            image, mask = self._object_focused_crop(image, mask)

        out = self.transforms(image=image, mask=mask)
        image_t = out["image"]
        mask_t = out["mask"].long()
        return image_t, mask_t

    def classes_in_index(self, idx: int) -> set[int]:
        return self.stats.image_class_sets[idx]

    @property
    def pixel_counts(self) -> np.ndarray:
        return self.stats.pixel_counts

    @property
    def class_to_indices(self) -> dict[int, list[int]]:
        return self.stats.class_to_indices

    @property
    def rare_classes(self) -> list[int]:
        return self.stats.rare_classes

    @property
    def present_classes(self) -> set[int]:
        return self.stats.present_classes

