from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "paths": {
        "train_images": "",
        "train_masks": "",
        "val_images": "",
        "val_masks": "",
        "output_dir": "outputs/seg_hardfix",
    },
    "data": {
        "num_classes": 10,
        "ignore_index": -100,
        "background_index": 0,
        "label_map": None,  # Optional: list or dict mapping raw mask values -> [0..num_classes-1]
        "image_exts": [".png", ".jpg", ".jpeg"],
        "mask_exts": [".png", ".jpg", ".jpeg"],
        "image_size": [384, 384],
        "object_crop_size": [448, 448],
        "object_crop_prob": 0.9,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "sampler": {
        "strategy": "class_aware",  # class_aware | weighted | random
        "rare_quantile": 0.35,
        "rare_per_batch": 2,
        "class_sample_alpha": 1.7,
        "weighted_alpha": 1.5,
        "steps_per_epoch": None,  # If None uses ceil(len(train_set)/batch_size)
    },
    "model": {
        "name": "unet_scratch",
        "in_channels": 3,
        "base_channels": 32,
        "deep_supervision": True,
        "aux_weight": 0.4,
        # Hard requirement: no pretrained backbones.
        "pretrained": False,
    },
    "loss": {
        "ce_weight": 0.5,
        "dice_weight": 0.3,
        "focal_weight": 0.2,
        "focal_gamma": 2.0,
        "dist_kl_weight": 0.08,
        "dist_temp": 1.0,
    },
    "train": {
        "device": "cuda",
        "batch_size": 8,
        "num_workers": 4,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "amp": True,
        "grad_clip_norm": 1.0,
        "log_interval": 20,
    },
    "debug": {
        "assert_rare_in_every_batch": True,
        "assert_all_dataset_classes_seen_per_epoch": True,
        "pred_only_warn_threshold": 0.01,
    },
}


def _merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _resolve_path(path_str: str, base: Path) -> str:
    path = Path(path_str)
    return str(path if path.is_absolute() else (base / path).resolve())


def load_config(path: str) -> dict:
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    _merge(cfg, user_cfg)

    # Resolve filesystem paths relative to config location.
    base = cfg_path.parent
    for key, value in list(cfg["paths"].items()):
        if isinstance(value, str) and value:
            cfg["paths"][key] = _resolve_path(value, base)

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    num_classes = int(cfg["data"]["num_classes"])
    if num_classes <= 1:
        raise ValueError("data.num_classes must be > 1.")

    ignore_index = int(cfg["data"]["ignore_index"])
    if ignore_index >= num_classes:
        raise ValueError("data.ignore_index must be < data.num_classes or negative.")

    if cfg["model"].get("pretrained", False):
        raise ValueError("Pretrained models are disabled by requirement. Set model.pretrained=false.")

    required_paths = ("train_images", "train_masks", "val_images", "val_masks")
    for key in required_paths:
        p = Path(cfg["paths"][key])
        if not p.exists():
            raise FileNotFoundError(f"paths.{key} does not exist: {p}")

    if cfg["train"]["batch_size"] <= 0:
        raise ValueError("train.batch_size must be > 0.")
    if cfg["train"]["epochs"] <= 0:
        raise ValueError("train.epochs must be > 0.")
    if cfg["train"]["lr"] <= 0:
        raise ValueError("train.lr must be > 0.")

