import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model
from src.training.losses import compute_loss
from src.utils.config import load_config, resolve_paths


def parse_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def try_config(model, device, cfg, batch_size: int, size: int, use_amp: bool) -> bool:
    in_channels = cfg["model"]["in_channels"]
    num_classes = cfg["model"]["num_classes"]
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        images = torch.randn(batch_size, in_channels, size, size, device=device)
        targets = torch.randint(0, num_classes, (batch_size, size, size), device=device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, _ = compute_loss(logits, targets, cfg)
        loss.backward()
        return True
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return False
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", default="512,448,384,320,256")
    parser.add_argument("--batches", default="8,6,4,2,1")
    parser.add_argument("--out-config", default="configs/auto_3060.yaml")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available. Auto-tune requires a CUDA GPU.")

    model = build_model(cfg).to(device)
    use_amp = bool(cfg.get("train", {}).get("mixed_precision", False))

    sizes = parse_list(args.sizes)
    batches = parse_list(args.batches)

    best = None
    for size in sizes:
        for batch in batches:
            ok = try_config(model, device, cfg, batch, size, use_amp)
            if ok:
                best = (size, batch)
                break
        if best is not None:
            break

    if best is None:
        raise RuntimeError("No valid (image_size, batch_size) found. Try smaller sizes/batches.")

    size, batch = best
    train_cfg = cfg.setdefault("train", {})
    dataset_cfg = cfg.setdefault("dataset", {})
    train_cfg["image_size"] = [size, size]
    dataset_cfg["patch_size"] = [size, size]

    current_bs = train_cfg.get("batch_size", 1)
    current_accum = int(train_cfg.get("grad_accum", 1))
    target_effective = int(train_cfg.get("target_effective_batch", current_bs * current_accum))
    max_accum = int(train_cfg.get("max_grad_accum", 16))
    new_accum = max(1, int(math.ceil(target_effective / batch)))
    new_accum = min(new_accum, max_accum)

    train_cfg["batch_size"] = batch
    train_cfg["grad_accum"] = new_accum

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Auto-tuned config saved to: {out_path}")
    print(f"image_size: {size}x{size}, batch_size: {batch}, grad_accum: {new_accum}")


if __name__ == "__main__":
    main()
