from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.inference_utils import (  # noqa: E402
    default_palette,
    load_model_bundle,
    map_mask_values,
    mask_to_color,
    overlay_mask,
    predict_mask,
    read_image_rgb,
    read_mask_raw,
)
from project.utils import confusion_matrix, metrics_from_confusion  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PRED_ONLY_WARN_THRESHOLD = 0.01
PAIRING_CHOICES = ("auto", "stem", "normalized", "numeric", "hash", "index")

PAIRING_STOPWORDS = {
    "before",
    "after",
    "blur",
    "blurred",
    "deblur",
    "deblurred",
    "clean",
    "clear",
    "enhanced",
    "restored",
    "output",
    "result",
    "img",
    "image",
    "frame",
}


def _class_names(cfg: dict) -> list[str]:
    names = cfg["data"].get("class_names")
    num_classes = int(cfg["data"]["num_classes"])
    if isinstance(names, list) and len(names) == num_classes:
        return [str(x) for x in names]
    return [f"class_{i}" for i in range(num_classes)]


def build_stem_map_from_dir(directory: str | Path, exts: Iterable[str] = IMAGE_EXTS) -> dict[str, Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    ext_set = {str(e).lower() for e in exts}

    out: dict[str, Path] = {}
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ext_set:
            continue
        out[p.stem] = p
    return out


def _safe_dist_from_counts(counts: np.ndarray) -> list[float]:
    total = max(1.0, float(counts.sum()))
    return (counts.astype(np.float64) / total).tolist()


def _compute_pair_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, cfg: dict) -> tuple[dict, torch.Tensor]:
    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"]["ignore_index"])
    cm = confusion_matrix(
        torch.from_numpy(pred_mask).long(),
        torch.from_numpy(gt_mask).long(),
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return metrics_from_confusion(cm), cm


def _resize_mask_to_shape(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if mask.shape[:2] == (h, w):
        return mask
    resized = cv2.resize(mask.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int64)


def _pred_only_classes(metrics: dict, threshold: float = PRED_ONLY_WARN_THRESHOLD) -> list[int]:
    gt_dist = np.asarray(metrics["gt_dist"], dtype=np.float64)
    pred_dist = np.asarray(metrics["pred_dist"], dtype=np.float64)
    return np.where((gt_dist <= 1e-12) & (pred_dist >= threshold))[0].tolist()


def _normalize_stem(stem: str) -> str:
    tokens = re.split(r"[^a-z0-9]+", stem.lower())
    tokens = [t for t in tokens if t and t not in PAIRING_STOPWORDS]
    return "".join(tokens)


def _numeric_key(stem: str) -> str | None:
    groups = re.findall(r"\d+", stem)
    if not groups:
        return None
    groups = sorted(groups, key=lambda x: (len(x), x))
    return groups[-1]


def _build_key_map(paths: list[Path], key_fn: Callable[[Path], str | None]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in paths:
        key = key_fn(p)
        if not key:
            continue
        out.setdefault(key, []).append(p)
    return out


def _unique_key_match(
    left: list[Path],
    right: list[Path],
    key_fn: Callable[[Path], str | None],
) -> dict[Path, Path]:
    left_map = _build_key_map(left, key_fn)
    right_map = _build_key_map(right, key_fn)
    pairs: dict[Path, Path] = {}
    for key in sorted(set(left_map.keys()) & set(right_map.keys())):
        if len(left_map[key]) == 1 and len(right_map[key]) == 1:
            pairs[left_map[key][0]] = right_map[key][0]
    return pairs


def _consume_pairs(
    unmatched_left: list[Path],
    unmatched_right: list[Path],
    pairs: dict[Path, Path],
    new_pairs: dict[Path, Path],
) -> tuple[list[Path], list[Path]]:
    if not new_pairs:
        return unmatched_left, unmatched_right
    pairs.update(new_pairs)
    used_left = set(new_pairs.keys())
    used_right = set(new_pairs.values())
    left_next = [p for p in unmatched_left if p not in used_left]
    right_next = [p for p in unmatched_right if p not in used_right]
    return left_next, right_next


def _dhash(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, b in enumerate(diff.reshape(-1)):
        if bool(b):
            bits |= 1 << i
    return bits


def _match_by_hash(left: list[Path], right: list[Path]) -> dict[Path, Path]:
    if not left or not right:
        return {}
    left_hash = {p: _dhash(p) for p in left}
    right_hash = {p: _dhash(p) for p in right}
    remaining = set(right)
    pairs: dict[Path, Path] = {}
    for lp in left:
        if not remaining:
            break
        lhash = left_hash[lp]
        best = min(remaining, key=lambda rp: (lhash ^ right_hash[rp]).bit_count())
        pairs[lp] = best
        remaining.remove(best)
    return pairs


def _pair_before_after(
    before_paths: list[Path],
    after_paths: list[Path],
    strategy: str = "auto",
) -> tuple[dict[Path, Path], list[dict], list[Path], list[Path]]:
    strategy = str(strategy).lower()
    if strategy not in PAIRING_CHOICES:
        raise ValueError(f"Unknown pairing strategy '{strategy}'. Choose from {PAIRING_CHOICES}.")

    unmatched_before = sorted(before_paths, key=lambda p: p.name.lower())
    unmatched_after = sorted(after_paths, key=lambda p: p.name.lower())
    pairs: dict[Path, Path] = {}
    steps: list[dict] = []

    def apply_unique(name: str, key_fn: Callable[[Path], str | None]) -> None:
        nonlocal unmatched_before, unmatched_after
        new_pairs = _unique_key_match(unmatched_before, unmatched_after, key_fn)
        unmatched_before, unmatched_after = _consume_pairs(unmatched_before, unmatched_after, pairs, new_pairs)
        steps.append({"method": name, "matched": len(new_pairs)})

    def apply_index() -> None:
        nonlocal unmatched_before, unmatched_after
        k = min(len(unmatched_before), len(unmatched_after))
        new_pairs = {unmatched_before[i]: unmatched_after[i] for i in range(k)}
        unmatched_before, unmatched_after = _consume_pairs(unmatched_before, unmatched_after, pairs, new_pairs)
        steps.append({"method": "index", "matched": len(new_pairs)})

    def apply_hash() -> None:
        nonlocal unmatched_before, unmatched_after
        new_pairs = _match_by_hash(unmatched_before, unmatched_after)
        unmatched_before, unmatched_after = _consume_pairs(unmatched_before, unmatched_after, pairs, new_pairs)
        steps.append({"method": "hash", "matched": len(new_pairs)})

    if strategy == "auto":
        apply_unique("stem", lambda p: p.stem.lower())
        apply_unique("normalized", lambda p: _normalize_stem(p.stem))
        apply_unique("numeric", lambda p: _numeric_key(p.stem))
        apply_hash()
        apply_index()
    elif strategy == "stem":
        apply_unique("stem", lambda p: p.stem.lower())
    elif strategy == "normalized":
        apply_unique("normalized", lambda p: _normalize_stem(p.stem))
    elif strategy == "numeric":
        apply_unique("numeric", lambda p: _numeric_key(p.stem))
    elif strategy == "hash":
        apply_hash()
    elif strategy == "index":
        apply_index()

    return pairs, steps, unmatched_before, unmatched_after


def _pair_before_gt(
    before_paths: list[Path],
    gt_paths: list[Path],
) -> tuple[dict[Path, Path], list[dict], list[Path], list[Path]]:
    # GT masks typically cannot be matched with image hash, so we use name/number plus index fallback.
    unmatched_before = sorted(before_paths, key=lambda p: p.name.lower())
    unmatched_gt = sorted(gt_paths, key=lambda p: p.name.lower())
    pairs: dict[Path, Path] = {}
    steps: list[dict] = []

    def apply_unique(name: str, key_fn: Callable[[Path], str | None]) -> None:
        nonlocal unmatched_before, unmatched_gt
        new_pairs = _unique_key_match(unmatched_before, unmatched_gt, key_fn)
        unmatched_before, unmatched_gt = _consume_pairs(unmatched_before, unmatched_gt, pairs, new_pairs)
        steps.append({"method": name, "matched": len(new_pairs)})

    def apply_index() -> None:
        nonlocal unmatched_before, unmatched_gt
        k = min(len(unmatched_before), len(unmatched_gt))
        new_pairs = {unmatched_before[i]: unmatched_gt[i] for i in range(k)}
        unmatched_before, unmatched_gt = _consume_pairs(unmatched_before, unmatched_gt, pairs, new_pairs)
        steps.append({"method": "index", "matched": len(new_pairs)})

    apply_unique("stem", lambda p: p.stem.lower())
    apply_unique("normalized", lambda p: _normalize_stem(p.stem))
    apply_unique("numeric", lambda p: _numeric_key(p.stem))
    apply_index()
    return pairs, steps, unmatched_before, unmatched_gt


def _build_triplets(
    before_paths: list[Path],
    after_paths: list[Path],
    gt_paths: list[Path],
    pair_strategy: str,
) -> dict:
    before_to_after, after_steps, unmatched_before_for_after, unmatched_after = _pair_before_after(
        before_paths=before_paths,
        after_paths=after_paths,
        strategy=pair_strategy,
    )
    before_to_gt, gt_steps, unmatched_before_for_gt, unmatched_gt = _pair_before_gt(
        before_paths=before_paths,
        gt_paths=gt_paths,
    )

    triplets: list[tuple[Path, Path, Path]] = []
    missing_after: list[Path] = []
    missing_gt: list[Path] = []

    for b in sorted(before_paths, key=lambda p: p.name.lower()):
        a = before_to_after.get(b)
        g = before_to_gt.get(b)
        if g is None:
            missing_gt.append(b)
            continue
        if a is None:
            missing_after.append(b)
            continue
        triplets.append((b, a, g))

    return {
        "triplets": triplets,
        "after_steps": after_steps,
        "gt_steps": gt_steps,
        "unmatched_before_for_after": unmatched_before_for_after,
        "unmatched_after": unmatched_after,
        "unmatched_before_for_gt": unmatched_before_for_gt,
        "unmatched_gt": unmatched_gt,
        "missing_after_for_before": missing_after,
        "missing_gt_for_before": missing_gt,
    }


def _make_pair_panel(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    gt_overlay = overlay_mask(before_rgb, mask_to_color(gt_mask, palette.shape[0], palette), alpha=0.55)
    before_overlay = overlay_mask(before_rgb, mask_to_color(pred_before, palette.shape[0], palette), alpha=0.55)
    after_overlay = overlay_mask(after_rgb, mask_to_color(pred_after, palette.shape[0], palette), alpha=0.55)

    labels = ["Before Input", "After Input", "GT Overlay", "Before Pred", "After Pred"]
    cells = [before_rgb, after_rgb, gt_overlay, before_overlay, after_overlay]
    target_h, target_w = 240, 330
    bar_h = 32

    rendered: list[np.ndarray] = []
    for label, cell in zip(labels, cells):
        img = cv2.resize(cell, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h + bar_h, target_w, 3), 248, dtype=np.uint8)
        canvas[bar_h:, :, :] = img
        cv2.putText(canvas, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (28, 28, 28), 2, cv2.LINE_AA)
        rendered.append(canvas)
    return np.concatenate(rendered, axis=1)


def evaluate_before_after_from_maps(
    bundle: dict,
    before_map: dict[str, Path],
    after_map: dict[str, Path],
    gt_map: dict[str, Path],
    save_panels: bool = False,
    panel_dir: str | Path | None = None,
    panel_limit: int = 40,
    pair_strategy: str = "auto",
) -> dict:
    cfg = bundle["cfg"]
    class_names = _class_names(cfg)
    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"]["ignore_index"])
    palette = default_palette(num_classes)

    before_paths = sorted(before_map.values(), key=lambda p: p.name.lower())
    after_paths = sorted(after_map.values(), key=lambda p: p.name.lower())
    gt_paths = sorted(gt_map.values(), key=lambda p: p.name.lower())

    match_info = _build_triplets(
        before_paths=before_paths,
        after_paths=after_paths,
        gt_paths=gt_paths,
        pair_strategy=pair_strategy,
    )
    triplets = match_info["triplets"]
    if not triplets:
        raise RuntimeError(
            "No matched before/after/GT triplets found. "
            "Try pairing='index' or check that uploads correspond to same scenes."
        )

    panel_root = None
    if save_panels:
        if panel_dir is None:
            raise ValueError("panel_dir must be provided when save_panels=True.")
        panel_root = Path(panel_dir)
        panel_root.mkdir(parents=True, exist_ok=True)

    cm_before = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    cm_after = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    gt_counts = np.zeros(num_classes, dtype=np.int64)
    pred_before_counts = np.zeros(num_classes, dtype=np.int64)
    pred_after_counts = np.zeros(num_classes, dtype=np.int64)
    per_image: list[dict] = []

    for i, (before_path, after_path, gt_path) in enumerate(triplets):
        before_rgb = read_image_rgb(before_path)
        after_rgb = read_image_rgb(after_path)
        gt_raw = read_mask_raw(gt_path)
        gt_mask = map_mask_values(gt_raw, cfg)

        pred_before, _ = predict_mask(bundle, before_rgb)
        pred_after, _ = predict_mask(bundle, after_rgb)
        pred_before_eval = _resize_mask_to_shape(pred_before, gt_mask.shape[:2])
        pred_after_eval = _resize_mask_to_shape(pred_after, gt_mask.shape[:2])

        metrics_before, cm_b = _compute_pair_metrics(pred_before_eval, gt_mask, cfg)
        metrics_after, cm_a = _compute_pair_metrics(pred_after_eval, gt_mask, cfg)
        cm_before += cm_b
        cm_after += cm_a

        valid = gt_mask != ignore_index
        valid_gt = gt_mask[valid]
        if valid_gt.size > 0:
            gt_counts += np.bincount(valid_gt.reshape(-1), minlength=num_classes)
            pred_before_counts += np.bincount(pred_before_eval[valid].reshape(-1), minlength=num_classes)
            pred_after_counts += np.bincount(pred_after_eval[valid].reshape(-1), minlength=num_classes)

        per_image.append(
            {
                "stem": before_path.stem,
                "before_file": before_path.name,
                "after_file": after_path.name,
                "gt_file": gt_path.name,
                "before": {
                    "miou": float(metrics_before["miou"]),
                    "map50": float(metrics_before["map50"]),
                    "dice": float(metrics_before["dice"]),
                    "pixel_acc": float(metrics_before["pixel_acc"]),
                },
                "after": {
                    "miou": float(metrics_after["miou"]),
                    "map50": float(metrics_after["map50"]),
                    "dice": float(metrics_after["dice"]),
                    "pixel_acc": float(metrics_after["pixel_acc"]),
                },
                "miou_gain": float(metrics_after["miou"] - metrics_before["miou"]),
                "map50_gain": float(metrics_after["map50"] - metrics_before["map50"]),
            }
        )

        if panel_root is not None and i < panel_limit:
            panel = _make_pair_panel(
                before_rgb=before_rgb,
                after_rgb=after_rgb,
                gt_mask=gt_mask,
                pred_before=pred_before,
                pred_after=pred_after,
                palette=palette,
            )
            panel_path = panel_root / f"{before_path.stem}_paired_panel.png"
            cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    aggregate_before = metrics_from_confusion(cm_before)
    aggregate_after = metrics_from_confusion(cm_after)
    aggregate_before["gt_dist"] = _safe_dist_from_counts(gt_counts)
    aggregate_before["pred_dist"] = _safe_dist_from_counts(pred_before_counts)
    aggregate_before["pred_only_classes"] = _pred_only_classes(aggregate_before)
    aggregate_after["gt_dist"] = _safe_dist_from_counts(gt_counts)
    aggregate_after["pred_dist"] = _safe_dist_from_counts(pred_after_counts)
    aggregate_after["pred_only_classes"] = _pred_only_classes(aggregate_after)

    return {
        "num_classes": num_classes,
        "class_names": class_names,
        "counts": {
            "before": len(before_paths),
            "after": len(after_paths),
            "gt": len(gt_paths),
            "paired": len(triplets),
        },
        "pairing": {
            "strategy_requested": pair_strategy,
            "after_steps": match_info["after_steps"],
            "gt_steps": match_info["gt_steps"],
            "unmatched_before_for_after": [p.name for p in match_info["unmatched_before_for_after"]],
            "unmatched_after": [p.name for p in match_info["unmatched_after"]],
            "unmatched_before_for_gt": [p.name for p in match_info["unmatched_before_for_gt"]],
            "unmatched_gt": [p.name for p in match_info["unmatched_gt"]],
            "missing_after_for_before": [p.name for p in match_info["missing_after_for_before"]],
            "missing_gt_for_before": [p.name for p in match_info["missing_gt_for_before"]],
        },
        "aggregate": {
            "before": aggregate_before,
            "after": aggregate_after,
            "miou_gain": float(aggregate_after["miou"] - aggregate_before["miou"]),
            "map50_gain": float(aggregate_after["map50"] - aggregate_before["map50"]),
            "dice_gain": float(aggregate_after["dice"] - aggregate_before["dice"]),
            "pixel_acc_gain": float(aggregate_after["pixel_acc"] - aggregate_before["pixel_acc"]),
            "class_names": class_names,
        },
        "per_image": per_image,
    }


def run_before_after_eval(
    config_path: str,
    checkpoint_path: str,
    before_dir: str,
    after_dir: str,
    gt_dir: str,
    output_dir: str,
    device: str = "cuda",
    save_panels: bool = True,
    panel_limit: int = 40,
    pair_strategy: str = "auto",
) -> Path:
    bundle = load_model_bundle(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    before_map = build_stem_map_from_dir(before_dir)
    after_map = build_stem_map_from_dir(after_dir)
    gt_map = build_stem_map_from_dir(gt_dir)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = out / "panels"
    if save_panels:
        panels.mkdir(parents=True, exist_ok=True)

    results = evaluate_before_after_from_maps(
        bundle=bundle,
        before_map=before_map,
        after_map=after_map,
        gt_map=gt_map,
        save_panels=save_panels,
        panel_dir=panels,
        panel_limit=panel_limit,
        pair_strategy=pair_strategy,
    )
    results["config_path"] = str(Path(config_path).resolve())
    results["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    results["before_dir"] = str(Path(before_dir).resolve())
    results["after_dir"] = str(Path(after_dir).resolve())
    results["gt_dir"] = str(Path(gt_dir).resolve())

    result_path = out / "results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation quality on before/after image sets.")
    parser.add_argument("--config", required=True, help="Path to model config yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--before", required=True, help="Directory with BEFORE images.")
    parser.add_argument("--after", required=True, help="Directory with AFTER images.")
    parser.add_argument("--gt", required=True, help="Directory with ground-truth masks.")
    parser.add_argument("--output", default="project/outputs/before_after_eval", help="Output directory for results.")
    parser.add_argument("--device", default="cuda", help="Device preference (cuda or cpu).")
    parser.add_argument("--no_panels", action="store_true", help="Disable writing visualization panels.")
    parser.add_argument("--panel_limit", type=int, default=40, help="Max number of saved panel images.")
    parser.add_argument(
        "--pairing",
        default="auto",
        choices=PAIRING_CHOICES,
        help="How to pair before/after images when filenames differ.",
    )
    args = parser.parse_args()

    result_path = run_before_after_eval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        before_dir=args.before,
        after_dir=args.after,
        gt_dir=args.gt,
        output_dir=args.output,
        device=args.device,
        save_panels=not args.no_panels,
        panel_limit=max(0, int(args.panel_limit)),
        pair_strategy=args.pairing,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    before = payload["aggregate"]["before"]
    after = payload["aggregate"]["after"]
    print(f"Matched pairs: {payload['counts']['paired']}")
    print(
        "Before  | mIoU={:.4f} | mAP50={:.4f} | Dice={:.4f}".format(
            before["miou"], before["map50"], before["dice"]
        )
    )
    print(
        "After   | mIoU={:.4f} | mAP50={:.4f} | Dice={:.4f}".format(
            after["miou"], after["map50"], after["dice"]
        )
    )
    print(
        "Delta   | mIoU={:+.4f} | mAP50={:+.4f} | Dice={:+.4f}".format(
            payload["aggregate"]["miou_gain"],
            payload["aggregate"]["map50_gain"],
            payload["aggregate"]["dice_gain"],
        )
    )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
