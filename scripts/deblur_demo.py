"""
Bonus Round: Blurred Image → Deblur → Segment → RGB Color Map

Pipeline per image:
  1. Load blurred image
  2. Wiener deconvolution to restore sharpness
  3. Run segmentation model on deblurred image
  4. Convert class predictions → RGB color map
  5. Save 3-panel: [Blurred | Deblurred | Segmentation]

Usage:
    python scripts/deblur_demo.py \
        --config configs/blur_robust.yaml \
        --checkpoint outputs/checkpoints/best.pth \
        --input path/to/blurred/images \
        --output outputs/deblur_results
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model
from src.data.transforms import build_eval_transforms
from src.utils.config import load_config, resolve_paths

# ── Class names and RGB colors (matches label_map order) ──────────────────────
CLASS_NAMES = [
    "Road", "Low Vegetation", "Unpaved Road", "Obstacle",
    "High Vegetation", "Sky", "Vehicle", "Rough Trail",
    "Smooth Trail", "Water",
]
CLASS_COLORS = np.array([
    [128,  64, 128],   # 0  Road             — purple
    [107, 142,  35],   # 1  Low Vegetation   — olive green
    [190, 153, 153],   # 2  Unpaved Road     — pink
    [220,  20,  60],   # 3  Obstacle         — crimson
    [ 34, 139,  34],   # 4  High Vegetation  — forest green
    [ 70, 130, 180],   # 5  Sky              — steel blue
    [  0,   0, 142],   # 6  Vehicle          — dark blue
    [244, 164,  96],   # 7  Rough Trail      — sandy brown
    [255, 255,   0],   # 8  Smooth Trail     — yellow
    [  0, 191, 255],   # 9  Water            — deep sky blue
], dtype=np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Deblurring ─────────────────────────────────────────────────────────────────
def wiener_deblur(img_rgb: np.ndarray, kernel_size: int = 21, noise_var: float = 0.02, blur_type: str = "gaussian") -> np.ndarray:
    """
    DFT-based Wiener deconvolution per channel.
    blur_type: 'motion' for horizontal motion blur, 'gaussian' for defocus/gaussian blur.
    Increase noise_var for heavily noisy images.
    """
    if blur_type == "gaussian":
        # Gaussian PSF — better for defocus and gaussian blur
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=np.float32)
        sigma = kernel_size / 6.0
        gauss_1d = np.exp(-ax ** 2 / (2 * sigma ** 2))
        kernel = np.outer(gauss_1d, gauss_1d)
        kernel /= kernel.sum()
    else:
        # Horizontal motion blur PSF
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

    img_f = img_rgb.astype(np.float32) / 255.0
    result = np.empty_like(img_f)
    h, w = img_f.shape[:2]

    kernel_pad = np.zeros((h, w), dtype=np.float32)
    kernel_pad[:kernel.shape[0], :kernel.shape[1]] = kernel
    K = np.fft.fft2(kernel_pad)
    K_conj = np.conj(K)
    denom = np.abs(K) ** 2 + noise_var

    for c in range(3):
        I = np.fft.fft2(img_f[:, :, c])
        restored = np.real(np.fft.ifft2((K_conj / denom) * I))
        result[:, :, c] = np.clip(restored, 0.0, 1.0)

    # Post-process: bilateral filter to suppress ringing artifacts from deconvolution
    result_uint8 = (result * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(result_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


# ── Segmentation ───────────────────────────────────────────────────────────────
def predict(model, img_rgb: np.ndarray, transforms, device: torch.device) -> np.ndarray:
    """Returns HxW uint8 class-id map at original image resolution."""
    orig_h, orig_w = img_rgb.shape[:2]
    tensor = transforms(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert HxW class-id mask → HxW×3 RGB color image."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


# ── Legend ─────────────────────────────────────────────────────────────────────
def draw_legend(width: int) -> np.ndarray:
    """Draw a horizontal legend strip showing class colors and names."""
    h = 28
    legend = np.ones((h, width, 3), dtype=np.uint8) * 30  # dark background
    n = len(CLASS_NAMES)
    cell_w = width // n
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        x = i * cell_w
        cv2.rectangle(legend, (x + 2, 4), (x + 18, h - 4), color.tolist(), -1)
        cv2.putText(legend, name, (x + 22, h - 8), FONT, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
    return legend


# ── Panel builder ──────────────────────────────────────────────────────────────
def build_panel(blurred_bgr: np.ndarray, deblurred_bgr: np.ndarray, seg_rgb: np.ndarray) -> np.ndarray:
    """3-column panel: Blurred | Deblurred | Segmentation + legend at bottom."""
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    def titled(img, title):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(out, title, (8, 26), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    row = np.hstack([
        titled(blurred_bgr,  "Blurred Input"),
        titled(deblurred_bgr, "Deblurred"),
        titled(seg_bgr,       "Segmentation (RGB)"),
    ])
    legend = draw_legend(row.shape[1])
    return np.vstack([row, legend])


# ── Per-image processing ───────────────────────────────────────────────────────
def process(img_path: Path, model, transforms, device, out_dir: Path,
            kernel_size: int, noise_var: float):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [skip] {img_path.name} — cannot read")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: deblur
    deblurred_rgb = wiener_deblur(img_rgb, kernel_size=kernel_size, noise_var=noise_var, blur_type="gaussian")

    # Step 2: segment the deblurred image
    mask = predict(model, deblurred_rgb, transforms, device)

    # Step 3: colorize
    seg_rgb = to_rgb(mask)

    # Step 4: save outputs
    stem = img_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_deblurred.png"),   cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_segmentation.png"), cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_panel.png"),        build_panel(img_bgr, cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR), seg_rgb))

    print(f"  ✓ {stem}_panel.png  |  {stem}_segmentation.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deblur → Segment → RGB color map")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      required=True,  help="Folder of blurred images")
    parser.add_argument("--output",     default="outputs/deblur_results")
    parser.add_argument("--kernel",     type=int,   default=21,   help="Wiener blur kernel size (odd number, larger = stronger deblur)")
    parser.add_argument("--noise",      type=float, default=0.02,  help="Wiener noise variance (higher = smoother, less ringing)")
    parser.add_argument("--ext",        default=".png,.jpg,.jpeg")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config), os.path.dirname(args.config))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg).to(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")

    dataset_cfg = cfg.get("dataset", {})
    transforms = build_eval_transforms(
        tuple(cfg["train"]["image_size"]),
        mean=dataset_cfg.get("mean"),
        std=dataset_cfg.get("std"),
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.ext.split(",")}
    images = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"\nProcessing {len(images)} blurred images → {out_dir}\n")
    for img_path in images:
        process(img_path, model, transforms, device, out_dir, args.kernel, args.noise)

    print(f"\nDone. All results saved to: {out_dir}")
    print("Output files per image:")
    print("  *_panel.png       — 3-column: Blurred | Deblurred | Segmentation")
    print("  *_deblurred.png   — restored image only")
    print("  *_segmentation.png — RGB color segmentation map only")


if __name__ == "__main__":
    main()
