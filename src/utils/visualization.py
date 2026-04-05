from __future__ import annotations

import numpy as np


def generate_palette(num_classes: int) -> np.ndarray:
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        hue = i / max(1, num_classes)
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        palette[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return palette


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    palette = generate_palette(num_classes)
    mask = np.clip(mask, 0, num_classes - 1)
    return palette[mask]


def overlay_image(image: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (image * (1 - alpha) + mask_color * alpha).astype(np.uint8)
