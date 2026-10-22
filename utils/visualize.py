import os

import cv2
import numpy as np


PALETTE = np.array(
    [[0, 0, 0], [0, 255, 255], [255, 0, 0], [153, 76, 0], [0, 153, 0]],
    dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = PALETTE[mask.clip(0, len(PALETTE) - 1)]
    return color.astype(np.uint8)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    color = colorize_mask(mask)
    overlay = (image_rgb * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def save_visualization(image_rgb: np.ndarray, pred_mask: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    overlay = overlay_mask(image_rgb, pred_mask)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
