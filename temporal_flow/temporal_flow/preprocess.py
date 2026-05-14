from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    width: int = 350
    height: int = 350
    crop_px: int = 175
    crop_py: int = 175
    crop_offset_x: int = 0
    crop_offset_y: int = -8
    circular_mask: bool = True
    mask_radius: int = 160
    radial_correction: bool = True
    radial_max_value: int = 60
    radial_power: float = 2.0
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8


def create_radial_mask(size: Tuple[int, int], max_value: int = 60, power: float = 2.0) -> np.ndarray:
    h, w = size
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    denom = max(np.sqrt(cx ** 2 + cy ** 2), 1.0)
    normalized = dist / denom
    return np.clip((normalized ** power) * max_value, 0, max_value).astype(np.uint8)


def preprocess_frame_bgr(frame_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Crop, mask, grayscale, resize, and illumination-correct a BioTacTip frame.

    Returns uint8 grayscale HxW image.
    """
    h0, w0 = frame_bgr.shape[:2]
    cx, cy = w0 // 2, h0 // 2
    x1 = max(cx - cfg.crop_px + cfg.crop_offset_x, 0)
    y1 = max(cy - cfg.crop_py + cfg.crop_offset_y, 0)
    x2 = min(cx + cfg.crop_px + cfg.crop_offset_x, w0)
    y2 = min(cy + cfg.crop_py + cfg.crop_offset_y, h0)
    cropped = frame_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("Crop produced an empty image. Check crop parameters.")

    resized = cv2.resize(cropped, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    if cfg.circular_mask:
        mask = np.zeros((cfg.height, cfg.width), dtype=np.uint8)
        cv2.circle(mask, (cfg.width // 2, cfg.height // 2), cfg.mask_radius, 255, -1)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    if cfg.radial_correction:
        radial = create_radial_mask((cfg.height, cfg.width), cfg.radial_max_value, cfg.radial_power)
        gray = cv2.subtract(gray, radial)

    if cfg.clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
        )
        gray = clahe.apply(gray)

    return gray


def configure_camera(cap: cv2.VideoCapture, exposure: Optional[float] = -7.8, brightness: Optional[float] = 0,
                     contrast: Optional[float] = 64, saturation: Optional[float] = 60,
                     hue: Optional[float] = 0, gain: Optional[float] = 0) -> None:
    props = [
        (cv2.CAP_PROP_EXPOSURE, exposure),
        (cv2.CAP_PROP_BRIGHTNESS, brightness),
        (cv2.CAP_PROP_CONTRAST, contrast),
        (cv2.CAP_PROP_SATURATION, saturation),
        (cv2.CAP_PROP_HUE, hue),
        (cv2.CAP_PROP_GAIN, gain),
    ]
    for prop, value in props:
        if value is not None:
            cap.set(prop, float(value))
