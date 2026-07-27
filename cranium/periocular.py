"""Periocular region-of-interest extraction.

Given facial landmarks, compute and crop the upper-face box (eyes, brows, nose
bridge, forehead, temples) that CRANIUM matches on. Working from this region —
rather than the whole face — is what lets the system operate when the lower face
is covered.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

from .config import PERIOCULAR_LANDMARKS, ROI_EXPAND


def periocular_roi(landmarks_xy: np.ndarray, image_shape,
                   expand: float = ROI_EXPAND) -> Tuple[int, int, int, int]:
    """Return an (x1, y1, x2, y2) pixel box over the periocular landmarks,
    padded by `expand`, biased slightly upward toward the forehead, and clamped
    to the image bounds."""
    h, w = image_shape[:2]
    pts = landmarks_xy[PERIOCULAR_LANDMARKS]
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = x2 - x1, y2 - y1

    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand
    y2 += bh * expand
    y1 -= bh * expand  # extend a little further up toward the forehead

    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(w, round(x2)))
    y2 = int(min(h, round(y2)))
    return x1, y1, x2, y2


def crop_periocular(image_bgr: np.ndarray, landmarks_xy: np.ndarray,
                    expand: float = ROI_EXPAND):
    """Return (crop, roi_box). The crop is a copy; roi_box is (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = periocular_roi(landmarks_xy, image_bgr.shape, expand)
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)
