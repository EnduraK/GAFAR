"""Liveness v0: an eye-blink / attention challenge.

This is a PLACEHOLDER for the layered Phase 2 liveness described in the brief
(passive texture CNN + attention challenge + active screen-illumination
reflection, plus injection-attack defence). A blink check alone is trivially
defeated by a video replay; it exists here to (a) prove the challenge-response
plumbing and (b) force user attention, which Apple's mask mode also does. Do
not represent this as presentation-attack detection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import (
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    EAR_BLINK_THRESHOLD,
    EAR_OPEN_THRESHOLD,
    BLINK_WINDOW_S,
)


def _ear(points: np.ndarray) -> float:
    """Eye Aspect Ratio for six landmarks ordered p1..p6 (corners + lids)."""
    p1, p2, p3, p4, p5, p6 = points
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    if horiz == 0:
        return 0.0
    return float(vert / (2.0 * horiz))


def eye_aspect_ratios(landmarks_xy: np.ndarray) -> Tuple[float, float, float]:
    left = _ear(landmarks_xy[LEFT_EYE_EAR])
    right = _ear(landmarks_xy[RIGHT_EYE_EAR])
    return left, right, (left + right) / 2.0


@dataclass
class BlinkResult:
    passed: bool
    blinks: int
    reason: str


class BlinkChallenge:
    """State machine over per-frame EAR. Counts a blink on each closed->open
    transition and passes once `required_blinks` occur inside the time window."""

    def __init__(self, required_blinks: int = 1, window_s: float = BLINK_WINDOW_S):
        self.required = required_blinks
        self.window_s = window_s
        self.reset()

    def reset(self) -> None:
        self._eye_closed = False
        self.blinks = 0
        self._t0: Optional[float] = None

    def update(self, ear: float, t: float) -> BlinkResult:
        if self._t0 is None:
            self._t0 = t
        # open -> closed
        if not self._eye_closed and ear < EAR_BLINK_THRESHOLD:
            self._eye_closed = True
        # closed -> open  (one completed blink)
        elif self._eye_closed and ear > EAR_OPEN_THRESHOLD:
            self._eye_closed = False
            self.blinks += 1

        if self.blinks >= self.required:
            return BlinkResult(True, self.blinks, "blink detected")
        if t - self._t0 > self.window_s:
            return BlinkResult(False, self.blinks, "timeout: no blink")
        return BlinkResult(False, self.blinks, "waiting for blink")
