"""End-to-end periocular embedding pipeline: image -> landmarks -> crop -> vector."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .periocular import crop_periocular


@dataclass
class FaceResult:
    embedding: np.ndarray
    landmarks: np.ndarray
    roi: Tuple[int, int, int, int]


class PeriocularPipeline:
    """Ties a landmarker and an embedder into a single `embed()` call."""

    def __init__(self, landmarker, embedder, expand: Optional[float] = None):
        self.landmarker = landmarker
        self.embedder = embedder
        self.expand = expand

    def embed(self, image_bgr: np.ndarray) -> Optional[FaceResult]:
        landmarks = self.landmarker.detect(image_bgr)
        if landmarks is None:
            return None
        if self.expand is None:
            crop, roi = crop_periocular(image_bgr, landmarks)
        else:
            crop, roi = crop_periocular(image_bgr, landmarks, self.expand)
        if crop.size == 0:
            return None
        vector = self.embedder.embed(crop)
        return FaceResult(embedding=vector, landmarks=landmarks, roi=roi)
