"""Facial landmark detection via the MediaPipe Tasks FaceLandmarker API.

Returns 478 3D landmarks (projected to 2D pixels here). This is the current,
production-correct MediaPipe API (the older `solutions.face_mesh` is deprecated
and is not shipped in recent wheels).

The model file `face_landmarker.task` (~3.7 MB) is NOT bundled in the pip wheel.
Resolution order for the model path:
  1. explicit `model_path` argument
  2. env var CRANIUM_FACE_MODEL
  3. ./models/face_landmarker.task (relative to CWD)
  4. <repo>/models/face_landmarker.task (relative to this file)
The import of mediapipe is guarded so the rest of the package works without it.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np


def default_model_path() -> Optional[str]:
    """Return the first face_landmarker.task we can find, or None."""
    env = os.environ.get("CRANIUM_FACE_MODEL")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "models" / "face_landmarker.task")
    candidates.append(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task")
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


class FaceMeshLandmarker:
    def __init__(self, model_path: Optional[str] = None,
                 min_detection_confidence: float = 0.5) -> None:
        from mediapipe.tasks import python as mp_python  # guarded import
        from mediapipe.tasks.python import vision

        path = model_path or default_model_path()
        if path is None:
            raise FileNotFoundError(
                "face_landmarker.task not found. Set CRANIUM_FACE_MODEL or place "
                "the model at ./models/face_landmarker.task")

        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._vision = vision
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return an (N, 2) array of pixel-space landmarks, or None if no face."""
        import mediapipe as mp
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        h, w = image_bgr.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
