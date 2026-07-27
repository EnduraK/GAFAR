"""Pluggable embedding backends.

The embedder turns a periocular crop into an L2-normalized vector. Two backends
ship here:

  * StubEmbedder  - deterministic, model-free. For WIRING AND TESTS ONLY. It does
                    not recognise a person across different photos.
  * DlibEmbedder  - real 128-D ResNet descriptor (dlib, Boost licence -> the
                    commercially clean option flagged in the brief). Applied to
                    the periocular crop as a v0 stand-in until a periocular-
                    specific model is fine-tuned.

IP note: dlib's model is Boost-licensed and safe for commercial use. Do NOT
swap in InsightFace/ArcFace *weights* for production without a commercial
licence — their code is MIT but the pretrained models are non-commercial.
"""
from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
import numpy as np

from .config import EMBED_DIM


def l2_normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return (v / n).astype(np.float32)


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalized float32 embedding of a periocular crop."""
        raise NotImplementedError


class StubEmbedder(Embedder):
    """Deterministic hash-based pseudo-embedding. Identical crops map to
    identical vectors; different crops map to (uncorrelated) different vectors.
    Enough to exercise the full pipeline without a model — NOT a recogniser."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(image_bgr)
        digest = hashlib.sha256(arr.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return l2_normalize(rng.standard_normal(self.dim))


class DlibEmbedder(Embedder):
    """Real face descriptor via dlib. Requires `dlib`, a shape predictor, and
    the recognition model file. Kept optional so the package imports and tests
    run without the heavy dependency."""

    def __init__(self, shape_predictor_path: str, recognition_model_path: str):
        import dlib  # guarded import
        self.dim = 128
        self._dlib = dlib
        self._sp = dlib.shape_predictor(shape_predictor_path)
        self._rec = dlib.face_recognition_model_v1(recognition_model_path)
        self._detector = dlib.get_frontal_face_detector()

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        dets = self._detector(rgb, 1)
        if not dets:
            raise ValueError("DlibEmbedder: no face detected in crop")
        shape = self._sp(rgb, dets[0])
        desc = self._rec.compute_face_descriptor(rgb, shape)
        return l2_normalize(np.array(desc, dtype=np.float32))
