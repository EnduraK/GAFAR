"""1:1 verification, template storage, and matching.

CRANIUM is 1:1 verification, never 1:N identification — the decision that keeps
it in the EU AI Act's light-touch lane (Art 3(36), Annex III carve-out). There
is therefore NO nearest-neighbour search here: a probe is compared against the
single enrolled template for the claimed identity. The template store keeps
embeddings only, never raw images (privacy-by-design).
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import DEFAULT_MATCH_THRESHOLD
from .embedding import l2_normalize


@dataclass
class Template:
    subject_id: str
    vector: np.ndarray            # L2-normalized float32
    n_samples: int = 1
    created_at: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    subject_id: str
    score: float
    threshold: float
    is_match: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; equals the dot product for normalized vectors."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class TemplateStore:
    """In-memory template store with JSON persistence. Embeddings only."""

    def __init__(self) -> None:
        self._t: Dict[str, Template] = {}

    def put(self, template: Template) -> None:
        self._t[template.subject_id] = template

    def get(self, subject_id: str) -> Optional[Template]:
        return self._t.get(subject_id)

    def has(self, subject_id: str) -> bool:
        return subject_id in self._t

    def delete(self, subject_id: str) -> bool:
        return self._t.pop(subject_id, None) is not None

    def subjects(self) -> List[str]:
        return list(self._t.keys())

    def save(self, path) -> None:
        data = {
            sid: {
                "vector": t.vector.tolist(),
                "n_samples": t.n_samples,
                "created_at": t.created_at,
                "meta": t.meta,
            }
            for sid, t in self._t.items()
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path) -> "TemplateStore":
        store = cls()
        data = json.loads(Path(path).read_text())
        for sid, d in data.items():
            store.put(
                Template(
                    subject_id=sid,
                    vector=np.array(d["vector"], dtype=np.float32),
                    n_samples=int(d.get("n_samples", 1)),
                    created_at=float(d.get("created_at", 0.0)),
                    meta=d.get("meta", {}),
                )
            )
        return store


class Verifier:
    def __init__(self, store: Optional[TemplateStore] = None,
                 threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        self.store = store or TemplateStore()
        self.threshold = threshold

    def enroll(self, subject_id: str, embeddings: List[np.ndarray],
               meta: Optional[dict] = None) -> Template:
        """Enroll from one or more samples. Multiple samples are averaged then
        renormalized — a simple, robust template that smooths per-capture noise."""
        if not embeddings:
            raise ValueError("need at least one embedding to enroll")
        mat = np.stack([l2_normalize(e) for e in embeddings])
        mean_vec = l2_normalize(mat.mean(axis=0))
        template = Template(subject_id, mean_vec, n_samples=len(embeddings),
                            meta=meta or {})
        self.store.put(template)
        return template

    def verify(self, subject_id: str, probe: np.ndarray,
               threshold: Optional[float] = None) -> VerificationResult:
        thr = self.threshold if threshold is None else threshold
        template = self.store.get(subject_id)
        if template is None:
            return VerificationResult(subject_id, 0.0, thr, False)
        score = cosine_similarity(template.vector, probe)
        return VerificationResult(subject_id, score, thr, score >= thr)
