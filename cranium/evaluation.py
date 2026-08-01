"""Verification accuracy metrics — Equal Error Rate (EER) and friends.

Backend-agnostic. Give it identity labels and embeddings; it builds genuine and
impostor cosine-similarity distributions and computes the numbers that decide
whether the engine clears the Phase 1 gate (EER < 3%):

  * EER            — the balanced error point (false-accept rate == false-reject rate)
  * threshold      — the score at that operating point
  * AUC            — probability a genuine pair scores above an impostor pair
  * TAR @ FAR      — true-accept rate at fixed false-accept rates (10%, 1%, 0.1%)

None of this depends on which embedder produced the vectors, so the exact same
harness measures the stub, dlib, facenet, or a future fine-tuned periocular model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .embedding import l2_normalize


def pairwise_scores(embeddings: Sequence[np.ndarray], labels: Sequence,
                    max_impostor_pairs: int = 300_000, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine similarities for every image pair, split into genuine (same
    identity) and impostor (different identity). Impostor pairs are subsampled
    if there are more than `max_impostor_pairs` (they explode quadratically)."""
    E = np.stack([l2_normalize(e) for e in embeddings]).astype(np.float32)
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2:
        raise ValueError("need at least two embeddings")
    sim = E @ E.T                      # full cosine-similarity matrix (unit vectors)
    iu = np.triu_indices(n, k=1)       # upper triangle, no self-pairs
    same = labels[iu[0]] == labels[iu[1]]
    scores = sim[iu]
    genuine = scores[same].astype(np.float32)
    impostor = scores[~same].astype(np.float32)
    if impostor.size > max_impostor_pairs:
        rng = np.random.default_rng(seed)
        impostor = rng.choice(impostor, size=max_impostor_pairs, replace=False)
    return genuine, impostor


@dataclass
class EERResult:
    eer: float
    threshold: float
    auc: float
    n_genuine: int
    n_impostor: int
    tar_at_far: Dict[float, float]

    def summary(self) -> str:
        tar = ", ".join(f"TAR@FAR={f:g}: {t*100:.1f}%" for f, t in self.tar_at_far.items())
        return (f"EER {self.eer*100:.2f}%  @threshold {self.threshold:.3f}  "
                f"AUC {self.auc:.4f}  ({self.n_genuine} genuine / {self.n_impostor} impostor)\n  {tar}")


def compute_eer(genuine: np.ndarray, impostor: np.ndarray,
                far_targets: Sequence[float] = (0.1, 0.01, 0.001),
                grid: int = 4000) -> EERResult:
    """Compute EER by sweeping a decision threshold across the score range.

    Decision rule: accept if score >= threshold.
      FRR(t) = fraction of genuine pairs scoring < t   (false rejects)
      FAR(t) = fraction of impostor pairs scoring >= t (false accepts)
    EER is where FAR and FRR cross.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("need both genuine and impostor scores")

    g = np.sort(genuine)
    im = np.sort(impostor)
    lo = min(g[0], im[0])
    hi = max(g[-1], im[-1])
    thr = np.linspace(lo, hi, grid)

    frr = np.searchsorted(g, thr, side="left") / g.size          # genuine below t
    far = (im.size - np.searchsorted(im, thr, side="left")) / im.size  # impostor >= t

    k = int(np.argmin(np.abs(far - frr)))
    eer = float((far[k] + frr[k]) / 2.0)
    threshold = float(thr[k])

    tar_at: Dict[float, float] = {}
    for ft in far_targets:
        idx = np.where(far <= ft)[0]
        tar_at[float(ft)] = float(1.0 - frr[idx[0]]) if idx.size else 0.0

    return EERResult(eer=eer, threshold=threshold, auc=_auc(genuine, impostor),
                     n_genuine=int(genuine.size), n_impostor=int(impostor.size),
                     tar_at_far=tar_at)


def _auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (tie-aware).
    Equals P(genuine score > impostor score)."""
    allv = np.concatenate([genuine, impostor])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    ng = genuine.size
    ni = impostor.size
    rg = ranks[:ng].sum()
    return float((rg - ng * (ng + 1) / 2.0) / (ng * ni))


def evaluate(embeddings: Sequence[np.ndarray], labels: Sequence,
             **kwargs) -> EERResult:
    """Convenience: labels+embeddings -> EERResult in one call."""
    genuine, impostor = pairwise_scores(embeddings, labels,
                                        seed=kwargs.pop("seed", 0),
                                        max_impostor_pairs=kwargs.pop("max_impostor_pairs", 300_000))
    return compute_eer(genuine, impostor, **kwargs)
