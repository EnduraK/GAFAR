import numpy as np
import pytest

from cranium.evaluation import compute_eer, pairwise_scores, evaluate, _auc
from cranium.embedding import l2_normalize


def test_perfectly_separable_gives_zero_eer():
    genuine = np.full(500, 0.9)
    impostor = np.full(500, 0.1)
    r = compute_eer(genuine, impostor)
    assert r.eer == pytest.approx(0.0, abs=1e-6)
    assert r.auc == pytest.approx(1.0, abs=1e-6)
    assert 0.1 < r.threshold < 0.9


def test_identical_distributions_give_half_eer():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 5000)
    y = rng.normal(0.5, 0.1, 5000)
    r = compute_eer(x, y)
    assert r.eer == pytest.approx(0.5, abs=0.05)
    assert r.auc == pytest.approx(0.5, abs=0.05)


def test_more_overlap_means_higher_eer():
    rng = np.random.default_rng(1)
    g = rng.normal(0.8, 0.1, 4000)
    easy = compute_eer(g, rng.normal(0.2, 0.1, 4000)).eer   # well separated
    hard = compute_eer(g, rng.normal(0.6, 0.1, 4000)).eer   # overlapping
    assert hard > easy


def test_tar_at_far_monotonic_and_bounded():
    rng = np.random.default_rng(2)
    r = compute_eer(rng.normal(0.8, 0.1, 4000), rng.normal(0.3, 0.1, 4000))
    tars = [r.tar_at_far[f] for f in (0.1, 0.01, 0.001)]
    assert all(0.0 <= t <= 1.0 for t in tars)
    # looser FAR allows at least as high a TAR as a stricter FAR
    assert tars[0] >= tars[1] >= tars[2] - 1e-9


def test_auc_equals_probability_genuine_beats_impostor():
    g = np.array([1.0, 2.0, 3.0])
    i = np.array([0.0, 1.5, 2.5])
    # brute-force P(g>i) with 0.5 for ties
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in g for b in i) / (len(g) * len(i))
    assert _auc(g, i) == pytest.approx(wins, abs=1e-9)


def _cluster(dim, center_seed, n, spread, start_seed):
    r0 = np.random.default_rng(center_seed)
    base = l2_normalize(r0.standard_normal(dim))
    out = []
    for k in range(n):
        r = np.random.default_rng(start_seed + k)
        u = r.standard_normal(dim)
        u = u - u.dot(base) * base
        u = u / (np.linalg.norm(u) or 1)
        out.append(l2_normalize(np.sqrt(1 - spread**2) * base + spread * u))
    return out


def test_pairwise_scores_separates_identities():
    dim = 128
    embs, labels = [], []
    for ident in range(4):
        for e in _cluster(dim, ident, 6, spread=0.3, start_seed=ident * 100):
            embs.append(e); labels.append(ident)
    genuine, impostor = pairwise_scores(embs, labels)
    assert genuine.size > 0 and impostor.size > 0
    assert genuine.mean() > impostor.mean() + 0.2   # same-identity clearly closer


def test_evaluate_end_to_end_low_eer_for_tight_clusters():
    dim = 128
    embs, labels = [], []
    for ident in range(6):
        for e in _cluster(dim, ident, 8, spread=0.15, start_seed=ident * 50):
            embs.append(e); labels.append(ident)
    r = evaluate(embs, labels)
    assert r.eer < 0.05          # tight, well-separated clusters verify near-perfectly
    assert r.auc > 0.98
