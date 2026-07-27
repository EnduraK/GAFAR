import numpy as np
import pytest

from cranium.verifier import Verifier, TemplateStore, cosine_similarity
from cranium.embedding import l2_normalize
from tests.conftest import identity_vector, sample_of

DIM = 128


def test_cosine_identical_is_one():
    v = identity_vector(DIM, 1)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_symmetric():
    a = identity_vector(DIM, 1)
    b = identity_vector(DIM, 2)
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-6)


def test_genuine_matches_impostor_rejected():
    alice = identity_vector(DIM, 10)
    bob = identity_vector(DIM, 20)
    v = Verifier(threshold=0.5)
    # enroll Alice from several genuine captures (cosine ~0.85 to her identity)
    samples = [sample_of(alice, 0.85, s) for s in range(100, 105)]
    v.enroll("alice", samples)

    genuine = sample_of(alice, 0.85, 999)
    impostor = sample_of(bob, 0.85, 999)   # ~0.85 to Bob, ~0 to Alice

    assert v.verify("alice", genuine).is_match is True
    assert v.verify("alice", impostor).is_match is False


def test_threshold_boundary_controls_decision():
    alice = identity_vector(DIM, 10)
    v = Verifier()
    v.enroll("alice", [alice])
    probe = sample_of(alice, 0.6, 7)  # moderately noisy
    score = v.verify("alice", probe).score
    # a threshold just below the score accepts; just above rejects
    assert v.verify("alice", probe, threshold=score - 0.01).is_match is True
    assert v.verify("alice", probe, threshold=score + 0.01).is_match is False


def test_unknown_subject_returns_no_match():
    v = Verifier()
    res = v.verify("nobody", identity_vector(DIM, 3))
    assert res.is_match is False
    assert res.score == 0.0


def test_enroll_requires_samples():
    v = Verifier()
    with pytest.raises(ValueError):
        v.enroll("x", [])


def test_enroll_averages_and_normalizes():
    alice = identity_vector(DIM, 10)
    v = Verifier()
    tmpl = v.enroll("alice", [sample_of(alice, 0.8, s) for s in range(5)])
    assert tmpl.n_samples == 5
    assert np.linalg.norm(tmpl.vector) == pytest.approx(1.0, abs=1e-5)


def test_template_store_roundtrip(tmp_path):
    alice = identity_vector(DIM, 10)
    v = Verifier()
    v.enroll("alice", [alice], meta={"role": "pharmacist"})
    path = tmp_path / "templates.json"
    v.store.save(path)

    loaded = TemplateStore.load(path)
    assert loaded.has("alice")
    t = loaded.get("alice")
    assert t.meta["role"] == "pharmacist"
    np.testing.assert_allclose(t.vector, v.store.get("alice").vector, atol=1e-6)


def test_store_delete_and_subjects():
    v = Verifier()
    v.enroll("a", [identity_vector(DIM, 1)])
    v.enroll("b", [identity_vector(DIM, 2)])
    assert set(v.store.subjects()) == {"a", "b"}
    assert v.store.delete("a") is True
    assert v.store.delete("a") is False
    assert v.store.subjects() == ["b"]


def test_no_raw_image_stored():
    """Privacy-by-design: the template exposes only a vector, never pixels."""
    v = Verifier()
    tmpl = v.enroll("a", [identity_vector(DIM, 1)])
    fields = set(tmpl.__dataclass_fields__.keys())
    assert "vector" in fields
    assert not fields & {"image", "crop", "pixels", "frame"}
