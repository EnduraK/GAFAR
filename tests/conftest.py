"""Shared test helpers."""
import numpy as np
import pytest

from cranium.embedding import l2_normalize


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def identity_vector(dim, seed):
    """A stable 'true' identity direction."""
    r = np.random.default_rng(seed)
    return l2_normalize(r.standard_normal(dim))


def sample_of(base, similarity, seed):
    """A synthetic 'capture' of an identity with a CONTROLLED cosine similarity
    to `base`. Builds a unit vector = s*base + sqrt(1-s^2)*u, where u is a unit
    vector orthogonal to base, so cosine(base, result) ~= s exactly.

    (Naive `base + noise*randn` does NOT work in high dimensions: a 128-D noise
    vector has norm ~sqrt(128)*noise, which swamps a unit base even for small
    noise. That bug is why an earlier version of these tests failed.)
    """
    r = np.random.default_rng(seed)
    base = l2_normalize(base)
    d = base.shape[0]
    u = r.standard_normal(d).astype(np.float32)
    u = u - float(u.dot(base)) * base      # remove component along base
    nu = float(np.linalg.norm(u))
    u = u / nu if nu > 0 else u
    s = float(np.clip(similarity, -1.0, 1.0))
    return l2_normalize(s * base + np.sqrt(max(0.0, 1.0 - s * s)) * u)


# expose helpers as fixtures too
@pytest.fixture
def make_identity():
    return identity_vector


@pytest.fixture
def make_sample():
    return sample_of
