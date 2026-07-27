"""Integration test on a real face image.

Runs the actual MediaPipe landmarker + periocular crop + embedder on a bundled
photograph (skimage.data.astronaut, which contains a real human face). Skipped
automatically if mediapipe / opencv / skimage are not installed, so the pure-
logic suite still runs everywhere.
"""
import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")
skimage_data = pytest.importorskip("skimage.data")

from cranium.landmarks import FaceMeshLandmarker, default_model_path
from cranium.periocular import crop_periocular
from cranium.embedding import StubEmbedder
from cranium.pipeline import PeriocularPipeline
from cranium.config import EMBED_DIM

if default_model_path() is None:
    pytest.skip("face_landmarker.task model not available", allow_module_level=True)


@pytest.fixture(scope="module")
def astronaut_bgr():
    rgb = skimage_data.astronaut()          # (512, 512, 3) uint8 RGB, has a face
    return np.ascontiguousarray(rgb[:, :, ::-1])  # -> BGR


def test_landmarker_finds_a_face(astronaut_bgr):
    with FaceMeshLandmarker() as lm:
        pts = lm.detect(astronaut_bgr)
    assert pts is not None, "expected to detect a face in the astronaut image"
    assert pts.shape[0] >= 468
    assert pts.shape[1] == 2


def test_periocular_crop_is_upper_face(astronaut_bgr):
    with FaceMeshLandmarker() as lm:
        pts = lm.detect(astronaut_bgr)
    crop, (x1, y1, x2, y2) = crop_periocular(astronaut_bgr, pts)
    assert crop.size > 0
    # the crop is a strict sub-region of the full frame
    full_h, full_w = astronaut_bgr.shape[:2]
    assert (x2 - x1) < full_w
    assert (y2 - y1) < full_h


def test_full_pipeline_produces_normalized_embedding(astronaut_bgr):
    with FaceMeshLandmarker() as lm:
        pipe = PeriocularPipeline(lm, StubEmbedder(EMBED_DIM))
        result = pipe.embed(astronaut_bgr)
    assert result is not None
    assert result.embedding.shape == (EMBED_DIM,)
    assert float(np.linalg.norm(result.embedding)) == pytest.approx(1.0, abs=1e-5)


def test_same_image_same_embedding_stub(astronaut_bgr):
    """StubEmbedder is deterministic -> identical crops give identical vectors."""
    with FaceMeshLandmarker() as lm:
        pipe = PeriocularPipeline(lm, StubEmbedder(EMBED_DIM))
        a = pipe.embed(astronaut_bgr).embedding
        b = pipe.embed(astronaut_bgr).embedding
    np.testing.assert_allclose(a, b, atol=1e-6)
