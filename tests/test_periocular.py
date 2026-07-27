import numpy as np

from cranium.periocular import periocular_roi, crop_periocular
from cranium.config import PERIOCULAR_LANDMARKS


def _fake_landmarks(n=478, box=(100, 120, 300, 260)):
    """Place the periocular indices inside a known box; scatter the rest."""
    lm = np.zeros((n, 2), dtype=np.float32)
    rng = np.random.default_rng(0)
    lm[:] = rng.uniform(0, 640, size=(n, 2))
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y2], [x1, y2], [x2, y1]], dtype=np.float32)
    for i, idx in enumerate(PERIOCULAR_LANDMARKS):
        lm[idx] = corners[i % 4]
    return lm


def test_roi_covers_periocular_landmarks():
    lm = _fake_landmarks(box=(100, 120, 300, 260))
    x1, y1, x2, y2 = periocular_roi(lm, (480, 640, 3), expand=0.0)
    # with zero expansion the box should hug the landmark extremes
    assert x1 <= 100 and x2 >= 300
    assert y2 >= 260
    assert y1 <= 120


def test_roi_is_clamped_to_image():
    lm = _fake_landmarks(box=(0, 0, 5, 5))
    x1, y1, x2, y2 = periocular_roi(lm, (480, 640, 3), expand=0.5)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 640 and y2 <= 480


def test_expand_grows_the_box():
    lm = _fake_landmarks(box=(200, 200, 300, 300))
    tight = periocular_roi(lm, (600, 600, 3), expand=0.0)
    padded = periocular_roi(lm, (600, 600, 3), expand=0.3)
    assert (padded[2] - padded[0]) >= (tight[2] - tight[0])
    assert (padded[3] - padded[1]) >= (tight[3] - tight[1])


def test_crop_returns_subimage():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = _fake_landmarks(box=(100, 120, 300, 260))
    crop, roi = crop_periocular(img, lm)
    x1, y1, x2, y2 = roi
    assert crop.shape[0] == (y2 - y1)
    assert crop.shape[1] == (x2 - x1)
    assert crop.size > 0
