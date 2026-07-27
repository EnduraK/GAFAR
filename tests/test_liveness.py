import numpy as np

from cranium.liveness import BlinkChallenge, eye_aspect_ratios, _ear
from cranium.config import LEFT_EYE_EAR, RIGHT_EYE_EAR


def _eye(open_ratio):
    """Six EAR points (p1..p6). Vertical gap scales with `open_ratio`."""
    w = 30.0
    gap = open_ratio * w
    cx, cy = 100.0, 100.0
    return np.array([
        [cx - w / 2, cy],            # p1 left corner
        [cx - w / 4, cy - gap / 2],  # p2 top
        [cx + w / 4, cy - gap / 2],  # p3 top
        [cx + w / 2, cy],            # p4 right corner
        [cx + w / 4, cy + gap / 2],  # p5 bottom
        [cx - w / 4, cy + gap / 2],  # p6 bottom
    ], dtype=np.float32)


def test_ear_open_greater_than_closed():
    assert _ear(_eye(0.6)) > _ear(_eye(0.05))


def test_eye_aspect_ratios_reads_both_eyes():
    n = max(max(LEFT_EYE_EAR), max(RIGHT_EYE_EAR)) + 1
    lm = np.zeros((n, 2), dtype=np.float32)
    lm[LEFT_EYE_EAR] = _eye(0.5)
    lm[RIGHT_EYE_EAR] = _eye(0.5)
    left, right, avg = eye_aspect_ratios(lm)
    assert left > 0 and right > 0
    assert avg == (left + right) / 2.0


def test_blink_detected_on_close_then_open():
    ch = BlinkChallenge(required_blinks=1, window_s=10.0)
    open_ear, closed_ear = 0.35, 0.10
    seq = [(open_ear, 0.0), (open_ear, 0.1),
           (closed_ear, 0.2), (closed_ear, 0.3),   # eyes shut
           (open_ear, 0.4)]                          # reopened -> 1 blink
    result = None
    for ear, t in seq:
        result = ch.update(ear, t)
    assert result.passed is True
    assert result.blinks == 1


def test_no_blink_times_out():
    ch = BlinkChallenge(required_blinks=1, window_s=2.0)
    result = None
    for k in range(30):
        result = ch.update(0.35, k * 0.2)  # eyes stay open past the window
    assert result.passed is False
    assert "timeout" in result.reason


def test_reset_clears_state():
    ch = BlinkChallenge()
    ch.update(0.35, 0.0)
    ch.update(0.10, 0.1)
    ch.reset()
    assert ch.blinks == 0
    assert ch._t0 is None
