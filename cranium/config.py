"""Central configuration: thresholds, ROI geometry, risk policies.

Every numeric operating point in this file is a PROTOTYPE PLACEHOLDER. Real
thresholds come from an ROC/DET study on collected periocular data — that is
the Phase 1 gate in the concept brief (target EER < 3% on webcam-quality
captures). Do not treat these numbers as validated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# Dimensionality of the face/periocular embedding vector.
EMBED_DIM: int = 128

# Cosine-similarity match threshold for L2-normalized vectors (dot product in
# [-1, 1]). Higher = stricter. Placeholder operating point only.
DEFAULT_MATCH_THRESHOLD: float = 0.62


@dataclass(frozen=True)
class RiskPolicy:
    """The policy layer the brief calls the 'authorisation event': higher-stakes
    actions demand a stricter biometric score AND live-presence confirmation."""
    name: str
    match_threshold: float
    require_liveness: bool
    require_second_factor: bool  # badge / username asserting the claimed identity


# Per-risk operating points. A £50 approval and a controlled-drug release are
# not the same event; the policy makes that explicit.
RISK_POLICIES: Dict[str, RiskPolicy] = {
    "low":      RiskPolicy("low",      0.55, False, True),
    "standard": RiskPolicy("standard", 0.62, True,  True),
    "high":     RiskPolicy("high",     0.70, True,  True),
    "critical": RiskPolicy("critical", 0.78, True,  True),
}

# --- Periocular region of interest -----------------------------------------
# MediaPipe FaceMesh landmark indices bounding the upper face: eyes, brows,
# nose bridge, forehead centre, and temples. This is the region CRANIUM matches
# on so it survives a covered lower face (mask / respirator / veil).
PERIOCULAR_LANDMARKS: List[int] = [
    # eyes
    33, 133, 160, 158, 153, 144, 362, 263, 385, 387, 373, 380,
    # brows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # nose bridge
    168, 6, 197, 195,
    # forehead centre + temples
    10, 109, 338, 67, 297, 127, 356,
]
ROI_EXPAND: float = 0.15  # fractional padding around the tight bounding box

# --- Blink / attention liveness (v0) ---------------------------------------
# Six-point Eye-Aspect-Ratio (EAR) landmark indices per eye.
LEFT_EYE_EAR: List[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR: List[int] = [362, 385, 387, 263, 373, 380]
EAR_BLINK_THRESHOLD: float = 0.20  # below -> eye considered closed
EAR_OPEN_THRESHOLD: float = 0.25   # above -> eye considered open
BLINK_WINDOW_S: float = 4.0        # complete a blink within this many seconds
