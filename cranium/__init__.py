"""CRANIUM — upper-face (periocular) verification + authorisation engine (MVP).

Prototype scope: 1:1 verification only. Logic layers (verifier, authorisation,
liveness state machine) are unit-tested; the embedding-model quality and the
live-webcam paths require real captures to evaluate.
"""
from .config import RISK_POLICIES, RiskPolicy, DEFAULT_MATCH_THRESHOLD
from .embedding import Embedder, StubEmbedder, DlibEmbedder, l2_normalize
from .verifier import (
    Verifier,
    Template,
    TemplateStore,
    VerificationResult,
    cosine_similarity,
)
from .authorization import (
    AuthorizationEngine,
    AuthorizationRequest,
    AuthorizationDecision,
    AuditLog,
)
from .liveness import BlinkChallenge, BlinkResult, eye_aspect_ratios
from .pipeline import PeriocularPipeline, FaceResult

__version__ = "0.1.0"

__all__ = [
    "RISK_POLICIES", "RiskPolicy", "DEFAULT_MATCH_THRESHOLD",
    "Embedder", "StubEmbedder", "DlibEmbedder", "l2_normalize",
    "Verifier", "Template", "TemplateStore", "VerificationResult", "cosine_similarity",
    "AuthorizationEngine", "AuthorizationRequest", "AuthorizationDecision", "AuditLog",
    "BlinkChallenge", "BlinkResult", "eye_aspect_ratios",
    "PeriocularPipeline", "FaceResult",
    "__version__",
]
