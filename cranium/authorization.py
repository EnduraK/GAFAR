"""The authorisation-event engine.

This is the generic core the brief argues for: a signed request to perform a
high-stakes action (release medication X, approve transfer Y, unlock machine Z)
that requires upper-face confirmation above a risk-dependent policy, with a
tamper-evident audit trail. The banking and medication demos are just different
requests through this same engine.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .config import RISK_POLICIES, RiskPolicy
from .verifier import Verifier


@dataclass
class AuthorizationRequest:
    action: str                     # "release_medication", "approve_transfer", ...
    claimed_subject_id: str         # identity asserted by badge / username (2nd factor)
    risk_level: str = "standard"
    resource: str = ""              # "morphine 10 mg", "£25,000 -> acct ****1234"
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuthorizationDecision:
    approved: bool
    action: str
    claimed_subject_id: str
    risk_level: str
    resource: str
    score: float
    threshold: float
    liveness_required: bool
    liveness_passed: bool
    reasons: List[str]
    timestamp: float


class AuditLog:
    """Append-only, SHA-256 hash-chained log. Each record commits to the prior
    record's hash, so any retroactive edit breaks the chain — the property a
    controlled-drug register needs (attributable, unalterable, auditable)."""

    def __init__(self) -> None:
        self._records: List[dict] = []

    @property
    def records(self) -> List[dict]:
        return list(self._records)

    def _last_hash(self) -> str:
        return self._records[-1]["hash"] if self._records else "0" * 64

    def append(self, decision: AuthorizationDecision) -> dict:
        payload = asdict(decision)
        prev = self._last_hash()
        body = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256((prev + body).encode()).hexdigest()
        record = {"prev": prev, "hash": digest, "decision": payload}
        self._records.append(record)
        return record

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for record in self._records:
            body = json.dumps(record["decision"], sort_keys=True)
            if record["prev"] != prev:
                return False
            if record["hash"] != hashlib.sha256((prev + body).encode()).hexdigest():
                return False
            prev = record["hash"]
        return True


class AuthorizationEngine:
    def __init__(self, verifier: Verifier, audit: Optional[AuditLog] = None) -> None:
        self.verifier = verifier
        self.audit = audit or AuditLog()

    def policy_for(self, risk_level: str) -> RiskPolicy:
        return RISK_POLICIES.get(risk_level, RISK_POLICIES["standard"])

    def authorize(self, request: AuthorizationRequest, probe: np.ndarray,
                  liveness_passed: bool = False) -> AuthorizationDecision:
        policy = self.policy_for(request.risk_level)
        enrolled = self.verifier.store.has(request.claimed_subject_id)
        vres = self.verifier.verify(request.claimed_subject_id, probe,
                                    threshold=policy.match_threshold)

        reasons: List[str] = []
        approved = True

        if not enrolled:
            approved = False
            reasons.append("subject not enrolled")
        elif vres.is_match:
            reasons.append(
                f"biometric match {vres.score:.3f} >= {policy.match_threshold:.3f}")
        else:
            approved = False
            reasons.append(
                f"biometric score {vres.score:.3f} < threshold {policy.match_threshold:.3f}")

        if policy.require_liveness and not liveness_passed:
            approved = False
            reasons.append("liveness required but not confirmed")
        elif policy.require_liveness:
            reasons.append("liveness confirmed")

        decision = AuthorizationDecision(
            approved=approved,
            action=request.action,
            claimed_subject_id=request.claimed_subject_id,
            risk_level=request.risk_level,
            resource=request.resource,
            score=vres.score,
            threshold=policy.match_threshold,
            liveness_required=policy.require_liveness,
            liveness_passed=liveness_passed,
            reasons=reasons,
            timestamp=request.timestamp,
        )
        self.audit.append(decision)
        return decision
