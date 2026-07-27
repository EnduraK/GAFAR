import numpy as np
import pytest

from cranium.verifier import Verifier
from cranium.authorization import (
    AuthorizationEngine,
    AuthorizationRequest,
    AuditLog,
)
from tests.conftest import identity_vector, sample_of

DIM = 128


def _engine_with_alice(sample_sim=0.9):
    """Enroll Alice from genuine captures (high cosine to her identity) so the
    averaged template sits essentially on her identity direction."""
    alice = identity_vector(DIM, 10)
    v = Verifier()
    v.enroll("alice", [sample_of(alice, sample_sim, s) for s in range(200, 205)])
    return AuthorizationEngine(v), alice


def test_genuine_with_liveness_approved():
    engine, alice = _engine_with_alice()
    req = AuthorizationRequest("release_medication", "alice", risk_level="standard",
                               resource="morphine 10mg")
    probe = sample_of(alice, 0.85, 1)
    decision = engine.authorize(req, probe, liveness_passed=True)
    assert decision.approved is True


def test_genuine_without_liveness_denied_at_standard():
    engine, alice = _engine_with_alice()
    req = AuthorizationRequest("release_medication", "alice", risk_level="standard")
    probe = sample_of(alice, 0.85, 1)
    decision = engine.authorize(req, probe, liveness_passed=False)
    assert decision.approved is False
    assert any("liveness" in r for r in decision.reasons)


def test_impostor_denied():
    engine, _ = _engine_with_alice()
    bob = identity_vector(DIM, 20)
    req = AuthorizationRequest("approve_transfer", "alice", risk_level="standard",
                               resource="£25,000")
    decision = engine.authorize(req, sample_of(bob, 0.85, 1), liveness_passed=True)
    assert decision.approved is False
    assert any("score" in r for r in decision.reasons)


def test_unenrolled_subject_denied():
    engine, _ = _engine_with_alice()
    req = AuthorizationRequest("unlock_machine", "stranger", risk_level="low")
    decision = engine.authorize(req, identity_vector(DIM, 99), liveness_passed=True)
    assert decision.approved is False
    assert any("not enrolled" in r for r in decision.reasons)


def test_risk_level_raises_the_bar():
    """A borderline probe that clears 'standard' (0.62) fails 'critical' (0.78)."""
    alice = identity_vector(DIM, 10)
    v = Verifier()
    v.enroll("alice", [alice])                 # template == identity exactly
    engine = AuthorizationEngine(v)
    probe = sample_of(alice, 0.70, 3)          # score ~= 0.70, between the two thresholds
    score = engine.verifier.verify("alice", probe).score
    assert 0.62 <= score < 0.78, f"probe score {score:.3f} not in the target band"

    ok = engine.authorize(
        AuthorizationRequest("approve_transfer", "alice", risk_level="standard"),
        probe, liveness_passed=True)
    blocked = engine.authorize(
        AuthorizationRequest("approve_transfer", "alice", risk_level="critical"),
        probe, liveness_passed=True)
    assert ok.approved is True
    assert blocked.approved is False


def test_low_risk_skips_liveness():
    engine, alice = _engine_with_alice()
    req = AuthorizationRequest("view_record", "alice", risk_level="low")
    decision = engine.authorize(req, sample_of(alice, 0.85, 1), liveness_passed=False)
    assert decision.approved is True  # low risk does not require liveness


def test_audit_chain_records_every_event_and_verifies():
    engine, alice = _engine_with_alice()
    for i in range(4):
        engine.authorize(
            AuthorizationRequest("release_medication", "alice"),
            sample_of(alice, 0.85, i), liveness_passed=True)
    assert len(engine.audit.records) == 4
    assert engine.audit.verify_chain() is True


def test_audit_tamper_is_detected():
    engine, alice = _engine_with_alice()
    engine.authorize(AuthorizationRequest("release_medication", "alice"),
                     sample_of(alice, 0.85, 1), liveness_passed=True)
    engine.authorize(AuthorizationRequest("release_medication", "alice"),
                     sample_of(alice, 0.85, 2), liveness_passed=True)
    # retroactively alter a stored score -> must break the hash chain
    engine.audit._records[0]["decision"]["score"] = 0.999123
    assert engine.audit.verify_chain() is False
