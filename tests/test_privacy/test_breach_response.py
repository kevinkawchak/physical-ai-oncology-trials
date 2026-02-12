"""Unit tests for privacy/breach-response/breach_response_protocol.py.

Tests cover IncidentType, RiskLevel, NotificationRecipient enums,
Incident, RiskAssessment dataclasses, and risk calculation.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import load_module

MOD_PATH = "privacy/breach-response/breach_response_protocol.py"


@pytest.fixture()
def mod():
    return load_module("breach_response_protocol", MOD_PATH)


class TestEnums:
    def test_incident_type(self, mod):
        names = [e.name for e in mod.IncidentType]
        for t in ("UNAUTHORIZED_ACCESS", "RANSOMWARE", "AI_SYSTEM_BREACH"):
            assert t in names

    def test_risk_level(self, mod):
        names = [e.name for e in mod.RiskLevel]
        for r in ("LOW", "MEDIUM", "HIGH", "BREACH_CONFIRMED"):
            assert r in names

    def test_notification_recipient(self, mod):
        names = [e.name for e in mod.NotificationRecipient]
        for r in ("INDIVIDUALS", "HHS_OCR", "MEDIA"):
            assert r in names


class TestIncident:
    def test_creation(self, mod):
        incident = mod.Incident(
            incident_id="INC-001",
            incident_type=mod.IncidentType.UNAUTHORIZED_ACCESS,
            description="Unauthorized access to patient records",
            phi_types_involved=["NAME", "MRN", "DIAGNOSIS"],
            individuals_affected=50,
            discovery_date=datetime(2026, 2, 12).isoformat(),
            discovery_method="audit_log_review",
            affected_systems=["EHR", "PACS"],
        )
        assert incident.incident_id == "INC-001"
        assert incident.individuals_affected == 50

    def test_to_dict(self, mod):
        incident = mod.Incident(
            incident_id="INC-002",
            incident_type=mod.IncidentType.RANSOMWARE,
            description="Ransomware detected",
            phi_types_involved=["ALL"],
            individuals_affected=1000,
            discovery_date=datetime(2026, 2, 12).isoformat(),
            discovery_method="automated_alert",
            affected_systems=["ALL"],
        )
        d = incident.to_dict()
        assert isinstance(d, dict)
        assert d["incident_id"] == "INC-002"


class TestRiskAssessment:
    def test_creation(self, mod):
        assessment = mod.RiskAssessment(
            incident_id="INC-001",
            assessment_date=datetime(2026, 2, 12).isoformat(),
            phi_nature_score=4,
            unauthorized_party_score=3,
            acquisition_score=2,
            mitigation_score=3,
        )
        assert assessment.incident_id == "INC-001"

    def test_calculate_risk(self, mod):
        assessment = mod.RiskAssessment(
            incident_id="INC-CALC",
            assessment_date=datetime(2026, 2, 12).isoformat(),
            phi_nature_score=4,
            unauthorized_party_score=4,
            acquisition_score=3,
            mitigation_score=2,
        )
        assessment.calculate_risk()
        assert assessment.risk_level is not None
        assert assessment.overall_risk_score > 0

    def test_low_risk_assessment(self, mod):
        assessment = mod.RiskAssessment(
            incident_id="INC-LOW",
            assessment_date=datetime(2026, 2, 12).isoformat(),
            phi_nature_score=1,
            unauthorized_party_score=1,
            acquisition_score=1,
            mitigation_score=5,
        )
        assessment.calculate_risk()
        assert assessment.risk_level in (mod.RiskLevel.LOW, mod.RiskLevel.MEDIUM)

    def test_score_clamping(self, mod):
        """Regression: Out-of-range scores should be clamped."""
        assessment = mod.RiskAssessment(
            incident_id="INC-CLAMP",
            assessment_date=datetime(2026, 2, 12).isoformat(),
            phi_nature_score=10,  # Out of range
            unauthorized_party_score=0,  # Out of range
            acquisition_score=3,
            mitigation_score=3,
        )
        assessment.calculate_risk()
        # Should not crash, risk level should be set
        assert assessment.risk_level is not None
