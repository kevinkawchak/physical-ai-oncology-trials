"""Tests for privacy/breach-response/breach_response_protocol.py

Covers IncidentType, RiskAssessment, and BreachResponseManager.
"""

from __future__ import annotations


class TestEnums:
    def test_incident_types(self, breach_mod):
        it = breach_mod.IncidentType
        assert it.UNAUTHORIZED_ACCESS.value == "unauthorized_access"
        assert it.RANSOMWARE.value == "ransomware"
        assert it.AI_SYSTEM_BREACH.value == "ai_system_breach"

    def test_risk_levels(self, breach_mod):
        rl = breach_mod.RiskLevel
        assert rl.LOW.value == "low_probability_of_compromise"
        assert rl.BREACH_CONFIRMED.value == "breach_confirmed"


class TestIncident:
    def test_creation(self, breach_mod):
        incident = breach_mod.Incident(
            incident_id="INC-001",
            incident_type=breach_mod.IncidentType.UNAUTHORIZED_ACCESS,
            description="Unauthorized access to PHI",
            phi_types_involved=["name", "mrn"],
            individuals_affected=10,
            discovery_date="2026-03-01",
        )
        assert incident.incident_id == "INC-001"
        assert incident.individuals_affected == 10

    def test_to_dict(self, breach_mod):
        incident = breach_mod.Incident(
            incident_id="INC-002",
            incident_type=breach_mod.IncidentType.THEFT,
            description="Device theft",
            phi_types_involved=["ssn"],
            individuals_affected=1,
            discovery_date="2026-03-01",
        )
        d = incident.to_dict()
        assert isinstance(d, dict)
        assert d["incident_id"] == "INC-002"


class TestRiskAssessment:
    def test_calculate_risk(self, breach_mod):
        assessment = breach_mod.RiskAssessment(
            incident_id="RA-001",
            assessment_date="2026-03-01",
            phi_nature_score=3,
            unauthorized_party_score=3,
            acquisition_score=3,
            mitigation_score=3,
        )
        assessment.calculate_risk()
        assert assessment.overall_risk_score > 0
        assert assessment.risk_level is not None

    def test_low_risk(self, breach_mod):
        assessment = breach_mod.RiskAssessment(
            incident_id="RA-002",
            assessment_date="2026-03-01",
            phi_nature_score=1,
            unauthorized_party_score=1,
            acquisition_score=1,
            mitigation_score=1,
        )
        assessment.calculate_risk()
        assert assessment.risk_level == breach_mod.RiskLevel.LOW


class TestBreachResponseManager:
    def test_creation(self, breach_mod):
        mgr = breach_mod.BreachResponseManager(organization="Test Hospital")
        assert mgr.organization == "Test Hospital"

    def test_report_incident(self, breach_mod):
        mgr = breach_mod.BreachResponseManager(organization="Test Clinic")
        incident = mgr.report_incident(
            incident_type="unauthorized_access",
            description="Staff accessed records outside scope",
            phi_types_involved=["name", "diagnosis"],
            individuals_affected=5,
            discovery_date="2026-03-01",
        )
        assert isinstance(incident, breach_mod.Incident)
        assert incident.individuals_affected == 5

    def test_generate_notification_timeline(self, breach_mod):
        mgr = breach_mod.BreachResponseManager(organization="Test Hospital")
        incident = mgr.report_incident(
            incident_type="unauthorized_disclosure",
            description="PHI disclosed to unauthorized party",
            phi_types_involved=["ssn", "diagnosis"],
            individuals_affected=500,
            discovery_date="2026-03-01",
        )
        timeline = mgr.generate_notification_timeline(incident)
        assert isinstance(timeline, breach_mod.NotificationTimeline)
        assert len(timeline.deadlines) > 0
