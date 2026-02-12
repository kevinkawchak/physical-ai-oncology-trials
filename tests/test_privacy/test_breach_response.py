"""Tests for Breach Response Protocol.

Validates incident reporting, four-factor risk assessment with score
clamping (v0.9.1 fix), notification timelines, and state-law overlays.

LICENSE: MIT
"""

from __future__ import annotations


import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "privacy/breach-response/breach_response_protocol.py"


@pytest.fixture()
def br_mod():
    return load_module("breach_response_protocol", MOD_PATH)


@pytest.fixture()
def brm(br_mod):
    return br_mod.BreachResponseManager(
        organization="Test Hospital",
        hipaa_covered_entity=True,
        state_jurisdictions=["CA", "NY"],
    )


@pytest.fixture()
def sample_incident(brm):
    return brm.report_incident(
        incident_type="unauthorized_access",
        description="Test incident",
        phi_types_involved=["medical_record_numbers"],
        individuals_affected=100,
        discovery_date="2026-02-01",
        affected_systems=["ehr"],
        containment_actions=["isolated network"],
    )


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_incident_type_values(self, br_mod):
        names = {e.name for e in br_mod.IncidentType}
        assert "UNAUTHORIZED_ACCESS" in names
        assert "RANSOMWARE" in names
        assert "AI_SYSTEM_BREACH" in names

    def test_risk_level_values(self, br_mod):
        names = {e.name for e in br_mod.RiskLevel}
        assert "LOW" in names
        assert "BREACH_CONFIRMED" in names

    def test_notification_recipients(self, br_mod):
        names = {e.name for e in br_mod.NotificationRecipient}
        assert "HHS_OCR" in names
        assert "FDA" in names
        assert "IRB" in names


# ===========================================================================
# TestIncident
# ===========================================================================


class TestIncident:
    def test_incident_creation(self, sample_incident, br_mod):
        assert sample_incident.incident_id.startswith("INC-")
        assert sample_incident.incident_type == br_mod.IncidentType.UNAUTHORIZED_ACCESS
        assert sample_incident.individuals_affected == 100

    def test_incident_to_dict(self, sample_incident):
        d = sample_incident.to_dict()
        assert "incident_id" in d
        assert "incident_type" in d
        assert d["status"] == "open"

    def test_incident_stored_in_manager(self, brm, sample_incident):
        assert sample_incident.incident_id in brm._incidents


# ===========================================================================
# TestRiskAssessment
# ===========================================================================


class TestRiskAssessment:
    def test_low_risk_score(self, br_mod):
        ra = br_mod.RiskAssessment(
            incident_id="I-1",
            assessment_date="2026-01-01",
            phi_nature_score=1,
            unauthorized_party_score=1,
            acquisition_score=1,
            mitigation_score=5,
        )
        ra.calculate_risk()
        assert ra.risk_level == br_mod.RiskLevel.LOW
        assert ra.is_reportable_breach is False

    def test_high_risk_score(self, br_mod):
        ra = br_mod.RiskAssessment(
            incident_id="I-2",
            assessment_date="2026-01-01",
            phi_nature_score=5,
            unauthorized_party_score=5,
            acquisition_score=5,
            mitigation_score=1,
        )
        ra.calculate_risk()
        assert ra.risk_level == br_mod.RiskLevel.BREACH_CONFIRMED
        assert ra.is_reportable_breach is True

    def test_medium_risk_score(self, br_mod):
        ra = br_mod.RiskAssessment(
            incident_id="I-3",
            assessment_date="2026-01-01",
            phi_nature_score=3,
            unauthorized_party_score=2,
            acquisition_score=3,
            mitigation_score=3,
        )
        ra.calculate_risk()
        assert ra.overall_risk_score > 0

    def test_v091_clamping_out_of_range_scores(self, br_mod):
        """v0.9.1 fix: scores outside [1,5] are clamped, not rejected."""
        ra = br_mod.RiskAssessment(
            incident_id="I-4",
            assessment_date="2026-01-01",
            phi_nature_score=10,
            unauthorized_party_score=-2,
            acquisition_score=0,
            mitigation_score=99,
        )
        ra.calculate_risk()
        assert ra.phi_nature_score == 5
        assert ra.unauthorized_party_score == 1
        assert ra.acquisition_score == 1
        assert ra.mitigation_score == 5
        assert 0 < ra.overall_risk_score <= 5

    def test_risk_assessment_via_manager(self, brm, sample_incident, br_mod):
        assessment = brm.assess_breach_risk(
            sample_incident,
            phi_nature_score=4,
            unauthorized_party_score=3,
            acquisition_score=4,
            mitigation_score=2,
        )
        assert assessment.incident_id == sample_incident.incident_id
        assert assessment.overall_risk_score > 0
        assert assessment.rationale != ""

    def test_mitigation_inversion(self, br_mod):
        """High mitigation score should lower overall risk."""
        ra_low_mit = br_mod.RiskAssessment(
            incident_id="I-A",
            assessment_date="d",
            phi_nature_score=3,
            unauthorized_party_score=3,
            acquisition_score=3,
            mitigation_score=1,
        )
        ra_high_mit = br_mod.RiskAssessment(
            incident_id="I-B",
            assessment_date="d",
            phi_nature_score=3,
            unauthorized_party_score=3,
            acquisition_score=3,
            mitigation_score=5,
        )
        ra_low_mit.calculate_risk()
        ra_high_mit.calculate_risk()
        assert ra_high_mit.overall_risk_score < ra_low_mit.overall_risk_score


# ===========================================================================
# TestNotificationTimeline
# ===========================================================================


class TestNotificationTimeline:
    def test_timeline_has_individual_notification(self, brm, sample_incident):
        tl = brm.generate_notification_timeline(sample_incident)
        recipients = [d.recipient for d in tl.deadlines]
        assert any("Individual" in r for r in recipients)

    def test_large_breach_includes_media(self, brm):
        big_incident = brm.report_incident(
            incident_type="theft",
            description="Large breach",
            phi_types_involved=["ssn"],
            individuals_affected=600,
            discovery_date="2026-02-01",
        )
        tl = brm.generate_notification_timeline(big_incident)
        recipients = [d.recipient for d in tl.deadlines]
        assert any("Media" in r for r in recipients)

    def test_small_breach_no_media(self, brm, sample_incident):
        tl = brm.generate_notification_timeline(sample_incident)
        recipients = [d.recipient for d in tl.deadlines]
        assert not any("Media" in r for r in recipients)

    def test_state_specific_deadlines(self, brm, sample_incident):
        tl = brm.generate_notification_timeline(sample_incident)
        recipients = [d.recipient for d in tl.deadlines]
        # NY AG threshold is 0, so should always be present
        assert any("New York" in r for r in recipients)

    def test_timeline_includes_sponsor_and_irb(self, brm, sample_incident):
        tl = brm.generate_notification_timeline(sample_incident)
        recipients = [d.recipient for d in tl.deadlines]
        assert any("Sponsor" in r for r in recipients)
        assert any("IRB" in r for r in recipients)

    def test_timeline_sorted_by_due_date(self, brm, sample_incident):
        tl = brm.generate_notification_timeline(sample_incident)
        dates = [d.due_date for d in tl.deadlines]
        assert dates == sorted(dates)

    def test_timeline_to_dict(self, brm, sample_incident):
        tl = brm.generate_notification_timeline(sample_incident)
        d = tl.to_dict()
        assert "incident_id" in d
        assert "deadlines" in d
        assert len(d["deadlines"]) > 0


# ===========================================================================
# TestStateLaws
# ===========================================================================


class TestStateLaws:
    def test_state_breach_laws_defined(self, br_mod):
        laws = br_mod.STATE_BREACH_LAWS
        assert "CA" in laws
        assert "MA" in laws
        assert laws["CA"]["notification_days"] == 45
        assert laws["MA"]["notification_days"] == 30
