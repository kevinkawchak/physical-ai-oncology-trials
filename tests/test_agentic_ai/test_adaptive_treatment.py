"""Tests for agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py

Covers stream processing, anomaly detection, cross-modal correlation,
and treatment decision engine.
"""

from __future__ import annotations

import time


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_stream_types(self, adaptive_agent_mod):
        st = adaptive_agent_mod.StreamType
        assert st.FORCE_TORQUE.value == "force_torque"
        assert st.PATIENT_VITALS.value == "patient_vitals"
        assert st.IMAGING.value == "imaging"

    def test_alert_severity(self, adaptive_agent_mod):
        sev = adaptive_agent_mod.AlertSeverity
        assert sev.INFO.value == "info"
        assert sev.EMERGENCY.value == "emergency"

    def test_treatment_actions(self, adaptive_agent_mod):
        ta = adaptive_agent_mod.TreatmentAction
        assert ta.CONTINUE.value == "continue_as_planned"
        assert ta.ABORT.value == "abort_procedure"
        assert ta.PAUSE.value == "pause_procedure"


# ── DataSample dataclass ────────────────────────────────────────────────


class TestDataSample:
    def test_creation(self, adaptive_agent_mod):
        sample = adaptive_agent_mod.DataSample(
            stream_type=adaptive_agent_mod.StreamType.FORCE_TORQUE,
            timestamp=1000.0,
            values={"fx": 1.0, "fy": 0.0, "fz": 0.5},
        )
        assert sample.stream_type == adaptive_agent_mod.StreamType.FORCE_TORQUE
        assert sample.values["fx"] == 1.0


# ── TreatmentRecommendation dataclass ────────────────────────────────────


class TestTreatmentRecommendation:
    def test_creation(self, adaptive_agent_mod):
        rec = adaptive_agent_mod.TreatmentRecommendation(
            action=adaptive_agent_mod.TreatmentAction.PAUSE,
            severity=adaptive_agent_mod.AlertSeverity.WARNING,
            rationale="Force exceeds threshold",
        )
        assert rec.action == adaptive_agent_mod.TreatmentAction.PAUSE
        assert rec.requires_confirmation is True

    def test_to_dict(self, adaptive_agent_mod):
        rec = adaptive_agent_mod.TreatmentRecommendation(
            action=adaptive_agent_mod.TreatmentAction.CONTINUE,
            severity=adaptive_agent_mod.AlertSeverity.INFO,
            rationale="All nominal",
        )
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert "action" in d
        assert "severity" in d


# ── StreamBuffer ─────────────────────────────────────────────────────────


class TestStreamBuffer:
    def test_creation(self, adaptive_agent_mod):
        buf = adaptive_agent_mod.StreamBuffer(
            stream_type=adaptive_agent_mod.StreamType.PATIENT_VITALS,
        )
        assert buf.sample_count == 0

    def test_add_and_get(self, adaptive_agent_mod):
        buf = adaptive_agent_mod.StreamBuffer(
            stream_type=adaptive_agent_mod.StreamType.PATIENT_VITALS,
        )
        sample = adaptive_agent_mod.DataSample(
            stream_type=adaptive_agent_mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"hr": 72},
        )
        buf.add_sample(sample)
        assert buf.sample_count == 1
        assert buf.get_latest() is sample


# ── AdaptiveTreatmentAgent ───────────────────────────────────────────────


class TestAdaptiveTreatmentAgent:
    def test_creation(self, adaptive_agent_mod):
        agent = adaptive_agent_mod.AdaptiveTreatmentAgent(
            patient_id="ATA-001",
            procedure_type="lobectomy",
        )
        assert agent.patient_id == "ATA-001"

    def test_ingest_sample_returns_list(self, adaptive_agent_mod):
        agent = adaptive_agent_mod.AdaptiveTreatmentAgent(
            patient_id="ATA-002",
            procedure_type="lobectomy",
        )
        sample = adaptive_agent_mod.DataSample(
            stream_type=adaptive_agent_mod.StreamType.FORCE_TORQUE,
            timestamp=1.0,
            values={"fx": 0.5, "fy": 0.0, "fz": 0.2},
        )
        recs = agent.ingest_sample(sample)
        assert isinstance(recs, list)

    def test_get_status(self, adaptive_agent_mod):
        agent = adaptive_agent_mod.AdaptiveTreatmentAgent(
            patient_id="ATA-003",
            procedure_type="nephrectomy",
        )
        status = agent.get_status()
        assert isinstance(status, dict)
        assert "patient_id" in status

    def test_get_decision_log(self, adaptive_agent_mod):
        agent = adaptive_agent_mod.AdaptiveTreatmentAgent(
            patient_id="ATA-004",
            procedure_type="lobectomy",
        )
        log = agent.get_decision_log()
        assert isinstance(log, list)
