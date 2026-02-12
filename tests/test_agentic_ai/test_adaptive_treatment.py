"""Tests for agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py"""

from __future__ import annotations

import time

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "realtime_adaptive_treatment_agent",
        "agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py",
    )


# ============================================================
# Enum tests
# ============================================================


class TestStreamType:
    def test_force_torque_value(self, mod):
        assert mod.StreamType.FORCE_TORQUE.value == "force_torque"

    def test_patient_vitals_value(self, mod):
        assert mod.StreamType.PATIENT_VITALS.value == "patient_vitals"

    def test_imaging_value(self, mod):
        assert mod.StreamType.IMAGING.value == "imaging"

    def test_member_count(self, mod):
        assert len(mod.StreamType) == 6


class TestAlertSeverity:
    def test_values(self, mod):
        assert mod.AlertSeverity.INFO.value == "info"
        assert mod.AlertSeverity.EMERGENCY.value == "emergency"

    def test_ordering_by_list_index(self, mod):
        members = list(mod.AlertSeverity)
        assert members.index(mod.AlertSeverity.INFO) < members.index(mod.AlertSeverity.EMERGENCY)

    def test_member_count(self, mod):
        assert len(mod.AlertSeverity) == 5


class TestTreatmentAction:
    def test_continue_value(self, mod):
        assert mod.TreatmentAction.CONTINUE.value == "continue_as_planned"

    def test_abort_value(self, mod):
        assert mod.TreatmentAction.ABORT.value == "abort_procedure"

    def test_member_count(self, mod):
        assert len(mod.TreatmentAction) == 11


# ============================================================
# Dataclass tests
# ============================================================


class TestDataSample:
    def test_creation(self, mod):
        ts = time.time()
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=ts,
            values={"forces": [1.0, 2.0, 3.0]},
        )
        assert sample.stream_type == mod.StreamType.FORCE_TORQUE
        assert sample.timestamp == ts
        assert sample.values["forces"] == [1.0, 2.0, 3.0]

    def test_default_metadata(self, mod):
        sample = mod.DataSample(stream_type=mod.StreamType.IMAGING, timestamp=0.0)
        assert sample.metadata == {}


class TestAnomalyEvent:
    def test_creation(self, mod):
        event = mod.AnomalyEvent(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=100.0,
            anomaly_type="tachycardia",
            severity=mod.AlertSeverity.WARNING,
            description="Heart rate elevated",
            confidence=0.95,
        )
        assert event.anomaly_type == "tachycardia"
        assert event.confidence == 0.95


class TestTreatmentRecommendation:
    def test_creation(self, mod):
        rec = mod.TreatmentRecommendation(
            action=mod.TreatmentAction.PAUSE,
            severity=mod.AlertSeverity.CRITICAL,
            rationale="Hemorrhage suspected",
            confidence=0.85,
        )
        assert rec.action == mod.TreatmentAction.PAUSE
        assert rec.requires_confirmation is True

    def test_to_dict(self, mod):
        rec = mod.TreatmentRecommendation(
            action=mod.TreatmentAction.SLOW_DOWN,
            severity=mod.AlertSeverity.ADVISORY,
            rationale="Force spike detected",
        )
        d = rec.to_dict()
        assert d["action"] == "reduce_speed"
        assert d["severity"] == "advisory"


# ============================================================
# StreamBuffer tests
# ============================================================


class TestStreamBuffer:
    def test_add_and_get_window(self, mod):
        buf = mod.StreamBuffer(mod.StreamType.FORCE_TORQUE, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [1.0, 0.0, 0.0]},
        )
        buf.add_sample(sample)
        window = buf.get_window()
        assert len(window) == 1

    def test_empty_buffer(self, mod):
        buf = mod.StreamBuffer(mod.StreamType.PATIENT_VITALS)
        assert buf.get_window() == []
        assert buf.get_latest() is None
        assert buf.sample_count == 0

    def test_multiple_samples(self, mod):
        buf = mod.StreamBuffer(mod.StreamType.FORCE_TORQUE, window_seconds=60.0)
        now = time.time()
        for i in range(5):
            sample = mod.DataSample(
                stream_type=mod.StreamType.FORCE_TORQUE,
                timestamp=now + i,
                values={"forces": [float(i), 0.0, 0.0]},
            )
            buf.add_sample(sample)
        assert buf.sample_count == 5
        assert buf.get_latest().values["forces"][0] == 4.0


# ============================================================
# ForceTorqueProcessor tests
# ============================================================


class TestForceTorqueProcessor:
    def test_process_no_anomaly(self, mod):
        proc = mod.ForceTorqueProcessor()
        buf = mod.StreamBuffer(mod.StreamType.FORCE_TORQUE, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [0.5, 0.3, 0.2], "torques": [0.01, 0.01, 0.01]},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        assert anomalies == []

    def test_process_empty_buffer(self, mod):
        proc = mod.ForceTorqueProcessor()
        buf = mod.StreamBuffer(mod.StreamType.FORCE_TORQUE)
        assert proc.process(buf) == []

    def test_detect_excessive_force(self, mod):
        proc = mod.ForceTorqueProcessor()
        buf = mod.StreamBuffer(mod.StreamType.FORCE_TORQUE, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [4.0, 4.0, 4.0], "torques": [0.0, 0.0, 0.0]},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        types = [a.anomaly_type for a in anomalies]
        assert "excessive_force" in types

    def test_detect_force_spike(self, mod):
        proc = mod.ForceTorqueProcessor()
        buf = mod.StreamBuffer(mod.StreamType.FORCE_TORQUE, window_seconds=60.0)
        # First sample: low force to set baseline
        s1 = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [0.1, 0.1, 0.1], "torques": [0.0, 0.0, 0.0]},
        )
        buf.add_sample(s1)
        proc.process(buf)
        # Second sample: sudden high force
        s2 = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [3.0, 3.0, 3.0], "torques": [0.0, 0.0, 0.0]},
        )
        buf.add_sample(s2)
        anomalies = proc.process(buf)
        types = [a.anomaly_type for a in anomalies]
        assert "force_spike" in types


# ============================================================
# VitalsProcessor tests
# ============================================================


class TestVitalsProcessor:
    def test_process_normal_vitals(self, mod):
        proc = mod.VitalsProcessor()
        buf = mod.StreamBuffer(mod.StreamType.PATIENT_VITALS, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 72, "systolic_bp": 120, "spo2_percent": 98, "etco2_mmhg": 36},
        )
        buf.add_sample(sample)
        assert proc.process(buf) == []

    def test_detect_tachycardia(self, mod):
        proc = mod.VitalsProcessor()
        buf = mod.StreamBuffer(mod.StreamType.PATIENT_VITALS, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 130, "systolic_bp": 120, "spo2_percent": 98, "etco2_mmhg": 36},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        assert any(a.anomaly_type == "tachycardia" for a in anomalies)

    def test_detect_hypotension(self, mod):
        proc = mod.VitalsProcessor()
        buf = mod.StreamBuffer(mod.StreamType.PATIENT_VITALS, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 72, "systolic_bp": 75, "spo2_percent": 98, "etco2_mmhg": 36},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        assert any(a.anomaly_type == "hypotension" for a in anomalies)

    def test_detect_desaturation(self, mod):
        proc = mod.VitalsProcessor()
        buf = mod.StreamBuffer(mod.StreamType.PATIENT_VITALS, window_seconds=60.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 72, "systolic_bp": 120, "spo2_percent": 85, "etco2_mmhg": 36},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        assert any(a.anomaly_type == "desaturation" for a in anomalies)


# ============================================================
# ImagingProcessor tests
# ============================================================


class TestImagingProcessor:
    def test_process_no_anomaly(self, mod):
        proc = mod.ImagingProcessor()
        buf = mod.StreamBuffer(mod.StreamType.IMAGING, window_seconds=120.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.IMAGING,
            timestamp=time.time(),
            values={"estimated_margin_mm": 10.0},
        )
        buf.add_sample(sample)
        assert proc.process(buf) == []

    def test_detect_inadequate_margin(self, mod):
        proc = mod.ImagingProcessor()
        buf = mod.StreamBuffer(mod.StreamType.IMAGING, window_seconds=120.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.IMAGING,
            timestamp=time.time(),
            values={"estimated_margin_mm": 2.0},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        assert any(a.anomaly_type == "inadequate_margin" for a in anomalies)

    def test_detect_perfusion_change(self, mod):
        proc = mod.ImagingProcessor()
        buf = mod.StreamBuffer(mod.StreamType.IMAGING, window_seconds=120.0)
        sample = mod.DataSample(
            stream_type=mod.StreamType.IMAGING,
            timestamp=time.time(),
            values={"perfusion_change_percent": -45.0},
        )
        buf.add_sample(sample)
        anomalies = proc.process(buf)
        assert any(a.anomaly_type == "perfusion_change" for a in anomalies)


# ============================================================
# CrossModalCorrelator tests
# ============================================================


class TestCrossModalCorrelator:
    def test_add_anomalies_empty(self, mod):
        corr = mod.CrossModalCorrelator(correlation_window_seconds=60.0)
        corr.add_anomalies([])
        assert corr.detect_correlations() == []

    def test_correlate_force_spike_and_vitals(self, mod):
        corr = mod.CrossModalCorrelator(correlation_window_seconds=60.0)
        now = time.time()
        force_spike = mod.AnomalyEvent(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=now,
            anomaly_type="force_spike",
            severity=mod.AlertSeverity.WARNING,
            description="Force spike detected",
        )
        tachycardia = mod.AnomalyEvent(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=now,
            anomaly_type="tachycardia",
            severity=mod.AlertSeverity.WARNING,
            description="Tachycardia detected",
        )
        corr.add_anomalies([force_spike, tachycardia])
        correlations = corr.detect_correlations()
        assert any(c.event_type == "possible_hemorrhage" for c in correlations)

    def test_correlate_hemodynamic_instability(self, mod):
        corr = mod.CrossModalCorrelator(correlation_window_seconds=60.0)
        now = time.time()
        hypo = mod.AnomalyEvent(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=now,
            anomaly_type="hypotension",
            severity=mod.AlertSeverity.WARNING,
            description="Hypotension",
        )
        tachy = mod.AnomalyEvent(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=now,
            anomaly_type="tachycardia",
            severity=mod.AlertSeverity.WARNING,
            description="Tachycardia",
        )
        corr.add_anomalies([hypo, tachy])
        correlations = corr.detect_correlations()
        assert any(c.event_type == "hemodynamic_instability" for c in correlations)


# ============================================================
# TreatmentDecisionEngine tests
# ============================================================


class TestTreatmentDecisionEngine:
    def test_evaluate_no_input(self, mod):
        engine = mod.TreatmentDecisionEngine()
        recs = engine.evaluate([], [])
        assert recs == []

    def test_generate_recommendation_from_anomaly(self, mod):
        engine = mod.TreatmentDecisionEngine()
        anomaly = mod.AnomalyEvent(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            anomaly_type="excessive_force",
            severity=mod.AlertSeverity.WARNING,
            description="Force too high",
            confidence=0.9,
        )
        recs = engine.evaluate([anomaly], [])
        assert len(recs) >= 1
        assert recs[0].action == mod.TreatmentAction.REDUCE_FORCE

    def test_priority_ordering(self, mod):
        engine = mod.TreatmentDecisionEngine()
        anomalies = [
            mod.AnomalyEvent(
                stream_type=mod.StreamType.FORCE_TORQUE,
                timestamp=time.time(),
                anomaly_type="excessive_force",
                severity=mod.AlertSeverity.WARNING,
                description="Force high",
            ),
            mod.AnomalyEvent(
                stream_type=mod.StreamType.PATIENT_VITALS,
                timestamp=time.time(),
                anomaly_type="desaturation",
                severity=mod.AlertSeverity.CRITICAL,
                description="SpO2 low",
            ),
        ]
        recs = engine.evaluate(anomalies, [])
        # CRITICAL should come before WARNING
        if len(recs) >= 2:
            severity_order = {
                mod.AlertSeverity.EMERGENCY: 0,
                mod.AlertSeverity.CRITICAL: 1,
                mod.AlertSeverity.WARNING: 2,
                mod.AlertSeverity.ADVISORY: 3,
                mod.AlertSeverity.INFO: 4,
            }
            assert severity_order[recs[0].severity] <= severity_order[recs[1].severity]

    def test_recommendation_history(self, mod):
        engine = mod.TreatmentDecisionEngine()
        anomaly = mod.AnomalyEvent(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            anomaly_type="force_spike",
            severity=mod.AlertSeverity.WARNING,
            description="Spike",
        )
        engine.evaluate([anomaly], [])
        history = engine.get_recommendation_history()
        assert len(history) >= 1


# ============================================================
# AdaptiveTreatmentAgent tests
# ============================================================


class TestAdaptiveTreatmentAgent:
    def test_init(self, mod):
        agent = mod.AdaptiveTreatmentAgent(
            patient_id="PT-001",
            procedure_type="lobectomy",
        )
        assert agent.patient_id == "PT-001"
        assert agent.procedure_type == "lobectomy"

    def test_ingest_normal_sample(self, mod):
        agent = mod.AdaptiveTreatmentAgent(patient_id="PT-001", procedure_type="lobectomy")
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [0.5, 0.3, 0.2], "torques": [0.01, 0.01, 0.01]},
        )
        recs = agent.ingest_sample(sample)
        assert recs == []

    def test_ingest_anomalous_sample(self, mod):
        agent = mod.AdaptiveTreatmentAgent(patient_id="PT-001", procedure_type="lobectomy")
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values={"forces": [5.0, 5.0, 5.0], "torques": [0.0, 0.0, 0.0]},
        )
        recs = agent.ingest_sample(sample)
        assert len(recs) >= 1

    def test_get_status(self, mod):
        agent = mod.AdaptiveTreatmentAgent(patient_id="PT-001", procedure_type="lobectomy")
        status = agent.get_status()
        assert status["patient_id"] == "PT-001"
        assert status["samples_processed"] == 0

    def test_status_increments_after_ingest(self, mod):
        agent = mod.AdaptiveTreatmentAgent(patient_id="PT-001", procedure_type="lobectomy")
        sample = mod.DataSample(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values={"heart_rate_bpm": 72, "systolic_bp": 120, "spo2_percent": 98, "etco2_mmhg": 36},
        )
        agent.ingest_sample(sample)
        assert agent.get_status()["samples_processed"] == 1

    def test_unknown_stream_type_ignored(self, mod):
        agent = mod.AdaptiveTreatmentAgent(patient_id="PT-001", procedure_type="lobectomy")
        sample = mod.DataSample(
            stream_type=mod.StreamType.ANESTHESIA,
            timestamp=time.time(),
            values={"gas_flow": 2.0},
        )
        recs = agent.ingest_sample(sample)
        assert recs == []
