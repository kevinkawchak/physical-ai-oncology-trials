"""Unit tests for agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py.

Tests cover StreamType, AlertSeverity, TreatmentAction enums, DataSample,
AnomalyEvent, TreatmentRecommendation dataclasses, and StreamBuffer.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py"


@pytest.fixture()
def mod():
    return load_module("adaptive_treatment_agent", MOD_PATH)


class TestEnums:
    def test_stream_type(self, mod):
        names = [e.name for e in mod.StreamType]
        for s in ("FORCE_TORQUE", "PATIENT_VITALS", "IMAGING", "ROBOT_STATE"):
            assert s in names

    def test_alert_severity_ordering(self, mod):
        severities = list(mod.AlertSeverity)
        # Enum members are defined in escalating order:
        # INFO, ADVISORY, WARNING, CRITICAL, EMERGENCY
        expected_names = ["INFO", "ADVISORY", "WARNING", "CRITICAL", "EMERGENCY"]
        assert [s.name for s in severities] == expected_names

    def test_treatment_action(self, mod):
        names = [e.name for e in mod.TreatmentAction]
        for a in ("CONTINUE", "PAUSE", "ABORT", "REDUCE_FORCE"):
            assert a in names


class TestDataSample:
    def test_creation(self, mod):
        sample = mod.DataSample(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            values=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        )
        assert sample.stream_type == mod.StreamType.FORCE_TORQUE
        assert len(sample.values) == 6

    def test_metadata_optional(self, mod):
        sample = mod.DataSample(
            stream_type=mod.StreamType.PATIENT_VITALS,
            timestamp=time.time(),
            values=np.array([120.0, 80.0, 72.0, 98.0]),
        )
        assert hasattr(sample, "metadata")


class TestAnomalyEvent:
    def test_creation(self, mod):
        event = mod.AnomalyEvent(
            stream_type=mod.StreamType.FORCE_TORQUE,
            timestamp=time.time(),
            anomaly_type="force_spike",
            severity=mod.AlertSeverity.WARNING,
            description="Force exceeded warning threshold",
            values=np.array([15.0, 0.0, 0.0]),
            confidence=0.92,
        )
        assert event.severity == mod.AlertSeverity.WARNING
        assert event.confidence > 0.9


class TestTreatmentRecommendation:
    def test_creation(self, mod):
        rec = mod.TreatmentRecommendation(
            action=mod.TreatmentAction.REDUCE_FORCE,
            severity=mod.AlertSeverity.WARNING,
            rationale="Approaching force limits near critical structure",
            supporting_evidence=["Force trend increasing", "Proximity to vessel"],
            confidence=0.85,
            requires_confirmation=True,
            timestamp=time.time(),
        )
        assert rec.action == mod.TreatmentAction.REDUCE_FORCE
        assert rec.requires_confirmation is True


class TestStreamBuffer:
    def test_add_and_retrieve(self, mod):
        buf = mod.StreamBuffer(
            stream_type=mod.StreamType.FORCE_TORQUE,
            window_seconds=10.0,
        )
        now = time.time()
        for i in range(5):
            sample = mod.DataSample(
                stream_type=mod.StreamType.FORCE_TORQUE,
                timestamp=now + i,
                values=np.array([float(i)]),
            )
            buf.add_sample(sample)
        assert buf.sample_count == 5

    def test_get_latest(self, mod):
        buf = mod.StreamBuffer(
            stream_type=mod.StreamType.PATIENT_VITALS,
            window_seconds=10.0,
        )
        now = time.time()
        for i in range(3):
            buf.add_sample(
                mod.DataSample(
                    stream_type=mod.StreamType.PATIENT_VITALS,
                    timestamp=now + i,
                    values=np.array([float(i)]),
                )
            )
        latest = buf.get_latest()
        assert latest is not None
        assert latest.values[0] == pytest.approx(2.0)

    def test_empty_buffer(self, mod):
        buf = mod.StreamBuffer(
            stream_type=mod.StreamType.IMAGING,
            window_seconds=5.0,
        )
        assert buf.sample_count == 0
        assert buf.get_latest() is None
