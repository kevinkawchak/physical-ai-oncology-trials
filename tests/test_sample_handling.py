"""Tests for examples-new/06_robotic_sample_handling.py

Covers Specimen data model, BarcodeVerifier, ColdChainMonitor,
SpecimenRobotController, and BatchProcessor.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def sample_mod():
    return load_module(
        "robotic_sample_handling",
        "examples-new/06_robotic_sample_handling.py",
    )


def _make_specimen(sample_mod):
    """Create a standard test specimen."""
    return sample_mod.Specimen(
        specimen_id="SPE-001",
        barcode="BC123456",
        trial_id="NCT-2026-0001",
        patient_id="PT-001",
        specimen_type=sample_mod.SpecimenType.TISSUE_BIOPSY,
        container_type=sample_mod.ContainerType.CRYOVIAL_2ML,
        collection_datetime="2026-03-01T09:00:00",
        storage_condition=sample_mod.StorageCondition.FROZEN_MINUS80,
        volume_ml=1.5,
    )


# ── Enum values ───────────────────────────────────────────────────────────


class TestEnums:
    def test_specimen_types(self, sample_mod):
        assert sample_mod.SpecimenType.TISSUE_BIOPSY.value == "tissue_biopsy"
        assert sample_mod.SpecimenType.FROZEN_TISSUE.value == "frozen_tissue"

    def test_container_types(self, sample_mod):
        assert sample_mod.ContainerType.CRYOVIAL_2ML.value == "cryovial_2ml"

    def test_storage_conditions(self, sample_mod):
        assert sample_mod.StorageCondition.FROZEN_MINUS80.value == "frozen_minus80"


# ── Specimen data model ───────────────────────────────────────────────────


class TestSpecimen:
    def test_specimen_creation(self, sample_mod):
        s = _make_specimen(sample_mod)
        assert s.specimen_id == "SPE-001"
        assert s.barcode == "BC123456"
        assert s.volume_ml == 1.5

    def test_record_event_adds_to_trail(self, sample_mod):
        s = _make_specimen(sample_mod)
        s.record_event("pickup", "Robot picked up specimen", "ROBOT-ARM-1")
        assert len(s.audit_trail) == 1
        assert s.audit_trail[0]["event_type"] == "pickup"

    def test_audit_trail_has_hash(self, sample_mod):
        s = _make_specimen(sample_mod)
        s.record_event("scan", "Barcode verified", "SCANNER-1")
        assert "integrity_hash" in s.audit_trail[0]
        assert len(s.audit_trail[0]["integrity_hash"]) == 16

    def test_compute_hash_deterministic(self, sample_mod):
        s = _make_specimen(sample_mod)
        h1 = s._compute_hash()
        h2 = s._compute_hash()
        assert h1 == h2


# ── BarcodeVerifier ───────────────────────────────────────────────────────


class TestBarcodeVerifier:
    def test_scan_and_verify_matching(self, sample_mod):
        verifier = sample_mod.BarcodeVerifier()
        result = verifier.scan_and_verify("BC123456", "station_1")
        assert result["verified"] is True
        assert result["scanned"] == "BC123456"

    def test_scan_log_populated(self, sample_mod):
        verifier = sample_mod.BarcodeVerifier()
        verifier.scan_and_verify("BC123456", "station_1")
        assert len(verifier._scan_log) >= 1


# ── ColdChainMonitor ──────────────────────────────────────────────────────


class TestColdChainMonitor:
    def test_in_range_ok(self, sample_mod):
        monitor = sample_mod.ColdChainMonitor()
        specimen = _make_specimen(sample_mod)
        result = monitor.check_temperature(specimen, current_temp_c=-80.0)
        assert result["in_range"] is True

    def test_out_of_range_alert(self, sample_mod):
        monitor = sample_mod.ColdChainMonitor()
        specimen = _make_specimen(sample_mod)
        result = monitor.check_temperature(specimen, current_temp_c=-20.0)
        assert result["in_range"] is False

    def test_refrigerated_range(self, sample_mod):
        monitor = sample_mod.ColdChainMonitor()
        specimen = _make_specimen(sample_mod)
        specimen.storage_condition = sample_mod.StorageCondition.REFRIGERATED
        result = monitor.check_temperature(specimen, current_temp_c=4.0)
        assert result["in_range"] is True

    def test_excursion_logged(self, sample_mod):
        monitor = sample_mod.ColdChainMonitor()
        specimen = _make_specimen(sample_mod)
        monitor.check_temperature(specimen, current_temp_c=-20.0)
        assert len(monitor._excursion_log) == 1

    def test_temperature_log_updated(self, sample_mod):
        monitor = sample_mod.ColdChainMonitor()
        specimen = _make_specimen(sample_mod)
        monitor.check_temperature(specimen, current_temp_c=-80.0)
        assert len(specimen.temperature_log) == 1


# ── SpecimenRobotController ───────────────────────────────────────────────


class TestSpecimenRobotController:
    def test_controller_creation(self, sample_mod):
        controller = sample_mod.SpecimenRobotController()
        assert controller._is_holding_specimen is None

    def test_pick_specimen(self, sample_mod):
        controller = sample_mod.SpecimenRobotController()
        specimen = _make_specimen(sample_mod)
        result = controller.pick_specimen(specimen, "RACK-A", (0, 0))
        assert result["success"] is True
        assert result["specimen_id"] == "SPE-001"

    def test_pick_and_place(self, sample_mod):
        controller = sample_mod.SpecimenRobotController()
        specimen = _make_specimen(sample_mod)
        pick_result = controller.pick_specimen(specimen, "RACK-A", (0, 0))
        assert pick_result["success"] is True
        place_result = controller.place_specimen(specimen, "RACK-B", (1, 1))
        assert place_result["success"] is True


# ── BatchProcessor ────────────────────────────────────────────────────────


class TestBatchProcessor:
    def test_batch_creation(self, sample_mod):
        robot = sample_mod.SpecimenRobotController()
        processor = sample_mod.BatchProcessor(robot=robot)
        assert processor.robot is robot
