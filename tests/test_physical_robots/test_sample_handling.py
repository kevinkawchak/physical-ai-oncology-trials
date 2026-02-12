"""Unit tests for examples-new/06_robotic_sample_handling.py.

Tests cover SpecimenType, ContainerType, StorageCondition enums,
Specimen, BarcodeVerifier, ColdChainMonitor, SpecimenRobotController,
BatchProcessor classes.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "examples-new/06_robotic_sample_handling.py"


@pytest.fixture()
def mod():
    return load_module("sample_handling", MOD_PATH)


class TestEnums:
    def test_specimen_type(self, mod):
        names = [e.name for e in mod.SpecimenType]
        assert len(names) >= 3

    def test_container_type(self, mod):
        names = [e.name for e in mod.ContainerType]
        assert len(names) >= 3

    def test_storage_condition(self, mod):
        names = [e.name for e in mod.StorageCondition]
        assert len(names) >= 2


def _make_specimen(mod, specimen_id="SPEC-001", patient_id="PT-001", barcode="BC-001"):
    """Helper to build a Specimen with all required fields."""
    return mod.Specimen(
        specimen_id=specimen_id,
        barcode=barcode,
        trial_id="NCT-TEST-0001",
        patient_id=patient_id,
        specimen_type=list(mod.SpecimenType)[0],
        container_type=list(mod.ContainerType)[0],
        collection_datetime="2026-02-12T10:00:00",
        storage_condition=mod.StorageCondition.REFRIGERATED,
    )


class TestSpecimen:
    def test_creation(self, mod):
        spec = _make_specimen(mod, specimen_id="SPEC-001", patient_id="PT-001")
        assert spec.specimen_id == "SPEC-001"

    def test_record_event(self, mod):
        spec = _make_specimen(mod, specimen_id="SPEC-002", patient_id="PT-002")
        spec.record_event("picked", "Robot picked specimen", "robot")
        assert len(spec.audit_trail) >= 1

    def test_audit_trail_hash(self, mod):
        spec = _make_specimen(mod, specimen_id="SPEC-003", patient_id="PT-003")
        spec.record_event("created", "Specimen created", "operator")
        # Audit trail entries should have integrity hash
        entry = spec.audit_trail[-1]
        assert hasattr(entry, "integrity_hash") or "hash" in str(entry)


class TestBarcodeVerifier:
    def test_creation(self, mod):
        verifier = mod.BarcodeVerifier()
        assert verifier is not None

    def test_scan_matching_barcode(self, mod):
        verifier = mod.BarcodeVerifier()
        result = verifier.scan_and_verify(
            expected_barcode="SPEC-001",
            station_id="test_station",
        )
        assert result["verified"] is True

    def test_scan_mismatched_barcode(self, mod):
        verifier = mod.BarcodeVerifier()
        # Monkey-patch _simulate_scan to always return a wrong barcode
        verifier._simulate_scan = lambda expected, attempt: "WRONG-BARCODE"
        result = verifier.scan_and_verify(
            expected_barcode="SPEC-001",
            station_id="test_station",
        )
        assert result["verified"] is False


class TestColdChainMonitor:
    def test_in_range(self, mod):
        monitor = mod.ColdChainMonitor()
        specimen = _make_specimen(mod)
        specimen.storage_condition = mod.StorageCondition.REFRIGERATED
        result = monitor.check_temperature(specimen, 4.0)
        assert result["in_range"] is True

    def test_out_of_range(self, mod):
        monitor = mod.ColdChainMonitor()
        specimen = _make_specimen(mod)
        specimen.storage_condition = mod.StorageCondition.REFRIGERATED
        result = monitor.check_temperature(specimen, 25.0)
        assert result["in_range"] is False

    def test_temperature_log(self, mod):
        monitor = mod.ColdChainMonitor()
        specimen = _make_specimen(mod)
        specimen.storage_condition = mod.StorageCondition.REFRIGERATED
        monitor.check_temperature(specimen, 4.0)
        monitor.check_temperature(specimen, 5.0)
        assert len(specimen.temperature_log) >= 2


class TestSpecimenRobotController:
    def test_creation(self, mod):
        controller = mod.SpecimenRobotController()
        assert controller is not None

    def test_pick_specimen(self, mod):
        controller = mod.SpecimenRobotController()
        spec = _make_specimen(mod, specimen_id="SPEC-PICK", patient_id="PT-PICK", barcode="BC-PICK")
        result = controller.pick_specimen(spec, rack_id="rack_A", position=(0, 0))
        assert result is not None
