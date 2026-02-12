"""Unit tests for digital-twins/clinical-integration/clinical_dt_interface.py.

Tests cover enums, dataclasses, ClinicalConnector, ComplianceManager,
and related clinical integration components.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import load_module

MOD_PATH = "digital-twins/clinical-integration/clinical_dt_interface.py"


@pytest.fixture()
def mod():
    return load_module("clinical_dt_interface", MOD_PATH)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_connection_status(self, mod):
        names = [e.name for e in mod.ConnectionStatus]
        for s in ("CONNECTED", "DISCONNECTED", "ERROR", "AUTHENTICATING"):
            assert s in names

    def test_compliance_regulation(self, mod):
        names = [e.name for e in mod.ComplianceRegulation]
        assert "FDA_21CFR11" in names
        assert "HIPAA" in names
        assert "GDPR" in names
        assert "IEC_62304" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPatientRecord:
    def test_creation(self, mod):
        rec = mod.PatientRecord(
            mrn="MRN-001",
            name="Test Patient",
            dob="1960-01-15",
            sex="F",
        )
        assert rec.mrn == "MRN-001"

    def test_optional_fields_default(self, mod):
        rec = mod.PatientRecord(
            mrn="MRN-002",
            name="Test",
            dob="1970-05-20",
            sex="M",
        )
        # Optional fields should default gracefully
        assert hasattr(rec, "diagnoses")
        assert hasattr(rec, "imaging_studies")


class TestImagingStudy:
    def test_creation(self, mod):
        study = mod.ImagingStudy(
            study_id="STUDY-001",
            modality="CT",
            date="2026-01-15",
            body_part="CHEST",
        )
        assert study.modality == "CT"
        assert study.study_id == "STUDY-001"


class TestAuditEntry:
    def test_creation(self, mod):
        entry = mod.AuditEntry(
            timestamp=datetime.now().isoformat(),
            user="test_user",
            action="READ",
            resource="patient/MRN-001",
            details="Test access",
        )
        assert entry.user == "test_user"
        assert entry.action == "READ"


# ---------------------------------------------------------------------------
# ClinicalConnector tests
# ---------------------------------------------------------------------------


class TestClinicalConnector:
    def test_instantiation(self, mod):
        connector = mod.ClinicalConnector(
            pacs_endpoint="http://localhost:8042",
            fhir_endpoint="http://localhost:8080/fhir",
        )
        assert connector is not None

    def test_default_auth_method(self, mod):
        connector = mod.ClinicalConnector(
            pacs_endpoint="http://localhost:8042",
            fhir_endpoint="http://localhost:8080/fhir",
        )
        assert hasattr(connector, "auth_method")


# ---------------------------------------------------------------------------
# ComplianceManager tests
# ---------------------------------------------------------------------------


class TestComplianceManager:
    def test_instantiation(self, mod):
        cm = mod.ComplianceManager(regulation=mod.ComplianceRegulation.FDA_21CFR11)
        assert cm is not None

    def test_log_event(self, mod):
        cm = mod.ComplianceManager(regulation=mod.ComplianceRegulation.FDA_21CFR11)
        cm.enable_audit_trail(log_path="/tmp/test_audit.log")
        cm.log_event(
            event_type="ACCESS",
            user="test_user",
            resource="patient/001",
            action="READ",
            details="Test audit entry",
        )
        # Should not raise


class TestClinicalDigitalTwinFactory:
    def test_instantiation(self, mod):
        connector = mod.ClinicalConnector(
            pacs_endpoint="http://localhost:8042",
            fhir_endpoint="http://localhost:8080/fhir",
        )
        factory = mod.ClinicalDigitalTwinFactory(connector=connector)
        assert factory is not None
