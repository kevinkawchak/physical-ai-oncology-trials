"""Tests for digital-twins/clinical-integration/clinical_dt_interface.py."""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("clinical_dt_interface", "digital-twins/clinical-integration/clinical_dt_interface.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestConnectionStatus:
    def test_connected(self, mod):
        assert mod.ConnectionStatus.CONNECTED.value == "connected"

    def test_disconnected(self, mod):
        assert mod.ConnectionStatus.DISCONNECTED.value == "disconnected"

    def test_error(self, mod):
        assert mod.ConnectionStatus.ERROR.value == "error"

    def test_authenticating(self, mod):
        assert mod.ConnectionStatus.AUTHENTICATING.value == "authenticating"

    def test_member_count(self, mod):
        assert len(mod.ConnectionStatus) == 4


class TestComplianceRegulation:
    def test_fda_21cfr11(self, mod):
        assert mod.ComplianceRegulation.FDA_21CFR11.value == "21CFR11"

    def test_hipaa(self, mod):
        assert mod.ComplianceRegulation.HIPAA.value == "HIPAA"

    def test_gdpr(self, mod):
        assert mod.ComplianceRegulation.GDPR.value == "GDPR"

    def test_iec_62304(self, mod):
        assert mod.ComplianceRegulation.IEC_62304.value == "IEC62304"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestPatientRecord:
    def test_creation_required(self, mod):
        pr = mod.PatientRecord(mrn="MRN-001")
        assert pr.mrn == "MRN-001"

    def test_defaults(self, mod):
        pr = mod.PatientRecord(mrn="MRN-002")
        assert pr.name == ""
        assert pr.dob == ""
        assert pr.sex == ""
        assert pr.diagnoses == []
        assert pr.imaging_studies == []
        assert pr.treatments == []
        assert pr.biomarkers == {}

    def test_custom_fields(self, mod):
        pr = mod.PatientRecord(mrn="MRN-003", name="Test", sex="F", diagnoses=["C34.9"])
        assert pr.sex == "F"
        assert len(pr.diagnoses) == 1


class TestImagingStudy:
    def test_creation(self, mod):
        study = mod.ImagingStudy(study_id="1.2.3", modality="CT", date="2025-12-01")
        assert study.study_id == "1.2.3"
        assert study.modality == "CT"

    def test_defaults(self, mod):
        study = mod.ImagingStudy(study_id="S1", modality="MR", date="2025-01-01")
        assert study.body_part == ""
        assert study.series == []
        assert study.storage_path == ""


class TestAuditEntry:
    def test_creation(self, mod):
        ts = datetime(2026, 1, 15, 10, 30)
        entry = mod.AuditEntry(timestamp=ts, user="admin", action="read", resource="Patient/001")
        assert entry.user == "admin"
        assert entry.action == "read"

    def test_timestamp_type(self, mod):
        ts = datetime.now()
        entry = mod.AuditEntry(timestamp=ts, user="u", action="a", resource="r")
        assert isinstance(entry.timestamp, datetime)

    def test_details_default(self, mod):
        ts = datetime.now()
        entry = mod.AuditEntry(timestamp=ts, user="u", action="a", resource="r")
        assert entry.details == {}


# ---------------------------------------------------------------------------
# ClinicalConnector tests
# ---------------------------------------------------------------------------


class TestClinicalConnector:
    def test_init_defaults(self, mod):
        cc = mod.ClinicalConnector()
        assert cc.pacs_endpoint == ""
        assert cc.fhir_endpoint == ""
        assert cc.auth_method == "oauth2"

    def test_init_custom(self, mod):
        cc = mod.ClinicalConnector(
            pacs_endpoint="https://pacs.local",
            fhir_endpoint="https://fhir.local/R4",
            auth_method="basic",
        )
        assert cc.pacs_endpoint == "https://pacs.local"
        assert cc.fhir_endpoint == "https://fhir.local/R4"

    def test_get_status_disconnected(self, mod):
        cc = mod.ClinicalConnector()
        status = cc.test_connection()
        assert status["pacs"] == mod.ConnectionStatus.DISCONNECTED
        assert status["fhir"] == mod.ConnectionStatus.DISCONNECTED

    def test_get_status_connected(self, mod):
        cc = mod.ClinicalConnector(fhir_endpoint="https://fhir.local/R4")
        status = cc.test_connection()
        assert status["fhir"] == mod.ConnectionStatus.CONNECTED

    def test_get_patient(self, mod):
        cc = mod.ClinicalConnector(fhir_endpoint="https://fhir.local/R4")
        patient = cc.get_patient("12345")
        assert patient.mrn == "12345"


# ---------------------------------------------------------------------------
# FHIRClient tests
# ---------------------------------------------------------------------------


class TestFHIRClient:
    def test_init(self, mod):
        client = mod.FHIRClient("https://fhir.local/R4/")
        # Trailing slash stripped
        assert client.base_url == "https://fhir.local/R4"

    def test_search_patient(self, mod):
        client = mod.FHIRClient("https://fhir.local/R4")
        results = client.search("Patient", {"identifier": "MRN|123"})
        assert len(results) == 1
        assert results[0]["resourceType"] == "Patient"

    def test_read_resource(self, mod):
        client = mod.FHIRClient("https://fhir.local/R4")
        resource = client.read("Patient", "patient-123")
        assert resource["resourceType"] == "Patient"
        assert resource["id"] == "patient-123"


# ---------------------------------------------------------------------------
# ComplianceManager tests
# ---------------------------------------------------------------------------


class TestComplianceManager:
    def test_init_21cfr11(self, mod):
        cm = mod.ComplianceManager("21CFR11")
        assert cm.regulation == mod.ComplianceRegulation.FDA_21CFR11

    def test_init_hipaa(self, mod):
        cm = mod.ComplianceManager("HIPAA")
        assert cm.regulation == mod.ComplianceRegulation.HIPAA

    def test_log_event(self, mod):
        cm = mod.ComplianceManager("21CFR11")
        cm.log_event("test_event", "user1", "res1", "action1")
        assert len(cm._audit_entries) == 1
        assert cm._audit_entries[0]["event_type"] == "test_event"
