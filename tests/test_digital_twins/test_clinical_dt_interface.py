"""Tests for digital-twins/clinical-integration/clinical_dt_interface.py

Covers ClinicalConnector, FHIRClient, ComplianceManager, and data models.
"""

from __future__ import annotations


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_connection_status(self, clinical_dt_mod):
        cs = clinical_dt_mod.ConnectionStatus
        assert cs.CONNECTED.value == "connected"
        assert cs.DISCONNECTED.value == "disconnected"
        assert cs.ERROR.value == "error"

    def test_compliance_regulation(self, clinical_dt_mod):
        cr = clinical_dt_mod.ComplianceRegulation
        assert cr.FDA_21CFR11.value == "21CFR11"
        assert cr.HIPAA.value == "HIPAA"


# ── PatientRecord dataclass ──────────────────────────────────────────────


class TestPatientRecord:
    def test_creation(self, clinical_dt_mod):
        record = clinical_dt_mod.PatientRecord(mrn="MRN-001")
        assert record.mrn == "MRN-001"
        assert record.diagnoses == []
        assert record.imaging_studies == []

    def test_with_demographics(self, clinical_dt_mod):
        record = clinical_dt_mod.PatientRecord(
            mrn="MRN-002",
            name="Test Patient",
            sex="M",
        )
        assert record.name == "Test Patient"
        assert record.sex == "M"


# ── ImagingStudy dataclass ───────────────────────────────────────────────


class TestImagingStudy:
    def test_creation(self, clinical_dt_mod):
        study = clinical_dt_mod.ImagingStudy(
            study_id="STUDY-001",
            modality="CT",
            date="2026-03-01",
        )
        assert study.study_id == "STUDY-001"
        assert study.modality == "CT"


# ── ClinicalConnector ────────────────────────────────────────────────────


class TestClinicalConnector:
    def test_creation_defaults(self, clinical_dt_mod):
        conn = clinical_dt_mod.ClinicalConnector()
        assert conn.auth_method == "oauth2"

    def test_creation_with_endpoints(self, clinical_dt_mod):
        conn = clinical_dt_mod.ClinicalConnector(
            pacs_endpoint="http://pacs.local",
            fhir_endpoint="http://fhir.local",
        )
        assert conn.pacs_endpoint == "http://pacs.local"
        assert conn.fhir_endpoint == "http://fhir.local"

    def test_test_connection(self, clinical_dt_mod):
        conn = clinical_dt_mod.ClinicalConnector()
        result = conn.test_connection()
        assert isinstance(result, dict)

    def test_get_patient_returns_record(self, clinical_dt_mod):
        conn = clinical_dt_mod.ClinicalConnector()
        record = conn.get_patient("MRN-TEST-001")
        assert isinstance(record, clinical_dt_mod.PatientRecord)


# ── FHIRClient ───────────────────────────────────────────────────────────


class TestFHIRClient:
    def test_creation(self, clinical_dt_mod):
        client = clinical_dt_mod.FHIRClient(base_url="http://fhir.test")
        assert client.base_url == "http://fhir.test"

    def test_search_returns_list(self, clinical_dt_mod):
        client = clinical_dt_mod.FHIRClient(base_url="http://fhir.test")
        results = client.search("Patient", {"name": "Test"})
        assert isinstance(results, list)


# ── ComplianceManager ────────────────────────────────────────────────────


class TestComplianceManager:
    def test_creation(self, clinical_dt_mod):
        mgr = clinical_dt_mod.ComplianceManager()
        assert mgr.regulation == clinical_dt_mod.ComplianceRegulation.FDA_21CFR11

    def test_log_event(self, clinical_dt_mod):
        mgr = clinical_dt_mod.ComplianceManager()
        mgr.log_event(
            event_type="data_access",
            user="test_user",
            resource="patient_record",
            action="read",
        )
        # No crash means event was logged
