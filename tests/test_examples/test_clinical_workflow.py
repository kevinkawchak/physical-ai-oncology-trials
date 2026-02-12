"""Tests for examples/04_agentic_clinical_workflow.py

Covers TrialPhase, PatientStatus, AgentRole, and clinical trial models.
"""

from __future__ import annotations


class TestEnums:
    def test_trial_phases(self, clinical_workflow_mod):
        tp = clinical_workflow_mod.TrialPhase
        assert tp.PHASE_I.value == "Phase I"
        assert tp.PHASE_II.value == "Phase II"

    def test_patient_status(self, clinical_workflow_mod):
        ps = clinical_workflow_mod.PatientStatus
        assert ps.ENROLLED.value == "enrolled"
        assert ps.WITHDRAWN.value == "withdrawn"

    def test_agent_roles(self, clinical_workflow_mod):
        ar = clinical_workflow_mod.AgentRole
        assert ar.ELIGIBILITY.value == "eligibility_specialist"
        assert ar.SAFETY_OFFICER.value == "safety_officer"


class TestClinicalTrial:
    def test_creation(self, clinical_workflow_mod):
        trial = clinical_workflow_mod.ClinicalTrial(
            trial_id="NCT-TEST-001",
            title="Test Trial",
            phase=clinical_workflow_mod.TrialPhase.PHASE_III,
            sponsor="Test Sponsor",
        )
        assert trial.trial_id == "NCT-TEST-001"


class TestPatient:
    def test_creation(self, clinical_workflow_mod):
        patient = clinical_workflow_mod.Patient(
            patient_id="PT-001",
            site_id="Site-A",
            status=clinical_workflow_mod.PatientStatus.ENROLLED,
        )
        assert patient.patient_id == "PT-001"
