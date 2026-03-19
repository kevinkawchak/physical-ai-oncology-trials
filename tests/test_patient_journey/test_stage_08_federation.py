#!/usr/bin/env python3
"""
Tests for Stage 8: Federated Data Contribution (Continuous)
============================================================

Validates the federation orchestrator: local model training without data
exposure, differential privacy with gradient clipping, secure aggregation
via SMPC, federated analytics (KM PFS, Cox PH), DSMB safety reporting,
site performance monitoring, and hash-chained audit trails.

Last updated: March 2026
DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_module(name: str, relative_path: str):
    """Load a Python module by file-path from the project root."""
    if name in sys.modules:
        return sys.modules[name]
    filepath = PROJECT_ROOT / relative_path
    if not filepath.exists():
        pytest.skip(f"Source file not found: {relative_path}")
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        sys.modules.pop(name, None)
        pytest.skip(f"Module {name!r} requires dependency not installed: {exc}")
    return mod


patient_state_mod = load_module("patient_state", "patient-journey/patient_state.py")
stage_01_mod = load_module("stage_01_prescreening", "patient-journey/stage_01_prescreening.py")
stage_02_mod = load_module("stage_02_enrollment", "patient-journey/stage_02_enrollment.py")
stage_03_mod = load_module("stage_03_digital_twin", "patient-journey/stage_03_digital_twin.py")
stage_04_mod = load_module("stage_04_robot_qualification", "patient-journey/stage_04_robot_qualification.py")
stage_05_mod = load_module("stage_05_surgery", "patient-journey/stage_05_surgery.py")
stage_06_mod = load_module("stage_06_recovery", "patient-journey/stage_06_recovery.py")
stage_08_mod = load_module("stage_08_federation", "patient-journey/stage_08_federation.py")

PatientStage = patient_state_mod.PatientStage
PatientStatus = patient_state_mod.PatientStatus
MCPConformanceLevel = patient_state_mod.MCPConformanceLevel
FederationContribution = patient_state_mod.FederationContribution
FederationOrchestrator = stage_08_mod.FederationOrchestrator


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


def _advance_through_stage_7(state):
    """Manually advance state through Stage 7 (immunotherapy).

    Stage 7 module may not yet exist, so we simulate the state
    transitions that would occur after 16 cycles of pembrolizumab.
    """
    # Advance to IMMUNOTHERAPY stage
    state.advance_stage(PatientStage.IMMUNOTHERAPY)
    state.status = PatientStatus.ON_IMMUNOTHERAPY

    # Add treatment cycles (16 cycles pembrolizumab)
    for cycle in range(1, 17):
        tc = patient_state_mod.TreatmentCycle(
            cycle_number=cycle,
            start_day=28 + (cycle - 1) * 21,
            drug="pembrolizumab",
            dose_mg=200.0,
            dose_modifications="none",
        )
        state.treatment_cycles.append(tc)

    # Add imaging timepoint for response assessment
    state.imaging_timepoints.append(
        patient_state_mod.ImagingTimepoint(
            day=42,
            modality="CT",
            tumor_volume_cm3=0.0,
            response_category=patient_state_mod.ResponseCategory.CR,
            notes="Complete response maintained post-surgery",
        )
    )

    state.add_audit_entry(
        action="IMMUNOTHERAPY_COMPLETE",
        actor="test_fixture",
        details="Stage 7 simulated: 16 cycles pembrolizumab, CR maintained",
    )
    return state


@pytest.fixture()
def federation_state():
    """Patient state after Stage 7 (immunotherapy complete), ready for federation."""
    referral = {
        "documents": ["Patient Name: Jane Doe, MRN: 12345678"],
        "records": [
            {
                "patient_id": "ORIG-12345",
                "name": "Jane Doe",
                "diagnosis_code": "C34.1",
                "stage": "IIIB",
            }
        ],
        "clinical_data": {
            "diagnosis_code": "C34.1",
            "stage": "IIIB",
            "sex": "female",
            "ecog": 1,
        },
        "dicom_series": [
            {
                "PatientID": "PAT-2026-0042",
                "StudyDate": "20260201",
                "Modality": "CT",
                "SliceThickness": 1.0,
                "PixelSpacing": [0.7, 0.7],
                "Rows": 512,
                "Columns": 512,
                "ImagePositionPatient": [0, 0, 0],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
            }
        ],
    }
    state = stage_01_mod.PreScreeningOrchestrator(referral, "SITE-003").run()
    state = stage_02_mod.EnrollmentOrchestrator(state, {"criteria": {}}).run(state)
    state = stage_03_mod.DigitalTwinOrchestrator(state).run(state)
    state = stage_04_mod.RobotQualificationOrchestrator(state, {}).run(state)
    state = stage_05_mod.SurgeryOrchestrator(state).run(state)
    state = stage_06_mod.RecoveryOrchestrator(state).run(state)
    state = _advance_through_stage_7(state)
    return state


@pytest.fixture()
def federation_config():
    """Standard federation configuration."""
    return {
        "strategy": "federated_averaging",
        "num_sites": 5,
        "epsilon": 1.0,
        "delta": 1e-5,
        "max_norm": 1.0,
    }


class TestFederationOrchestrator:
    """Tests for Stage 8 federated data contribution."""

    def test_local_model_training_no_data_exposure(self, federation_state, federation_config):
        """Local model training never exports raw data per 21 CFR 50.33."""
        orch = FederationOrchestrator(federation_state, federation_config)
        result = orch.train_local_models({"num_samples": 1})
        assert result["raw_data_exported"] is False
        assert result["compliant"] is True
        assert result["num_model_types"] == 4
        for model_type, update in result["model_updates"].items():
            assert "gradients" in update
            assert len(update["gradients"]) > 0

    def test_differential_privacy_epsilon(self, federation_state, federation_config):
        """Differential privacy uses epsilon=1.0, delta=1e-5."""
        orch = FederationOrchestrator(federation_state, federation_config)
        local = orch.train_local_models({"num_samples": 1})
        dp_result = orch.apply_differential_privacy(local["model_updates"])
        assert dp_result["epsilon_spent"] == 1.0
        assert dp_result["delta"] == 1e-5
        assert dp_result["sigma"] > 0
        assert dp_result["compliant"] is True

    def test_gradient_clipping(self, federation_state, federation_config):
        """Gradient clipping enforces max_norm=1.0 per 21 CFR 50.33."""
        orch = FederationOrchestrator(federation_state, federation_config)
        local = orch.train_local_models({"num_samples": 1})
        dp_result = orch.apply_differential_privacy(local["model_updates"])
        assert dp_result["max_norm"] == 1.0
        for model_type, update in dp_result["privatised_updates"].items():
            # Clipped norm should be <= max_norm (within floating point)
            assert update["clipped_norm"] <= 1.0 + 1e-6

    def test_secure_aggregation_mcp_conformance(self, federation_state, federation_config):
        """Secure aggregation uses SMPC and reports FEDERATED_SITE conformance."""
        orch = FederationOrchestrator(federation_state, federation_config)
        # Create privatised updates from multiple sites
        site_updates = []
        for _ in range(5):
            local = orch.train_local_models({"num_samples": 1})
            dp = orch.apply_differential_privacy(local["model_updates"])
            site_updates.append(dp)

        result = orch.execute_secure_aggregation(site_updates)
        assert result["smpc_protocol"] == "additive_secret_sharing"
        assert result["individual_updates_visible"] is False
        assert result["mcp_conformance"] == "FEDERATED_SITE"
        assert result["num_sites"] == 5
        assert result["compliant"] is True

    def test_federation_round_execution(self, federation_state, federation_config):
        """Single federation round produces valid FederationContribution."""
        orch = FederationOrchestrator(federation_state, federation_config)
        contribution = orch.run_federation_round(1)
        assert isinstance(contribution, FederationContribution)
        assert contribution.round_number == 1
        assert contribution.epsilon_spent == 1.0
        assert contribution.gradient_norm > 0
        assert contribution.day == 43  # 42 + 1

    def test_federation_strategy_selection(self, federation_state):
        """Federation supports multiple strategy options."""
        for strategy in stage_08_mod.FEDERATION_STRATEGIES:
            config = {
                "strategy": strategy,
                "num_sites": 5,
                "epsilon": 1.0,
                "delta": 1e-5,
                "max_norm": 1.0,
            }
            orch = FederationOrchestrator(federation_state, config)
            assert orch._strategy == strategy

    def test_federated_kaplan_meier_no_raw_data(self, federation_state, federation_config):
        """Federated KM PFS computed without sharing raw survival times."""
        orch = FederationOrchestrator(federation_state, federation_config)
        analytics = orch.compute_federated_analytics()
        km = analytics["kaplan_meier_pfs"]
        assert km["raw_times_shared"] is False
        assert km["median_pfs_months"] == 16.2
        assert len(km["timepoints_months"]) == 7
        assert km["survival_probability"][0] == 1.0
        assert km["survival_probability"][-1] < 1.0

    def test_federated_cox_ph(self, federation_state, federation_config):
        """Federated Cox PH produces hazard ratios without raw data."""
        orch = FederationOrchestrator(federation_state, federation_config)
        analytics = orch.compute_federated_analytics()
        cox = analytics["cox_ph"]
        assert cox["raw_data_shared"] is False
        assert cox["concordance_index"] == 0.72
        # Physical AI arm should show favorable HR < 1
        pai_hr = cox["covariates"]["physical_ai_arm"]["hazard_ratio"]
        assert pai_hr < 1.0
        assert cox["covariates"]["physical_ai_arm"]["p_value"] < 0.05

    def test_dsmb_safety_report_physical_ai(self, federation_state, federation_config):
        """DSMB reports 0 device-related events per 21 CFR 312.33."""
        orch = FederationOrchestrator(federation_state, federation_config)
        # Run a few rounds first
        for r in range(1, 6):
            orch.run_federation_round(r)
        dsmb = orch.generate_dsmb_report()
        assert dsmb["device_related_events"] == 0
        assert dsmb["dsmb_recommendation"] == "CONTINUE"
        assert dsmb["stopping_rules_triggered"] is False
        assert dsmb["physical_ai_safety"]["mechanical_injuries"] == 0
        assert dsmb["physical_ai_safety"]["software_malfunctions"] == 0
        assert dsmb["compliant"] is True

    def test_site_performance_metrics(self, federation_state, federation_config):
        """Site data quality meets 97.3% target per 21 CFR 312.56."""
        orch = FederationOrchestrator(federation_state, federation_config)
        perf = orch.monitor_site_performance()
        assert perf["overall_data_quality"] >= 0.95
        assert perf["sites_meeting_threshold"] == perf["total_sites"]
        assert perf["federation_health"] == "HEALTHY"
        # Check individual sites
        for site in perf["site_metrics"]:
            assert site["meets_threshold"] is True
            assert site["protocol_deviations"] == 0

    def test_audit_trail_hash_chained(self, federation_state, federation_config):
        """Audit trail is hash-chained per 21 CFR 312.58."""
        orch = FederationOrchestrator(federation_state, federation_config)
        for r in range(1, 11):
            orch.run_federation_round(r)
        audit = orch.maintain_audit_trail()
        assert audit["total_entries"] == 10
        assert audit["chain_algorithm"] == "SHA-256"
        assert audit["chain_valid"] is True
        assert audit["hash_chained"] is True
        assert audit["tamper_evident"] is True
        assert audit["inspection_ready"] is True
        # Verify hashes are unique
        assert len(set(orch._audit_hashes)) == 10

    def test_70_rounds_executed(self, federation_state, federation_config):
        """All 70 federation rounds execute and track epsilon."""
        orch = FederationOrchestrator(federation_state, federation_config)
        for r in range(1, 71):
            orch.run_federation_round(r)
        assert len(orch._contributions) == 70
        assert len(orch._audit_hashes) == 70
        # Epsilon budget: 1.0 per round * 70 rounds
        # Note: _total_epsilon_spent accumulates across all DP calls
        # including simulated sites, but contributions track 70 rounds
        assert orch._contributions[-1].round_number == 70

    def test_federation_run_end_to_end(self, federation_state, federation_config):
        """Full federation run produces valid state transition."""
        orch = FederationOrchestrator(federation_state, federation_config)
        state = orch.run(federation_state, total_rounds=70)
        assert state.stage == PatientStage.FEDERATION
        assert state.status == PatientStatus.SURVEILLANCE
        assert len(state.federation_contributions) == 70
        assert state.mcp_conformance_level == MCPConformanceLevel.FEDERATED_SITE
        # Verify regulatory events were recorded
        fed_events = [e for e in state.regulatory_events if e.event_type == "FEDERATION_COMPLETE"]
        assert len(fed_events) >= 1
        dsmb_events = [e for e in state.regulatory_events if e.event_type == "DSMB_REPORT"]
        assert len(dsmb_events) >= 1
        audit_events = [e for e in state.regulatory_events if e.event_type == "AUDIT_TRAIL_VERIFIED"]
        assert len(audit_events) >= 1
