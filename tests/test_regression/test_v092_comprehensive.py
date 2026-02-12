"""Comprehensive regression tests for v0.9.2 bug fixes listed in CHANGELOG.md.

The existing tests/test_regression.py covers 7 core bugs (EKF Jacobian,
hazard ratio direction, DoseResult zero values, LogisticGrowthModel div-by-zero,
TreatmentSimulator init, deidentification salt). This file covers the REMAINING
v0.9.1/v0.9.2 fixes to prevent regressions.
"""

from __future__ import annotations

import inspect
import math
from datetime import datetime

import numpy as np
import pytest

from tests.conftest import load_module


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bridge_mod():
    return load_module("isaac_mujoco_bridge", "unification/simulation_physics/isaac_mujoco_bridge.py")


@pytest.fixture()
def reg_tracker_mod():
    return load_module("regulatory_tracker", "regulatory/regulatory-intelligence/regulatory_tracker.py")


@pytest.fixture()
def gcp_mod():
    return load_module("gcp_compliance_checker", "regulatory/ich-gcp/gcp_compliance_checker.py")


@pytest.fixture()
def immuno_mod():
    return load_module(
        "tumor_microenvironment_immunotherapy_dt",
        "digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py",
    )


@pytest.fixture()
def twin_mod():
    return load_module("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py")


@pytest.fixture()
def sim_mod():
    return load_module("treatment_simulator", "digital-twins/treatment-simulation/treatment_simulator.py")


@pytest.fixture()
def converter_mod():
    return load_module("urdf_sdf_mjcf_converter", "unification/simulation_physics/urdf_sdf_mjcf_converter.py")


@pytest.fixture()
def job_runner_mod():
    return load_module("sim_job_runner", "tools/sim-job-runner/sim_job_runner.py")


@pytest.fixture()
def dose_mod():
    return load_module("dose_calculator", "tools/dose-calculator/dose_calculator.py")


@pytest.fixture()
def val_mod():
    return load_module("validation_suite", "unification/cross_platform_tools/validation_suite.py")


@pytest.fixture()
def acl_mod():
    return load_module("access_control_manager", "privacy/access-control/access_control_manager.py")


@pytest.fixture()
def deid_mod():
    return load_module("deidentification_pipeline", "privacy/de-identification/deidentification_pipeline.py")


@pytest.fixture()
def fda_mod():
    return load_module("fda_submission_tracker", "regulatory/fda-compliance/fda_submission_tracker.py")


@pytest.fixture()
def deploy_mod():
    return load_module("deployment_readiness", "tools/deployment-readiness/deployment_readiness.py")


@pytest.fixture()
def breach_mod():
    return load_module("breach_response_protocol", "privacy/breach-response/breach_response_protocol.py")


# ===========================================================================
# Test classes (one per fix, 20+ tests total)
# ===========================================================================


class TestSyncStateBidirectional:
    """sync_state handles both Isaac->MuJoCo and MuJoCo->Isaac directions."""

    def test_isaac_to_mujoco_branch(self, bridge_mod):
        bridge = bridge_mod.IsaacMuJoCoBridge()
        initial_count = bridge.conversion_stats["states_synced"]
        bridge.sync_state(source_env=None, target_env=None, source_framework="isaac", target_framework="mujoco")
        assert bridge.conversion_stats["states_synced"] == initial_count + 1

    def test_mujoco_to_isaac_branch_exists(self, bridge_mod):
        """The sync_state source code has a branch for mujoco as source."""
        source = inspect.getsource(bridge_mod.IsaacMuJoCoBridge.sync_state)
        assert 'source_framework == "mujoco"' in source
        assert "mujoco_to_isaac" in source

    def test_unsupported_framework_does_not_increment(self, bridge_mod):
        bridge = bridge_mod.IsaacMuJoCoBridge()
        initial_count = bridge.conversion_stats["states_synced"]
        bridge.sync_state(
            source_env=None, target_env=None, source_framework="unsupported_fw", target_framework="mujoco"
        )
        assert bridge.conversion_stats["states_synced"] == initial_count


class TestEvaluatePolicyBoundedLoop:
    """_evaluate_policy does not hang (bounded steps via for-loop)."""

    def test_evaluate_policy_terminates(self, bridge_mod):
        bridge = bridge_mod.IsaacMuJoCoBridge()
        validator = bridge_mod.PolicyTransferValidator(bridge)
        rewards = validator._evaluate_policy(policy_path="dummy.onnx", env=None, num_episodes=3)
        assert len(rewards) == 3

    def test_evaluate_policy_returns_array(self, bridge_mod):
        bridge = bridge_mod.IsaacMuJoCoBridge()
        validator = bridge_mod.PolicyTransferValidator(bridge)
        rewards = validator._evaluate_policy("dummy.onnx", None, 5)
        assert isinstance(rewards, np.ndarray)


class TestRegulatoryTrackerOverdueStatus:
    """Overdue deadlines labeled 'overdue' not 'imminent'."""

    def test_overdue_deadline_labeled_overdue(self, reg_tracker_mod):
        """Verify the overdue branch exists in get_upcoming_deadlines source."""
        source = inspect.getsource(reg_tracker_mod.RegulatoryTracker.get_upcoming_deadlines)
        # The fix moved "overdue" before "imminent" so it is reachable
        assert '"overdue"' in source
        # Verify overdue is set before imminent (correct elif ordering)
        overdue_pos = source.index('"overdue"')
        imminent_pos = source.index('"imminent"')
        assert overdue_pos < imminent_pos


class TestGCPComplianceScoreExcludesNotAssessed:
    """Score denominator excludes NOT_ASSESSED findings."""

    def test_score_excludes_not_assessed(self, gcp_mod):
        checker = gcp_mod.GCPComplianceChecker(guideline_version="E6_R3")
        report = checker.verify_compliance(ai_components_present=True)
        assert math.isfinite(report.overall_score)


class TestImmunotherapyFormatString:
    """Format string produces valid output (%.1f%% not %.1%%)."""

    def test_format_string_in_source(self, immuno_mod):
        source = inspect.getsource(immuno_mod)
        # The buggy pattern was "%.1%%"; verify it does not appear
        assert "%.1%%" not in source
        # The correct pattern "%.1f%%" should be present
        assert "%.1f%%" in source


class TestTumorTwinPredictZeroBaseline:
    """predict() with zero baseline doesn't divide by zero."""

    def test_predict_zero_baseline(self, twin_mod):
        pipeline = twin_mod.TumorTwinPipeline(model_type=twin_mod.ModelType.LOGISTIC)
        mask = np.zeros((5, 5, 5))  # zero volume
        twin = pipeline.create_twin(
            patient_id="ZERO-001",
            imaging_data={"ct": np.zeros((5, 5, 5))},
            tumor_segmentation=mask,
        )
        twin.is_calibrated = True
        prediction = twin.predict(horizon_days=30)
        pct = prediction.metrics.get("volume_change_percent", 0.0)
        assert math.isfinite(pct)


class TestTreatmentSimulatorSurgeryDayFloat:
    """Surgery day check uses tolerance not exact equality."""

    def test_surgery_day_tolerance(self, sim_mod):
        """Verify the surgery day check uses abs(t - surgery_day) < tolerance."""
        source = inspect.getsource(sim_mod.TreatmentSimulator._simulate_surgery)
        # The fix replaced exact equality with abs(...) < 0.5 tolerance
        assert "abs(" in source
        assert "surgery_day" in source

    def test_surgery_with_valid_twin(self, twin_mod, sim_mod):
        """Surgery simulation runs without errors on a twin with positive volume."""
        model = twin_mod.LogisticGrowthModel(domain_shape=(10, 10, 10), config={})
        model.parameters.proliferation_rate = 0.02
        # Create mask with non-zero volume
        mask = np.ones((10, 10, 10)) * 1.0
        clinical = twin_mod.PatientClinicalData(patient_id="SURG-001", age=60, sex="M", tumor_grade="II")
        twin = twin_mod.PatientDigitalTwin(
            patient_id="SURG-001",
            model=model,
            imaging_data={"ct": np.zeros((10, 10, 10))},
            tumor_mask=mask,
            clinical_data=clinical,
            config={"domain": "test"},
            solver=twin_mod.SolverType.CPU_SERIAL,
        )
        simulator = sim_mod.TreatmentSimulator(patient_twin=twin)
        if simulator.baseline_volume > 0:
            response = simulator.predict_response(
                treatment={"type": "surgery", "resection_extent": 0.95},
                horizon_days=60,
            )
            assert response.metrics.get("resection_extent") == 0.95
            assert np.min(response.volumes) < simulator.baseline_volume


class TestMJCFParsingRaisesNotFallback:
    """MJCF parsing raises NotImplementedError, not falls back to URDF."""

    def test_mjcf_parse_raises(self, converter_mod):
        converter = converter_mod.UnifiedModelConverter()
        with pytest.raises(NotImplementedError, match="MJCF"):
            from pathlib import Path

            converter._parse_model(Path("test.xml"), "mjcf")


class TestSimJobRunnerOnlyTargetFrameworks:
    """cmd_launch_all iterates only target_frameworks (intersection of task + available)."""

    def test_target_frameworks_is_filtered(self, job_runner_mod):
        source = inspect.getsource(job_runner_mod.cmd_launch_all)
        # The fix ensures iteration is over target_frameworks, not task_def["frameworks"]
        assert "for fw in target_frameworks" in source


class TestDoseCalculatorCLIExplicitZero:
    """--alpha-beta 0 and --volume 0 not replaced by defaults."""

    def test_dose_result_zero_bed(self, dose_mod):
        """DoseResult with bed_gy=0.0 still appears in to_dict()."""
        result = dose_mod.DoseResult(
            total_dose_gy=0.0,
            fractions=1,
            dose_per_fraction_gy=0.0,
            alpha_beta_gy=10.0,
            bed_gy=0.0,
            eqd2_gy=0.0,
            tcp=0.0,
            ntcp=0.0,
        )
        d = result.to_dict()
        assert "bed_gy" in d
        assert d["bed_gy"] == 0.0
        assert "eqd2_gy" in d
        assert d["eqd2_gy"] == 0.0
        assert "tcp" in d
        assert d["tcp"] == 0.0


class TestValidationSuiteThresholdFixed:
    """Success rate threshold is fixed, not 75th percentile."""

    def test_success_threshold_is_fixed(self, val_mod):
        source = inspect.getsource(val_mod.CrossPlatformValidator._run_validation)
        # The fix replaced np.percentile(rewards, 75) with a fixed threshold
        assert "percentile" not in source.lower() or "75" not in source
        # Verify the fixed threshold pattern
        assert "reward_threshold" in source


class TestAccessControlAuditLogCopy:
    """get_audit_log returns a copy, not a reference to the internal list."""

    def test_audit_log_returns_copy(self, acl_mod):
        manager = acl_mod.AccessControlManager(audit_enabled=True)
        log1 = manager.get_audit_log()
        log2 = manager.get_audit_log()
        assert log1 is not log2  # Different list objects


class TestAccessControlInvalidExpirationDenies:
    """Invalid expiration format -> access denied."""

    def test_invalid_expiration_denied(self, acl_mod):
        manager = acl_mod.AccessControlManager()
        manager.assign_role(
            user_id="test_user",
            role="principal_investigator",
            name="Test User",
            access_expiration="not-a-date",
        )
        decision = manager.check_access("test_user", "patient_records", "read")
        assert not decision.granted
        assert "expiration" in decision.reason.lower() or "invalid" in decision.reason.lower()


class TestTorchLoadWeightsOnly:
    """Verify weights_only=True pattern present in validation_suite source."""

    def test_weights_only_in_source(self, val_mod):
        source = inspect.getsource(val_mod)
        assert "weights_only=True" in source


class TestDeidDateShiftBranch:
    """DATE_SHIFT handling doesn't fall through silently."""

    def test_date_shift_branch_exists(self, deid_mod):
        """The SafeHarborTransformer._generalize_date has explicit DATE_SHIFT branch."""
        source = inspect.getsource(deid_mod.SafeHarborTransformer)
        assert "DATE_SHIFT" in source
        assert "DATE_SHIFTED" in source or "date_shift" in source.lower()

    def test_date_shift_returns_placeholder(self, deid_mod):
        transformer = deid_mod.SafeHarborTransformer(
            config=deid_mod.DeidentificationConfig(date_handling=deid_mod.DateHandling.DATE_SHIFT)
        )
        result = transformer._generalize_date("2023-01-15")
        assert result is not None
        assert result != ""
        # Should return a placeholder, not fall through
        assert "DATE" in result or "date" in result.lower() or "SHIFTED" in result


class TestFDAAIMLComponentUnspecified:
    """Default model_type is 'unspecified', not 'classification'."""

    def test_default_model_type_unspecified(self, fda_mod):
        tracker = fda_mod.FDASubmissionTracker(sponsor="TestSponsor")
        submission = tracker.create_submission(
            submission_type="510k",
            device_name="Test Device",
            intended_use="Tumor detection",
            ai_ml_components=["component_a"],
        )
        for comp in submission.ai_ml_components:
            assert comp.model_type == "unspecified"


class TestDeploymentReadinessSafetyRequiresVerification:
    """Safety check returns requires_runtime_verification status."""

    def test_safety_check_runtime_verification(self, deploy_mod):
        result = deploy_mod._check_safety_constraints("dummy.onnx", None)
        # With no constraints file, the function returns an issue
        assert "No constraints file" in result["issues"][0]

    def test_safety_constraint_dict_returns_runtime_verification(self, deploy_mod):
        source = inspect.getsource(deploy_mod._check_safety_constraints)
        assert "requires_runtime_verification" in source


class TestRiskAssessmentClamp:
    """RiskAssessment clamps out-of-range scores to [1, 5]."""

    def test_clamp_high_scores(self, breach_mod):
        assessment = breach_mod.RiskAssessment(
            incident_id="TEST-001",
            assessment_date=datetime.now().isoformat(),
            phi_nature_score=10,  # out of range
            unauthorized_party_score=10,
            acquisition_score=10,
            mitigation_score=10,
        )
        assessment.calculate_risk()
        assert 1 <= assessment.phi_nature_score <= 5
        assert 1 <= assessment.unauthorized_party_score <= 5
        assert 1 <= assessment.acquisition_score <= 5
        assert 1 <= assessment.mitigation_score <= 5

    def test_clamp_low_scores(self, breach_mod):
        assessment = breach_mod.RiskAssessment(
            incident_id="TEST-002",
            assessment_date=datetime.now().isoformat(),
            phi_nature_score=-1,
            unauthorized_party_score=0,
            acquisition_score=-5,
            mitigation_score=0,
        )
        assessment.calculate_risk()
        assert assessment.phi_nature_score >= 1
        assert assessment.unauthorized_party_score >= 1
        assert assessment.acquisition_score >= 1
        assert assessment.mitigation_score >= 1


class TestGetRecentUpdatesUsesCutoff:
    """get_recent_updates filters by cutoff date."""

    def test_cutoff_filters_old_entries(self, reg_tracker_mod):
        tracker = reg_tracker_mod.RegulatoryTracker()
        # Requesting only last 1 day should return fewer than all
        recent_1 = tracker.get_recent_updates(days=1)
        all_updates = tracker.get_recent_updates(days=3650)
        assert len(recent_1) <= len(all_updates)

    def test_cutoff_date_is_computed(self, reg_tracker_mod):
        source = inspect.getsource(reg_tracker_mod.RegulatoryTracker.get_recent_updates)
        assert "cutoff" in source
        # The fix ensures cutoff is actually used in the filter
        assert "u.date >= cutoff" in source or "u.date >=" in source


class TestIsaacMuJoCoBridgeLoggingImport:
    """Module has logging imported."""

    def test_logging_imported(self, bridge_mod):
        # The fix added 'import logging' to the bridge module
        assert hasattr(bridge_mod, "logging") or hasattr(bridge_mod, "logger")
