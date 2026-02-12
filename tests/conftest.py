"""Shared fixtures, helpers, and mock factories for the test suite.

All source modules live in directories with hyphens (e.g. ``examples-new/``)
which cannot be imported with a normal ``import`` statement.  The
``load_module`` helper uses ``importlib.util`` to load them by file path.

Every fixture seeds the NumPy RNG for deterministic, reproducible tests.

Fixture hierarchy:
    conftest.py  (this file — project-wide)
    └── test_digital_twins/conftest.py   (digital-twin-specific)
    └── test_agentic_ai/conftest.py      (agentic-AI-specific)
    └── test_tools/conftest.py           (CLI-tool-specific)
    └── test_physical_robots/conftest.py (robot-specific)
    └── test_integration/conftest.py     (cross-module)

LICENSE: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project root (one level up from tests/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative_path: str):
    """Load a Python module by file-path from the project root.

    Args:
        name: Module name to register in ``sys.modules``.
        relative_path: Path relative to the project root
            (e.g. ``"examples-new/01_realtime_safety_monitoring.py"``).

    Returns:
        The loaded module object.
    """
    if name in sys.modules:
        return sys.modules[name]

    filepath = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Deterministic random state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed NumPy random state for deterministic tests."""
    np.random.seed(42)
    yield


# =========================================================================
# Mock data factories
# =========================================================================


class PatientFactory:
    """Factory for synthetic patient data used across test suites."""

    @staticmethod
    def clinical_data() -> dict:
        """Minimal synthetic patient clinical data."""
        return {
            "patient_id": "TEST-PT-001",
            "age": 62,
            "sex": "M",
            "tumor_grade": "III",
            "tumor_stage": "IIIA",
            "molecular_markers": {"EGFR": "positive", "PD-L1": 80},
            "treatment_history": [],
            "comorbidities": ["hypertension"],
        }

    @staticmethod
    def tumor_volume(shape: tuple[int, ...] = (10, 10, 10)) -> np.ndarray:
        """Synthetic 3-D tumor volume with a central spherical lesion."""
        vol = np.zeros(shape, dtype=np.float64)
        center = np.array(shape) / 2.0
        for idx in np.ndindex(shape):
            if np.linalg.norm(np.array(idx) - center) < min(shape) / 4:
                vol[idx] = 1.0
        return vol

    @staticmethod
    def dose_distribution(shape: tuple[int, ...] = (10, 10, 10)) -> np.ndarray:
        """Synthetic 3-D dose distribution in Gy."""
        dose = np.zeros(shape, dtype=np.float64)
        center = np.array(shape) / 2.0
        for idx in np.ndindex(shape):
            dist = np.linalg.norm(np.array(idx) - center)
            dose[idx] = max(0, 60.0 * (1.0 - dist / (min(shape) / 2)))
        return dose

    @staticmethod
    def patient_state_8d() -> np.ndarray:
        """8-dimensional patient state: [V, g, d, ANC, Cr, Hb, W, ECOG]."""
        return np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])

    @staticmethod
    def patient_record_with_phi() -> dict:
        """Synthetic patient record with PHI for deidentification testing."""
        return {
            "patient_name": "John Doe",
            "mrn": "MRN-123456",
            "date_of_birth": "1965-03-15",
            "ssn": "123-45-6789",
            "phone": "(555) 123-4567",
            "email": "john.doe@example.com",
            "address": "123 Main St, Springfield, IL 62701",
            "diagnosis": "Stage IIIA Non-Small Cell Lung Cancer",
            "tumor_volume_cm3": 12.5,
            "treatment": "Concurrent Chemoradiation",
        }


class TrialFactory:
    """Factory for synthetic clinical trial configurations."""

    @staticmethod
    def two_arm_design() -> dict:
        """Minimal two-arm trial design."""
        return {
            "trial_id": "TEST-TRIAL-001",
            "n_patients_per_arm": 50,
            "arms": {"control": 1.0, "experimental": 0.5},
        }

    @staticmethod
    def site_manifest() -> list[dict]:
        """Synthetic multi-site enrollment manifest."""
        return [
            {
                "site_id": "SITE-001",
                "site_name": "Memorial Cancer Center",
                "total_screened": 100,
                "total_enrolled": 45,
                "total_completed": 30,
                "total_withdrawn": 5,
            },
            {
                "site_id": "SITE-002",
                "site_name": "University Hospital Oncology",
                "total_screened": 80,
                "total_enrolled": 35,
                "total_completed": 20,
                "total_withdrawn": 3,
            },
        ]


class RobotFactory:
    """Factory for synthetic robot state data."""

    @staticmethod
    def joint_positions(n_joints: int = 7) -> np.ndarray:
        return np.array([0.0, 0.1, -0.2, 0.3, -0.1, 0.05, 0.0])[:n_joints]

    @staticmethod
    def ee_pose() -> dict:
        return {
            "position_m": np.array([0.0, 0.0, -0.15]),
            "orientation_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }

    @staticmethod
    def force_torque_zero() -> dict:
        return {
            "force_xyz_n": np.zeros(3),
            "torque_xyz_nm": np.zeros(3),
        }


# =========================================================================
# Robot / Safety fixtures
# =========================================================================


@pytest.fixture()
def safety_mod():
    """Load the real-time safety monitoring module."""
    return load_module(
        "safety_monitoring",
        "examples-new/01_realtime_safety_monitoring.py",
    )


@pytest.fixture()
def default_safety_limits(safety_mod):
    """Default SafetyLimits with conservative oncology parameters."""
    return safety_mod.SafetyLimits()


@pytest.fixture()
def zero_ft_reading(safety_mod):
    """Zero-load ForceTorqueReading."""
    return safety_mod.ForceTorqueReading(
        force_xyz_n=np.zeros(3),
        torque_xyz_nm=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )


@pytest.fixture()
def nominal_robot_state(safety_mod, zero_ft_reading):
    """A RobotState that passes all safety checks."""
    return safety_mod.RobotState(
        joint_positions_rad=np.array([0.0, 0.1, -0.2, 0.3, -0.1, 0.05, 0.0]),
        joint_velocities_rad_s=np.zeros(7),
        ee_position_m=np.array([0.0, 0.0, -0.15]),
        ee_velocity_m_s=np.array([0.01, 0.0, 0.0]),
        ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        force_torque=zero_ft_reading,
        timestamp_ns=1_000_000_000,
    )


# =========================================================================
# Digital-twin fixtures
# =========================================================================


@pytest.fixture()
def dt_sync_mod():
    """Load the digital-twin synchronization module."""
    return load_module(
        "dt_synchronization",
        "digital-twins/examples-twins/01_realtime_dt_synchronization.py",
    )


@pytest.fixture()
def tumor_twin_mod():
    """Load the tumor twin pipeline module."""
    return load_module(
        "tumor_twin_pipeline",
        "digital-twins/patient-modeling/tumor_twin_pipeline.py",
    )


@pytest.fixture()
def treatment_sim_mod():
    """Load the treatment simulator module."""
    return load_module(
        "treatment_simulator",
        "digital-twins/treatment-simulation/treatment_simulator.py",
    )


@pytest.fixture()
def clinical_dt_mod():
    """Load the clinical digital-twin interface module."""
    return load_module(
        "clinical_dt_interface",
        "digital-twins/clinical-integration/clinical_dt_interface.py",
    )


@pytest.fixture()
def toxicity_mod():
    """Load the multi-organ toxicity twin module."""
    return load_module(
        "multi_organ_toxicity",
        "digital-twins/examples-twins/02_multi_organ_toxicity_twin.py",
    )


@pytest.fixture()
def adaptive_rt_mod():
    """Load the adaptive radiation therapy DT module."""
    return load_module(
        "adaptive_radiation_therapy",
        "digital-twins/examples-twins/03_adaptive_radiation_therapy_dt.py",
    )


@pytest.fixture()
def immunotherapy_mod():
    """Load the immunotherapy TME DT module."""
    return load_module(
        "immunotherapy_tme",
        "digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py",
    )


@pytest.fixture()
def virtual_trial_mod():
    """Load the virtual trial cohort DT module."""
    return load_module(
        "virtual_trial_cohort",
        "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
    )


@pytest.fixture()
def dt_validation_mod():
    """Load the DT validation/verification module."""
    return load_module(
        "dt_validation",
        "digital-twins/examples-twins/06_dt_validation_verification.py",
    )


@pytest.fixture()
def baseline_patient_state() -> np.ndarray:
    """Baseline 8-dim state: [V, g, d, ANC, Cr, Hb, W, ECOG]."""
    return np.array([15.0, 0.01, 0.0, 4.5, 1.0, 13.0, 70.0, 1.0])


@pytest.fixture()
def baseline_covariance() -> np.ndarray:
    """Diagonal covariance for baseline patient state."""
    return np.diag([9.0, 0.0001, 0.01, 0.25, 0.01, 0.25, 4.0, 0.09])


@pytest.fixture()
def reference_timestamp() -> datetime:
    """Reference timestamp for time-based tests."""
    return datetime(2026, 3, 1, 9, 0)


# =========================================================================
# Dose-calculator fixtures
# =========================================================================


@pytest.fixture()
def dose_calc_mod():
    """Load the dose calculator module."""
    return load_module(
        "dose_calculator",
        "tools/dose-calculator/dose_calculator.py",
    )


@pytest.fixture()
def standard_fractionation() -> dict:
    """Standard 60 Gy / 30 fx scheme."""
    return {"total_dose": 60.0, "fractions": 30, "alpha_beta": 10.0}


@pytest.fixture()
def hypofractionation() -> dict:
    """Hypofractionated 42.56 Gy / 16 fx scheme."""
    return {"total_dose": 42.56, "fractions": 16, "alpha_beta": 10.0}


# =========================================================================
# MCP server fixtures
# =========================================================================


@pytest.fixture()
def mcp_mod():
    """Load the MCP clinical robotics server module."""
    return load_module(
        "mcp_clinical_robotics",
        "agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py",
    )


@pytest.fixture()
def mcp_server(mcp_mod):
    """Instantiate a ClinicalRoboticsMCPServer for testing."""
    return mcp_mod.ClinicalRoboticsMCPServer(trial_id="TEST-TRIAL-001")


# =========================================================================
# Agentic AI fixtures
# =========================================================================


@pytest.fixture()
def react_planner_mod():
    """Load the ReAct procedure planner module."""
    return load_module(
        "react_procedure_planner",
        "agentic-ai/examples-agentic-ai/02_react_procedure_planner.py",
    )


@pytest.fixture()
def adaptive_treatment_mod():
    """Load the real-time adaptive treatment agent module."""
    return load_module(
        "adaptive_treatment_agent",
        "agentic-ai/examples-agentic-ai/03_realtime_adaptive_treatment_agent.py",
    )


@pytest.fixture()
def sim_orchestrator_mod():
    """Load the autonomous simulation orchestrator module."""
    return load_module(
        "simulation_orchestrator",
        "agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py",
    )


@pytest.fixture()
def safety_executor_mod():
    """Load the safety-constrained agent executor module."""
    return load_module(
        "safety_executor",
        "agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py",
    )


@pytest.fixture()
def protocol_rag_mod():
    """Load the protocol RAG compliance agent module."""
    return load_module(
        "protocol_rag",
        "agentic-ai/examples-agentic-ai/06_protocol_rag_compliance_agent.py",
    )


# =========================================================================
# Tools fixtures
# =========================================================================


@pytest.fixture()
def dicom_inspector_mod():
    """Load the DICOM inspector module."""
    return load_module(
        "dicom_inspector",
        "tools/dicom-inspector/dicom_inspector.py",
    )


@pytest.fixture()
def sim_job_runner_mod():
    """Load the sim job runner module."""
    return load_module(
        "sim_job_runner",
        "tools/sim-job-runner/sim_job_runner.py",
    )


@pytest.fixture()
def trial_monitor_mod():
    """Load the trial site monitor module."""
    return load_module(
        "trial_site_monitor",
        "tools/trial-site-monitor/trial_site_monitor.py",
    )


@pytest.fixture()
def deployment_mod():
    """Load the deployment readiness module."""
    return load_module(
        "deployment_readiness",
        "tools/deployment-readiness/deployment_readiness.py",
    )


# =========================================================================
# Calibration fixtures
# =========================================================================


@pytest.fixture()
def calibration_mod():
    """Load the hand-eye calibration module."""
    return load_module(
        "hand_eye_calibration",
        "examples-new/04_hand_eye_calibration_registration.py",
    )


# =========================================================================
# Privacy fixtures
# =========================================================================


@pytest.fixture()
def deid_mod():
    """Load the de-identification pipeline module."""
    return load_module(
        "deidentification_pipeline",
        "privacy/de-identification/deidentification_pipeline.py",
    )


@pytest.fixture()
def phi_detector_mod():
    """Load the PHI detector module."""
    return load_module(
        "phi_detector",
        "privacy/phi-pii-management/phi_detector.py",
    )


@pytest.fixture()
def access_control_mod():
    """Load the access control manager module."""
    return load_module(
        "access_control_manager",
        "privacy/access-control/access_control_manager.py",
    )


@pytest.fixture()
def breach_response_mod():
    """Load the breach response protocol module."""
    return load_module(
        "breach_response_protocol",
        "privacy/breach-response/breach_response_protocol.py",
    )


@pytest.fixture()
def dua_generator_mod():
    """Load the DUA generator module."""
    return load_module(
        "dua_generator",
        "privacy/dua-templates/dua_generator.py",
    )


# =========================================================================
# Regulatory fixtures
# =========================================================================


@pytest.fixture()
def fda_tracker_mod():
    """Load the FDA submission tracker module."""
    return load_module(
        "fda_submission_tracker",
        "regulatory/fda-compliance/fda_submission_tracker.py",
    )


@pytest.fixture()
def irb_manager_mod():
    """Load the IRB protocol manager module."""
    return load_module(
        "irb_protocol_manager",
        "regulatory/irb-management/irb_protocol_manager.py",
    )


@pytest.fixture()
def gcp_checker_mod():
    """Load the GCP compliance checker module."""
    return load_module(
        "gcp_compliance_checker",
        "regulatory/ich-gcp/gcp_compliance_checker.py",
    )


@pytest.fixture()
def reg_tracker_mod():
    """Load the regulatory intelligence tracker module."""
    return load_module(
        "regulatory_tracker",
        "regulatory/regulatory-intelligence/regulatory_tracker.py",
    )


# =========================================================================
# Unification fixtures
# =========================================================================


@pytest.fixture()
def agent_interface_mod():
    """Load the unified agent interface module."""
    return load_module(
        "unified_agent_interface",
        "unification/agentic_generative_ai/unified_agent_interface.py",
    )


@pytest.fixture()
def isaac_mujoco_mod():
    """Load the Isaac-MuJoCo bridge module."""
    return load_module(
        "isaac_mujoco_bridge",
        "unification/simulation_physics/isaac_mujoco_bridge.py",
    )


@pytest.fixture()
def converter_mod():
    """Load the URDF/SDF/MJCF converter module."""
    return load_module(
        "urdf_sdf_mjcf_converter",
        "unification/simulation_physics/urdf_sdf_mjcf_converter.py",
    )


@pytest.fixture()
def framework_detector_mod():
    """Load the framework detector module."""
    return load_module(
        "framework_detector",
        "unification/cross_platform_tools/framework_detector.py",
    )


# =========================================================================
# Q1 2026 Standards fixtures
# =========================================================================


@pytest.fixture()
def isaac_to_mujoco_mod():
    """Load the Isaac-to-MuJoCo pipeline module."""
    return load_module(
        "isaac_to_mujoco_pipeline",
        "q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py",
    )


@pytest.fixture()
def mujoco_to_isaac_mod():
    """Load the MuJoCo-to-Isaac pipeline module."""
    return load_module(
        "mujoco_to_isaac_pipeline",
        "q1-2026-standards/objective-1-bidirectional-conversion/mujoco_to_isaac_pipeline.py",
    )


@pytest.fixture()
def model_validator_mod():
    """Load the model validator module."""
    return load_module(
        "model_validator",
        "q1-2026-standards/objective-2-robot-model-repository/model_validator.py",
    )


@pytest.fixture()
def benchmark_runner_mod():
    """Load the benchmark runner module."""
    return load_module(
        "benchmark_runner",
        "q1-2026-standards/objective-3-validation-benchmark/benchmark_runner.py",
    )


# =========================================================================
# Physical robot example fixtures
# =========================================================================


@pytest.fixture()
def sensor_fusion_mod():
    """Load the sensor fusion intraoperative module."""
    return load_module(
        "sensor_fusion",
        "examples-new/02_sensor_fusion_intraoperative.py",
    )


@pytest.fixture()
def ros2_deployment_mod():
    """Load the ROS 2 surgical deployment module."""
    return load_module(
        "ros2_deployment",
        "examples-new/03_ros2_surgical_deployment.py",
    )


@pytest.fixture()
def shared_autonomy_mod():
    """Load the shared autonomy teleoperation module."""
    return load_module(
        "shared_autonomy",
        "examples-new/05_shared_autonomy_teleoperation.py",
    )


@pytest.fixture()
def sample_handling_mod():
    """Load the robotic sample handling module."""
    return load_module(
        "sample_handling",
        "examples-new/06_robotic_sample_handling.py",
    )


# =========================================================================
# Scripts fixtures
# =========================================================================


@pytest.fixture()
def verify_install_mod():
    """Load the verify_installation script module."""
    return load_module(
        "verify_installation",
        "scripts/verify_installation.py",
    )


# =========================================================================
# Shared synthetic data fixtures
# =========================================================================


@pytest.fixture()
def sample_patient_record() -> dict:
    """Synthetic patient record with PHI for deidentification testing."""
    return PatientFactory.patient_record_with_phi()


@pytest.fixture()
def synthetic_tumor_volume() -> np.ndarray:
    """Synthetic 3-D tumor volume."""
    return PatientFactory.tumor_volume()


@pytest.fixture()
def synthetic_dose_distribution() -> np.ndarray:
    """Synthetic 3-D dose distribution."""
    return PatientFactory.dose_distribution()
