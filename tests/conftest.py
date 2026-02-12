"""Shared fixtures and helpers for the physical-ai-oncology-trials test suite.

All source modules live in directories with hyphens (e.g. ``examples-new/``)
which cannot be imported with a normal ``import`` statement.  The
``load_module`` helper uses ``importlib.util`` to load them by file path.

Every fixture seeds the NumPy RNG for deterministic, reproducible tests.

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

    If the module depends on packages that are not installed in the
    current environment (e.g. torch, mujoco, langchain, monai) the
    test is automatically **skipped** instead of erroring out.  This
    keeps CI green when only core dependencies (numpy, scipy, pytest)
    are available.

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
    if not filepath.exists():
        pytest.skip(f"Source file not found: {relative_path}")

    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        # Remove the partially-initialised module so later attempts
        # also trigger a skip rather than returning a broken object.
        sys.modules.pop(name, None)
        pytest.skip(f"Module {name!r} requires a dependency not installed in this environment: {exc}")
    return mod


# ---------------------------------------------------------------------------
# Deterministic random state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed NumPy random state for deterministic tests."""
    np.random.seed(42)
    yield


# ---------------------------------------------------------------------------
# Robot / Safety fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Digital-twin fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dt_sync_mod():
    """Load the digital-twin synchronization module."""
    return load_module(
        "dt_synchronization",
        "digital-twins/examples-twins/01_realtime_dt_synchronization.py",
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


# ---------------------------------------------------------------------------
# Dose-calculator fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MCP server fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Calibration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calibration_mod():
    """Load the hand-eye calibration module."""
    return load_module(
        "hand_eye_calibration",
        "examples-new/04_hand_eye_calibration_registration.py",
    )


# ---------------------------------------------------------------------------
# Deidentification fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def deid_mod():
    """Load the de-identification pipeline module."""
    return load_module(
        "deidentification_pipeline",
        "privacy/de-identification/deidentification_pipeline.py",
    )


@pytest.fixture()
def sample_patient_record() -> dict:
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


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_tumor_geometry() -> dict:
    """Synthetic 3D tumor geometry for digital twin tests."""
    return {
        "center_mm": np.array([50.0, 30.0, -20.0]),
        "axes_mm": np.array([15.0, 12.0, 10.0]),
        "volume_cm3": 4.0 / 3.0 * np.pi * 15.0 * 12.0 * 10.0 / 1000.0,
        "voxel_spacing_mm": np.array([1.0, 1.0, 2.5]),
    }


@pytest.fixture()
def synthetic_dose_distribution() -> dict:
    """Synthetic 3D dose distribution for radiation tests."""
    grid = np.zeros((20, 20, 10), dtype=np.float64)
    grid[5:15, 5:15, 2:8] = 2.0  # 2 Gy per fraction in PTV
    return {
        "dose_grid_gy": grid,
        "prescription_gy": 60.0,
        "fractions": 30,
        "voxel_volume_cm3": 0.1 * 0.1 * 0.25,
    }


@pytest.fixture()
def trial_cohort_config() -> dict:
    """Configuration for virtual trial cohort generation."""
    return {
        "n_patients": 50,
        "tumor_site": "NSCLC",
        "treatment_arms": ["control", "experimental"],
        "primary_endpoint": "PFS",
        "follow_up_months": 12,
        "randomization_ratio": [1, 1],
    }
