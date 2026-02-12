"""Fixtures for digital twin tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


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
        "adaptive_rt",
        "digital-twins/examples-twins/03_adaptive_radiation_therapy_dt.py",
    )


@pytest.fixture()
def tme_mod():
    """Load the tumor microenvironment module."""
    return load_module(
        "tumor_microenvironment",
        "digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py",
    )


@pytest.fixture()
def cohort_mod():
    """Load the virtual trial cohort DT module."""
    return load_module(
        "virtual_trial_cohort",
        "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
    )


@pytest.fixture()
def vv_mod():
    """Load the DT validation and verification module."""
    return load_module(
        "dt_validation",
        "digital-twins/examples-twins/06_dt_validation_verification.py",
    )


@pytest.fixture()
def clinical_dt_mod():
    """Load the clinical DT interface module."""
    return load_module(
        "clinical_dt_interface",
        "digital-twins/clinical-integration/clinical_dt_interface.py",
    )


@pytest.fixture()
def twin_pipeline_mod():
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
def synthetic_planning_ct():
    """Synthetic 3D planning CT volume."""
    return np.random.rand(20, 20, 20).astype(np.float32) * 1000


@pytest.fixture()
def synthetic_tumor_mask():
    """Synthetic 3D binary tumor mask."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[8:12, 8:12, 8:12] = 1.0
    return mask
