"""Fixtures for expanded regression tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def dt_sync_mod():
    """Load digital-twin synchronization module."""
    return load_module(
        "dt_synchronization",
        "digital-twins/examples-twins/01_realtime_dt_synchronization.py",
    )


@pytest.fixture()
def cohort_mod():
    """Load virtual trial cohort module."""
    return load_module(
        "virtual_trial_cohort",
        "digital-twins/examples-twins/05_virtual_trial_cohort_dt.py",
    )


@pytest.fixture()
def dose_mod():
    """Load dose calculator module."""
    return load_module(
        "dose_calculator",
        "tools/dose-calculator/dose_calculator.py",
    )


@pytest.fixture()
def twin_mod():
    """Load tumor twin pipeline module."""
    return load_module(
        "tumor_twin_pipeline",
        "digital-twins/patient-modeling/tumor_twin_pipeline.py",
    )


@pytest.fixture()
def deid_mod():
    """Load deidentification pipeline module."""
    return load_module(
        "deidentification_pipeline",
        "privacy/de-identification/deidentification_pipeline.py",
    )


@pytest.fixture()
def safety_mod():
    """Load safety monitoring module."""
    return load_module(
        "safety_monitoring",
        "examples-new/01_realtime_safety_monitoring.py",
    )


@pytest.fixture()
def access_mod():
    """Load access control module."""
    return load_module(
        "access_control_manager",
        "privacy/access-control/access_control_manager.py",
    )


@pytest.fixture()
def breach_mod():
    """Load breach response module."""
    return load_module(
        "breach_response_protocol",
        "privacy/breach-response/breach_response_protocol.py",
    )


@pytest.fixture()
def reg_tracker_mod():
    """Load regulatory tracker module."""
    return load_module(
        "regulatory_tracker",
        "regulatory/regulatory-intelligence/regulatory_tracker.py",
    )


@pytest.fixture()
def gcp_mod():
    """Load GCP compliance checker module."""
    return load_module(
        "gcp_compliance_checker",
        "regulatory/ich-gcp/gcp_compliance_checker.py",
    )
