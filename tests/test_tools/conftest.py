"""Fixtures for CLI tools tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def deploy_mod():
    """Load the deployment readiness module."""
    return load_module(
        "deployment_readiness",
        "tools/deployment-readiness/deployment_readiness.py",
    )


@pytest.fixture()
def dicom_mod():
    """Load the DICOM inspector module."""
    return load_module(
        "dicom_inspector",
        "tools/dicom-inspector/dicom_inspector.py",
    )


@pytest.fixture()
def sim_runner_mod():
    """Load the sim job runner module."""
    return load_module(
        "sim_job_runner",
        "tools/sim-job-runner/sim_job_runner.py",
    )


@pytest.fixture()
def site_monitor_mod():
    """Load the trial site monitor module."""
    return load_module(
        "trial_site_monitor",
        "tools/trial-site-monitor/trial_site_monitor.py",
    )
