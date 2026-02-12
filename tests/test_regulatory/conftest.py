"""Fixtures for regulatory module tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def fda_mod():
    """Load the FDA submission tracker module."""
    return load_module(
        "fda_submission_tracker",
        "regulatory/fda-compliance/fda_submission_tracker.py",
    )


@pytest.fixture()
def gcp_mod():
    """Load the GCP compliance checker module."""
    return load_module(
        "gcp_compliance_checker",
        "regulatory/ich-gcp/gcp_compliance_checker.py",
    )


@pytest.fixture()
def irb_mod():
    """Load the IRB protocol manager module."""
    return load_module(
        "irb_protocol_manager",
        "regulatory/irb-management/irb_protocol_manager.py",
    )


@pytest.fixture()
def reg_tracker_mod():
    """Load the regulatory tracker module."""
    return load_module(
        "regulatory_tracker",
        "regulatory/regulatory-intelligence/regulatory_tracker.py",
    )
