"""Fixtures for privacy module tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def access_mod():
    """Load the access control manager module."""
    return load_module(
        "access_control_manager",
        "privacy/access-control/access_control_manager.py",
    )


@pytest.fixture()
def breach_mod():
    """Load the breach response protocol module."""
    return load_module(
        "breach_response_protocol",
        "privacy/breach-response/breach_response_protocol.py",
    )


@pytest.fixture()
def dua_mod():
    """Load the DUA generator module."""
    return load_module(
        "dua_generator",
        "privacy/dua-templates/dua_generator.py",
    )


@pytest.fixture()
def phi_mod():
    """Load the PHI detector module."""
    return load_module(
        "phi_detector",
        "privacy/phi-pii-management/phi_detector.py",
    )
