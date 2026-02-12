"""Fixtures for examples/ module tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def surgical_training_mod():
    """Load the surgical robot training module."""
    return load_module(
        "surgical_robot_training",
        "examples/01_surgical_robot_training.py",
    )


@pytest.fixture()
def surgical_planning_mod():
    """Load the digital twin surgical planning module."""
    return load_module(
        "dt_surgical_planning",
        "examples/02_digital_twin_surgical_planning.py",
    )


@pytest.fixture()
def cross_framework_mod():
    """Load the cross-framework validation module."""
    return load_module(
        "cross_framework_validation",
        "examples/03_cross_framework_validation.py",
    )


@pytest.fixture()
def clinical_workflow_mod():
    """Load the agentic clinical workflow module."""
    return load_module(
        "agentic_clinical_workflow",
        "examples/04_agentic_clinical_workflow.py",
    )


@pytest.fixture()
def treatment_prediction_mod():
    """Load the treatment response prediction module."""
    return load_module(
        "treatment_response_prediction",
        "examples/05_treatment_response_prediction.py",
    )
