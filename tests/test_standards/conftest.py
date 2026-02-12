"""Fixtures for standards and unification tests."""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def isaac_to_mujoco_mod():
    """Load the Isaac to MuJoCo pipeline module."""
    return load_module(
        "isaac_to_mujoco",
        "q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py",
    )


@pytest.fixture()
def mujoco_to_isaac_mod():
    """Load the MuJoCo to Isaac pipeline module."""
    return load_module(
        "mujoco_to_isaac",
        "q1-2026-standards/objective-1-bidirectional-conversion/mujoco_to_isaac_pipeline.py",
    )


@pytest.fixture()
def physics_equiv_mod():
    """Load the physics equivalence tests module."""
    return load_module(
        "physics_equiv",
        "q1-2026-standards/objective-1-bidirectional-conversion/physics_equivalence_tests.py",
    )


@pytest.fixture()
def model_validator_mod():
    """Load the model validator module."""
    return load_module(
        "model_validator",
        "q1-2026-standards/objective-2-robot-model-repository/model_validator.py",
    )


@pytest.fixture()
def benchmark_mod():
    """Load the benchmark runner module."""
    return load_module(
        "benchmark_runner",
        "q1-2026-standards/objective-3-validation-benchmark/benchmark_runner.py",
    )


@pytest.fixture()
def unified_agent_mod():
    """Load the unified agent interface module."""
    return load_module(
        "unified_agent_interface",
        "unification/agentic_generative_ai/unified_agent_interface.py",
    )


@pytest.fixture()
def framework_detector_mod():
    """Load the framework detector module."""
    return load_module(
        "framework_detector",
        "unification/cross_platform_tools/framework_detector.py",
    )


@pytest.fixture()
def validation_suite_mod():
    """Load the validation suite module."""
    return load_module(
        "validation_suite",
        "unification/cross_platform_tools/validation_suite.py",
    )


@pytest.fixture()
def bridge_mod():
    """Load the Isaac-MuJoCo bridge module."""
    return load_module(
        "isaac_mujoco_bridge",
        "unification/simulation_physics/isaac_mujoco_bridge.py",
    )


@pytest.fixture()
def converter_mod():
    """Load the URDF/SDF/MJCF converter module."""
    return load_module(
        "urdf_converter",
        "unification/simulation_physics/urdf_sdf_mjcf_converter.py",
    )
