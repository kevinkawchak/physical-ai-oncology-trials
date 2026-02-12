"""Unit tests for unification/simulation_physics/isaac_mujoco_bridge.py.

Tests cover PhysicsParameters, ContactParameters dataclasses,
PhysicsParameterMapper, StateConverter, and framework-specific conversions.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "unification/simulation_physics/isaac_mujoco_bridge.py"


@pytest.fixture()
def mod():
    return load_module("isaac_mujoco_bridge", MOD_PATH)


class TestPhysicsParameters:
    def test_creation(self, mod):
        params = mod.PhysicsParameters(
            timestep=0.002,
            gravity=np.array([0.0, 0.0, -9.81]),
            friction_coefficient=0.5,
            contact_stiffness=1000.0,
            contact_damping=100.0,
            solver_iterations=50,
        )
        assert params.timestep == 0.002
        assert params.gravity[2] == pytest.approx(-9.81)

    def test_positive_timestep(self, mod):
        params = mod.PhysicsParameters(
            timestep=0.001,
            gravity=np.array([0.0, 0.0, -9.81]),
            friction_coefficient=0.3,
            contact_stiffness=500.0,
            contact_damping=50.0,
            solver_iterations=20,
        )
        assert params.timestep > 0


class TestContactParameters:
    def test_creation(self, mod):
        contact = mod.ContactParameters(stiffness=1000.0, damping=100.0)
        assert contact.stiffness == 1000.0
        assert contact.damping == 100.0

    def test_to_isaac_physx(self, mod):
        contact = mod.ContactParameters(stiffness=1000.0, damping=100.0)
        physx = contact.to_isaac_physx()
        assert physx is not None
        assert isinstance(physx, dict)

    def test_to_mujoco_solref(self, mod):
        contact = mod.ContactParameters(stiffness=500.0, damping=50.0)
        solref = contact.to_mujoco_solref()
        assert solref is not None

    def test_to_pybullet(self, mod):
        contact = mod.ContactParameters(stiffness=800.0, damping=80.0)
        pb = contact.to_pybullet()
        assert pb is not None


class TestPhysicsParameterMapper:
    def test_parameter_mapping_exists(self, mod):
        mapper = mod.PhysicsParameterMapper()
        assert hasattr(mapper, "PARAMETER_MAPPING") or hasattr(mapper, "convert_parameters")


class TestStateConverter:
    def test_converter_exists(self, mod):
        converter = mod.StateConverter()
        assert hasattr(converter, "isaac_to_mujoco")
        assert hasattr(converter, "mujoco_to_isaac")
