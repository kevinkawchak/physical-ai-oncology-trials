"""Unit tests for q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py.

Tests cover PhysXContactParams, MuJoCoContactParams, ConversionConfig
dataclasses, and PhysicsParameterConverter.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py"


@pytest.fixture()
def mod():
    return load_module("isaac_to_mujoco_pipeline", MOD_PATH)


class TestPhysXContactParams:
    def test_creation(self, mod):
        params = mod.PhysXContactParams(
            contact_offset=0.02,
            rest_offset=0.0,
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.1,
        )
        assert params.static_friction == 0.5
        assert params.dynamic_friction == 0.4

    def test_friction_ordering(self, mod):
        params = mod.PhysXContactParams(
            contact_offset=0.02,
            rest_offset=0.0,
            static_friction=0.6,
            dynamic_friction=0.4,
            restitution=0.1,
        )
        assert params.static_friction >= params.dynamic_friction


class TestMuJoCoContactParams:
    def test_creation(self, mod):
        params = mod.MuJoCoContactParams(
            solref=(-1000.0, -100.0),
            solimp=(0.9, 0.95, 0.001),
            friction=(0.5, 0.005, 0.0001),
        )
        assert params.solref[0] < 0  # Stiffness is negative in solref


class TestConversionConfig:
    def test_defaults(self, mod):
        config = mod.ConversionConfig()
        assert config.timestep > 0
        assert config.solver_iterations > 0


class TestPhysicsParameterConverter:
    def test_physx_to_mujoco(self, mod):
        converter = mod.PhysicsParameterConverter()
        physx = mod.PhysXContactParams(
            contact_offset=0.02,
            rest_offset=0.0,
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.1,
        )
        mujoco_params = converter.physx_to_mujoco_contact(physx)
        assert isinstance(mujoco_params, mod.MuJoCoContactParams)

    def test_mujoco_to_physx(self, mod):
        converter = mod.PhysicsParameterConverter()
        mujoco = mod.MuJoCoContactParams(
            solref=(-1000.0, -100.0),
            solimp=(0.9, 0.95, 0.001),
            friction=(0.5, 0.005, 0.0001),
        )
        physx_params = converter.mujoco_to_physx_contact(mujoco)
        assert isinstance(physx_params, mod.PhysXContactParams)
