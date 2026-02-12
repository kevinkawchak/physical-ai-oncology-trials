"""Tests for examples-new/05_shared_autonomy_teleoperation.py

Covers autonomy levels, virtual fixtures, command blending,
and shared autonomy controller.
"""

from __future__ import annotations

import numpy as np


class TestEnums:
    def test_autonomy_levels(self, shared_autonomy_mod):
        al = shared_autonomy_mod.AutonomyLevel
        assert al.TELEOP_RAW.value == 0
        assert al.AUTONOMOUS_SUPERVISED.value == 4

    def test_virtual_fixture_types(self, shared_autonomy_mod):
        vft = shared_autonomy_mod.VirtualFixtureType
        assert vft.FORBIDDEN_REGION is not None
        assert vft.GUIDANCE_PATH is not None
        assert vft.BOUNDARY_PLANE is not None


class TestSharedAutonomyConfig:
    def test_defaults(self, shared_autonomy_mod):
        cfg = shared_autonomy_mod.SharedAutonomyConfig()
        assert cfg.surgeon_authority == 0.7
        assert cfg.scaling_factor == 0.3
        assert cfg.tremor_filter_cutoff_hz == 6.0


class TestVirtualFixtureEngine:
    def test_creation(self, shared_autonomy_mod):
        engine = shared_autonomy_mod.VirtualFixtureEngine()
        assert engine is not None

    def test_add_forbidden_region(self, shared_autonomy_mod):
        engine = shared_autonomy_mod.VirtualFixtureEngine()
        engine.add_forbidden_region(
            name="vessel",
            center_m=np.array([0.1, 0.0, 0.05]),
            radius_m=0.008,
        )
        # No crash means it was added

    def test_compute_no_fixtures(self, shared_autonomy_mod):
        engine = shared_autonomy_mod.VirtualFixtureEngine()
        pos = np.array([0.1, 0.0, 0.05])
        target = np.array([0.11, 0.0, 0.05])
        vel = np.zeros(3)
        correction, force = engine.compute(pos, target, vel)
        assert correction.shape == (3,)
        assert force.shape == (3,)


class TestCommandBlender:
    def test_creation(self, shared_autonomy_mod):
        cfg = shared_autonomy_mod.SharedAutonomyConfig()
        blender = shared_autonomy_mod.CommandBlender(config=cfg)
        assert blender is not None

    def test_blend_teleop_raw(self, shared_autonomy_mod):
        cfg = shared_autonomy_mod.SharedAutonomyConfig()
        blender = shared_autonomy_mod.CommandBlender(config=cfg)
        surgeon = np.array([0.1, 0.0, 0.0])
        ai = np.array([0.2, 0.0, 0.0])
        current = np.zeros(3)
        result = blender.blend(
            surgeon,
            ai,
            current,
            shared_autonomy_mod.AutonomyLevel.TELEOP_RAW,
        )
        assert result.shape == (3,)
        # In raw teleop, should follow surgeon command
        np.testing.assert_allclose(result, surgeon, atol=0.01)


class TestSharedAutonomyController:
    def test_creation(self, shared_autonomy_mod):
        cfg = shared_autonomy_mod.SharedAutonomyConfig()
        controller = shared_autonomy_mod.SharedAutonomyController(config=cfg)
        assert controller.autonomy_level == cfg.autonomy_level

    def test_set_autonomy_level(self, shared_autonomy_mod):
        cfg = shared_autonomy_mod.SharedAutonomyConfig()
        controller = shared_autonomy_mod.SharedAutonomyController(config=cfg)
        controller.set_autonomy_level(
            shared_autonomy_mod.AutonomyLevel.SHARED_BLENDED,
        )
        assert controller.autonomy_level == shared_autonomy_mod.AutonomyLevel.SHARED_BLENDED

    def test_compute_returns_dict(self, shared_autonomy_mod):
        cfg = shared_autonomy_mod.SharedAutonomyConfig()
        controller = shared_autonomy_mod.SharedAutonomyController(config=cfg)
        result = controller.compute(
            surgeon_input_position_m=np.array([0.1, 0.0, 0.05]),
            surgeon_input_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ai_target_position_m=np.array([0.1, 0.0, 0.06]),
            current_robot_position_m=np.array([0.1, 0.0, 0.05]),
            current_robot_velocity_m_s=np.zeros(3),
        )
        assert isinstance(result, dict)
