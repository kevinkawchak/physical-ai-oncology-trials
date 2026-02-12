from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("shared_autonomy_teleoperation", "examples-new/05_shared_autonomy_teleoperation.py")


# ---------------------------------------------------------------------------
# TestAutonomyLevel
# ---------------------------------------------------------------------------


class TestAutonomyLevel:
    """Tests for AutonomyLevel enum."""

    def test_teleop_raw(self, mod):
        assert mod.AutonomyLevel.TELEOP_RAW.value == 0

    def test_teleop_guided(self, mod):
        assert mod.AutonomyLevel.TELEOP_GUIDED.value == 1

    def test_teleop_constrained(self, mod):
        assert mod.AutonomyLevel.TELEOP_CONSTRAINED.value == 2

    def test_shared_blended(self, mod):
        assert mod.AutonomyLevel.SHARED_BLENDED.value == 3

    def test_autonomous_supervised(self, mod):
        assert mod.AutonomyLevel.AUTONOMOUS_SUPERVISED.value == 4


# ---------------------------------------------------------------------------
# TestVirtualFixtureType
# ---------------------------------------------------------------------------


class TestVirtualFixtureType:
    """Tests for VirtualFixtureType enum."""

    def test_forbidden_region(self, mod):
        assert mod.VirtualFixtureType.FORBIDDEN_REGION is not None

    def test_guidance_path(self, mod):
        assert mod.VirtualFixtureType.GUIDANCE_PATH is not None

    def test_boundary_plane(self, mod):
        assert mod.VirtualFixtureType.BOUNDARY_PLANE is not None


# ---------------------------------------------------------------------------
# TestSharedAutonomyConfig
# ---------------------------------------------------------------------------


class TestSharedAutonomyConfig:
    """Tests for SharedAutonomyConfig dataclass."""

    def test_defaults(self, mod):
        cfg = mod.SharedAutonomyConfig()
        assert cfg.autonomy_level == mod.AutonomyLevel.TELEOP_CONSTRAINED
        assert cfg.surgeon_authority == 0.7

    def test_custom_values(self, mod):
        cfg = mod.SharedAutonomyConfig(
            autonomy_level=mod.AutonomyLevel.SHARED_BLENDED,
            surgeon_authority=0.5,
            virtual_fixture_stiffness=3000.0,
        )
        assert cfg.autonomy_level == mod.AutonomyLevel.SHARED_BLENDED
        assert cfg.surgeon_authority == 0.5
        assert cfg.virtual_fixture_stiffness == 3000.0

    def test_tremor_filter_default(self, mod):
        cfg = mod.SharedAutonomyConfig()
        assert cfg.tremor_filter_cutoff_hz == 6.0

    def test_scaling_factor_default(self, mod):
        cfg = mod.SharedAutonomyConfig()
        assert cfg.scaling_factor == 0.3


# ---------------------------------------------------------------------------
# TestSurgeonInputProcessor
# ---------------------------------------------------------------------------


class TestSurgeonInputProcessor:
    """Tests for SurgeonInputProcessor class."""

    def test_init(self, mod):
        cfg = mod.SharedAutonomyConfig()
        proc = mod.SurgeonInputProcessor(cfg)
        assert proc.config is cfg

    def test_process_returns_tuple(self, mod):
        cfg = mod.SharedAutonomyConfig()
        proc = mod.SurgeonInputProcessor(cfg)
        pos, orient = proc.process(
            input_position_m=np.zeros(3),
            input_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            is_clutched=False,
            current_robot_position_m=np.zeros(3),
        )
        assert isinstance(pos, np.ndarray)
        assert isinstance(orient, np.ndarray)

    def test_clutched_returns_current_position(self, mod):
        cfg = mod.SharedAutonomyConfig()
        proc = mod.SurgeonInputProcessor(cfg)
        current = np.array([0.01, 0.02, -0.15])
        pos, _ = proc.process(
            input_position_m=np.array([0.1, 0.1, 0.1]),
            input_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            is_clutched=True,
            current_robot_position_m=current,
        )
        np.testing.assert_array_equal(pos, current)


# ---------------------------------------------------------------------------
# TestVirtualFixture
# ---------------------------------------------------------------------------


class TestVirtualFixture:
    """Tests for VirtualFixture dataclass."""

    def test_creation(self, mod):
        vf = mod.VirtualFixture(
            fixture_type=mod.VirtualFixtureType.FORBIDDEN_REGION,
            name="artery",
            parameters={"center_m": np.zeros(3), "radius_m": 0.01},
        )
        assert vf.name == "artery"
        assert vf.fixture_type == mod.VirtualFixtureType.FORBIDDEN_REGION

    def test_default_stiffness(self, mod):
        vf = mod.VirtualFixture(
            fixture_type=mod.VirtualFixtureType.BOUNDARY_PLANE,
            name="boundary",
            parameters={},
        )
        assert vf.stiffness == 2000.0

    def test_default_active(self, mod):
        vf = mod.VirtualFixture(
            fixture_type=mod.VirtualFixtureType.GUIDANCE_PATH,
            name="path",
            parameters={},
        )
        assert vf.is_active is True


# ---------------------------------------------------------------------------
# TestVirtualFixtureEngine
# ---------------------------------------------------------------------------


class TestVirtualFixtureEngine:
    """Tests for VirtualFixtureEngine class."""

    def test_init(self, mod):
        engine = mod.VirtualFixtureEngine()
        assert len(engine._fixtures) == 0

    def test_add_forbidden_region(self, mod):
        engine = mod.VirtualFixtureEngine()
        engine.add_forbidden_region("vessel", center_m=np.zeros(3), radius_m=0.01)
        assert len(engine._fixtures) == 1

    def test_compute_no_fixtures_passthrough(self, mod):
        engine = mod.VirtualFixtureEngine()
        target = np.array([0.01, 0.02, 0.03])
        force, constrained = engine.compute(
            current_position_m=np.zeros(3),
            target_position_m=target,
            velocity_m_s=np.zeros(3),
        )
        np.testing.assert_array_almost_equal(constrained, target)
        np.testing.assert_array_almost_equal(force, np.zeros(3))

    def test_compute_forbidden_region_repels(self, mod):
        engine = mod.VirtualFixtureEngine()
        center = np.array([0.0, 0.0, 0.0])
        engine.add_forbidden_region("test", center_m=center, radius_m=0.05)
        # Target inside the forbidden region
        target = np.array([0.01, 0.0, 0.0])
        force, constrained = engine.compute(
            current_position_m=np.array([0.06, 0.0, 0.0]),
            target_position_m=target,
            velocity_m_s=np.zeros(3),
        )
        # Constrained position should be pushed outward
        assert np.linalg.norm(constrained) >= 0.049


# ---------------------------------------------------------------------------
# TestCommandBlender
# ---------------------------------------------------------------------------


class TestCommandBlender:
    """Tests for CommandBlender class."""

    def test_teleop_raw_returns_surgeon(self, mod):
        cfg = mod.SharedAutonomyConfig(surgeon_authority=0.7)
        blender = mod.CommandBlender(cfg)
        surgeon = np.array([1.0, 0.0, 0.0])
        ai = np.array([0.0, 1.0, 0.0])
        result = blender.blend(surgeon, ai, np.zeros(3), mod.AutonomyLevel.TELEOP_RAW)
        np.testing.assert_array_equal(result, surgeon)

    def test_autonomous_returns_ai(self, mod):
        cfg = mod.SharedAutonomyConfig(surgeon_authority=0.7)
        blender = mod.CommandBlender(cfg)
        surgeon = np.array([1.0, 0.0, 0.0])
        ai = np.array([0.0, 1.0, 0.0])
        result = blender.blend(surgeon, ai, np.zeros(3), mod.AutonomyLevel.AUTONOMOUS_SUPERVISED)
        np.testing.assert_array_equal(result, ai)

    def test_shared_blended_mixes(self, mod):
        cfg = mod.SharedAutonomyConfig(surgeon_authority=0.5)
        blender = mod.CommandBlender(cfg)
        surgeon = np.array([1.0, 0.0, 0.0])
        ai = np.array([0.0, 1.0, 0.0])
        result = blender.blend(surgeon, ai, np.zeros(3), mod.AutonomyLevel.SHARED_BLENDED)
        expected = 0.5 * surgeon + 0.5 * ai
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# TestSharedAutonomyController
# ---------------------------------------------------------------------------


class TestSharedAutonomyController:
    """Tests for SharedAutonomyController class."""

    def test_init(self, mod):
        cfg = mod.SharedAutonomyConfig()
        ctrl = mod.SharedAutonomyController(cfg)
        assert ctrl.autonomy_level == mod.AutonomyLevel.TELEOP_CONSTRAINED

    def test_set_autonomy_level(self, mod):
        cfg = mod.SharedAutonomyConfig()
        ctrl = mod.SharedAutonomyController(cfg)
        ctrl.set_autonomy_level(mod.AutonomyLevel.SHARED_BLENDED)
        assert ctrl.autonomy_level == mod.AutonomyLevel.SHARED_BLENDED

    def test_compute_returns_dict(self, mod):
        cfg = mod.SharedAutonomyConfig()
        ctrl = mod.SharedAutonomyController(cfg)
        result = ctrl.compute(
            surgeon_input_position_m=np.zeros(3),
            surgeon_input_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ai_target_position_m=np.zeros(3),
            current_robot_position_m=np.zeros(3),
            current_robot_velocity_m_s=np.zeros(3),
        )
        assert isinstance(result, dict)
        assert "target_position_m" in result
        assert "haptic_force_n" in result
        assert "autonomy_level" in result

    def test_add_vessel_fixture(self, mod):
        cfg = mod.SharedAutonomyConfig()
        ctrl = mod.SharedAutonomyController(cfg)
        ctrl.add_vessel_fixture("artery", center_m=np.zeros(3), radius_m=0.008)
        assert len(ctrl._fixture_engine._fixtures) == 1
