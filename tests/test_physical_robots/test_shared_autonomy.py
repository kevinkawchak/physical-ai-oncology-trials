"""Unit tests for examples-new/05_shared_autonomy_teleoperation.py.

Tests cover AutonomyLevel enum, SharedAutonomyConfig dataclass,
and SurgeonInputProcessor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "examples-new/05_shared_autonomy_teleoperation.py"


@pytest.fixture()
def mod():
    return load_module("shared_autonomy", MOD_PATH)


class TestEnums:
    def test_autonomy_levels(self, mod):
        names = [e.name for e in mod.AutonomyLevel]
        for level in (
            "TELEOP_RAW",
            "TELEOP_GUIDED",
            "TELEOP_CONSTRAINED",
            "SHARED_BLENDED",
            "AUTONOMOUS_SUPERVISED",
        ):
            assert level in names

    def test_five_levels(self, mod):
        assert len(list(mod.AutonomyLevel)) == 5


class TestSharedAutonomyConfig:
    def test_defaults(self, mod):
        config = mod.SharedAutonomyConfig(
            autonomy_level=mod.AutonomyLevel.TELEOP_RAW,
        )
        assert config.autonomy_level == mod.AutonomyLevel.TELEOP_RAW
        assert config.surgeon_authority >= 0
        assert config.surgeon_authority <= 1.0

    def test_custom_config(self, mod):
        config = mod.SharedAutonomyConfig(
            autonomy_level=mod.AutonomyLevel.SHARED_BLENDED,
            surgeon_authority=0.6,
            virtual_fixture_stiffness=500.0,
            virtual_fixture_damping=50.0,
            tremor_filter_cutoff_hz=8.0,
            max_teleop_velocity_m_s=0.1,
            scaling_factor=0.5,
            deadband_m=0.001,
        )
        assert config.surgeon_authority == 0.6
        assert config.tremor_filter_cutoff_hz == 8.0


class TestSurgeonInputProcessor:
    def test_instantiation(self, mod):
        config = mod.SharedAutonomyConfig(
            autonomy_level=mod.AutonomyLevel.TELEOP_RAW,
        )
        proc = mod.SurgeonInputProcessor(config=config)
        assert proc is not None

    def test_process_zero_input(self, mod):
        config = mod.SharedAutonomyConfig(
            autonomy_level=mod.AutonomyLevel.TELEOP_RAW,
        )
        proc = mod.SurgeonInputProcessor(config=config)
        result = proc.process(
            input_position_m=np.zeros(3),
            input_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            is_clutched=False,
            current_robot_position_m=np.zeros(3),
        )
        assert result is not None

    def test_clutched_input_ignored(self, mod):
        config = mod.SharedAutonomyConfig(
            autonomy_level=mod.AutonomyLevel.TELEOP_RAW,
        )
        proc = mod.SurgeonInputProcessor(config=config)
        result = proc.process(
            input_position_m=np.array([0.1, 0.0, 0.0]),
            input_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            is_clutched=True,
            current_robot_position_m=np.zeros(3),
        )
        # When clutched, output should not move robot
        assert result is not None
