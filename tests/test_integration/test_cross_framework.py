"""Integration tests: Multi-framework simulation validation.

Tests cross-module interactions between the validation suite, framework
detector, and Isaac-MuJoCo bridge modules to ensure consistent behavior
across simulated environments.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def val_mod():
    return load_module("validation_suite", "unification/cross_platform_tools/validation_suite.py")


@pytest.fixture()
def fw_mod():
    return load_module("framework_detector", "unification/cross_platform_tools/framework_detector.py")


@pytest.fixture()
def bridge_mod():
    return load_module("isaac_mujoco_bridge", "unification/simulation_physics/isaac_mujoco_bridge.py")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossFramework:
    """Integration: multi-framework simulation validation."""

    def test_mock_environment_consistent(self, val_mod):
        """MockEnvironment produces same results with same seed across two runs."""
        np.random.seed(123)
        env1 = val_mod.MockEnvironment("isaac", "needle_insertion")
        obs1 = env1.reset()
        _, r1, _, _ = env1.step(np.zeros(7))
        env1.close()

        np.random.seed(123)
        env2 = val_mod.MockEnvironment("isaac", "needle_insertion")
        obs2 = env2.reset()
        _, r2, _, _ = env2.step(np.zeros(7))
        env2.close()

        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2

    def test_validation_metrics_bounded(self, val_mod):
        """ValidationMetrics fields are in [0,1] or non-negative."""
        metrics = val_mod.ValidationMetrics(
            framework="isaac",
            task="needle_insertion",
            episodes=10,
            success_rate=0.8,
            mean_reward=5.0,
            std_reward=1.0,
            mean_episode_length=100.0,
            force_violations=2,
            position_accuracy=0.005,
            execution_time=2.5,
        )
        assert 0.0 <= metrics.success_rate <= 1.0
        assert metrics.force_violations >= 0
        assert metrics.position_accuracy >= 0
        assert metrics.execution_time >= 0
        assert metrics.episodes > 0

    def test_cross_platform_comparison(self, val_mod):
        """Two mock environments produce comparable (finite) metrics."""
        env_isaac = val_mod.MockEnvironment("isaac", "needle_insertion")
        env_mujoco = val_mod.MockEnvironment("mujoco", "needle_insertion")

        obs_i = env_isaac.reset()
        obs_m = env_mujoco.reset()

        assert len(obs_i) == len(obs_m)

        _, r_i, _, info_i = env_isaac.step(np.zeros(7))
        _, r_m, _, info_m = env_mujoco.step(np.zeros(7))

        assert np.isfinite(r_i)
        assert np.isfinite(r_m)
        env_isaac.close()
        env_mujoco.close()

    def test_framework_detector_returns_info(self, fw_mod):
        """FrameworkDetector returns FrameworkInfo for each framework."""
        detector = fw_mod.FrameworkDetector(verbose=False)
        results = detector.detect_all()
        for fw_name in fw_mod.FrameworkDetector.FRAMEWORKS:
            assert fw_name in results
            info = results[fw_name]
            assert isinstance(info, fw_mod.FrameworkInfo)
            assert isinstance(info.name, str)
            assert len(info.name) > 0
            assert isinstance(info.available, bool)

    def test_mock_environment_step_increments(self, val_mod):
        """MockEnvironment step count increases with each step."""
        env = val_mod.MockEnvironment("pybullet", "tissue_retraction")
        env.reset()
        assert env.step_count == 0
        env.step(np.zeros(7))
        assert env.step_count == 1
        env.step(np.zeros(7))
        assert env.step_count == 2
        env.close()

    def test_cross_platform_validator_tolerance_defaults(self, val_mod):
        """CrossPlatformValidator has reasonable default tolerances."""
        validator = val_mod.CrossPlatformValidator()
        tol = validator.tolerance
        assert "reward_correlation" in tol
        assert "trajectory_divergence" in tol
        assert "force_similarity" in tol
        assert "success_rate_diff" in tol
        assert tol["reward_correlation"] > 0
        assert tol["trajectory_divergence"] > 0

    def test_mock_environment_get_state_structure(self, val_mod):
        """MockEnvironment.get_state() returns expected keys."""
        env = val_mod.MockEnvironment("gazebo", "instrument_handoff")
        env.reset()
        env.step(np.zeros(7))
        state = env.get_state()
        assert "joint_positions" in state
        assert "joint_velocities" in state
        assert "ee_position" in state
        assert "ee_force" in state
        env.close()

    def test_framework_info_dataclass_fields(self, fw_mod):
        """FrameworkInfo has the expected default fields."""
        info = fw_mod.FrameworkInfo(name="test", available=False)
        assert info.version == "unknown"
        assert info.gpu_support is False
        assert info.ros2_integration is False
        assert info.features == []
        assert info.warnings == []

    def test_bridge_state_converter_exists(self, bridge_mod):
        """IsaacMuJoCoBridge has a state_converter attribute."""
        bridge = bridge_mod.IsaacMuJoCoBridge()
        assert hasattr(bridge, "state_converter")

    def test_bridge_conversion_stats_tracking(self, bridge_mod):
        """IsaacMuJoCoBridge tracks conversion statistics."""
        bridge = bridge_mod.IsaacMuJoCoBridge()
        assert "states_synced" in bridge.conversion_stats

    def test_validation_report_dataclass(self, val_mod):
        """ValidationReport can be constructed with expected fields."""
        report = val_mod.ValidationReport(
            policy_path="test_policy.onnx",
            timestamp=1234567890.0,
            frameworks=["isaac", "mujoco"],
            metrics={},
            comparisons=[],
            overall_passed=True,
            recommendations=["All passed."],
        )
        assert report.overall_passed is True
        assert len(report.frameworks) == 2
