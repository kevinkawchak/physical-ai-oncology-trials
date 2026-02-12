"""Tests for Cross-Platform Validation Suite.

Validates ValidationMetrics, MockEnvironment, PolicyLoader,
CrossPlatformValidator tolerance system, and report generation.

LICENSE: MIT
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "unification/cross_platform_tools/validation_suite.py"


@pytest.fixture()
def vs_mod():
    return load_module("validation_suite", MOD_PATH)


@pytest.fixture()
def validator(vs_mod):
    return vs_mod.CrossPlatformValidator()


# ===========================================================================
# TestValidationMetrics
# ===========================================================================


class TestValidationMetrics:
    def test_construction(self, vs_mod):
        m = vs_mod.ValidationMetrics(
            framework="mujoco",
            task="needle_insertion",
            episodes=100,
            success_rate=0.85,
            mean_reward=10.5,
            std_reward=2.1,
            mean_episode_length=250.0,
            force_violations=3,
            position_accuracy=0.002,
            execution_time=45.0,
        )
        assert m.framework == "mujoco"
        assert m.success_rate == 0.85
        assert m.metadata == {}


# ===========================================================================
# TestCrossFrameworkComparison
# ===========================================================================


class TestCrossFrameworkComparison:
    def test_construction(self, vs_mod):
        c = vs_mod.CrossFrameworkComparison(
            reference_framework="isaac",
            comparison_framework="mujoco",
            reward_correlation=0.95,
            trajectory_divergence=0.02,
            force_profile_similarity=0.90,
            success_rate_difference=0.05,
            acceptable=True,
        )
        assert c.acceptable is True
        assert c.notes == []

    def test_with_notes(self, vs_mod):
        c = vs_mod.CrossFrameworkComparison(
            reference_framework="isaac",
            comparison_framework="pybullet",
            reward_correlation=0.80,
            trajectory_divergence=0.10,
            force_profile_similarity=0.70,
            success_rate_difference=0.15,
            acceptable=False,
            notes=["High trajectory divergence"],
        )
        assert c.acceptable is False
        assert len(c.notes) == 1


# ===========================================================================
# TestMockEnvironment
# ===========================================================================


class TestMockEnvironment:
    def test_reset(self, vs_mod):
        env = vs_mod.MockEnvironment("mujoco", "needle_insertion")
        obs = env.reset()
        assert obs.shape == (23,)
        np.testing.assert_array_equal(obs, np.zeros(23))

    def test_step(self, vs_mod):
        env = vs_mod.MockEnvironment("mujoco", "needle_insertion")
        env.reset()
        action = np.zeros(7)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (23,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "force" in info

    def test_get_state(self, vs_mod):
        env = vs_mod.MockEnvironment("isaac", "tissue_retraction")
        state = env.get_state()
        assert "joint_positions" in state
        assert "ee_position" in state
        assert state["joint_positions"].shape == (7,)

    def test_close(self, vs_mod):
        env = vs_mod.MockEnvironment("pybullet", "instrument_handoff")
        env.close()  # Should not raise

    def test_framework_specific_noise(self, vs_mod):
        env_isaac = vs_mod.MockEnvironment("isaac", "task")
        env_mujoco = vs_mod.MockEnvironment("mujoco", "task")
        assert env_isaac.noise_scale != env_mujoco.noise_scale

    def test_step_count(self, vs_mod):
        env = vs_mod.MockEnvironment("mujoco", "task")
        env.reset()
        assert env.step_count == 0
        env.step(np.zeros(7))
        assert env.step_count == 1


# ===========================================================================
# TestPolicyLoader
# ===========================================================================


class TestPolicyLoader:
    def test_unsupported_format_raises(self, vs_mod):
        with pytest.raises(ValueError, match="Unsupported"):
            vs_mod.PolicyLoader("model.bad_extension")

    def test_predict_returns_array(self, vs_mod):
        # ONNX not installed, should use mock
        try:
            loader = vs_mod.PolicyLoader("dummy.onnx")
        except Exception:
            pytest.skip("PolicyLoader requires onnxruntime or mock fallback")
            return
        action = loader.predict(np.zeros(23))
        assert isinstance(action, np.ndarray)


# ===========================================================================
# TestCrossPlatformValidator
# ===========================================================================


class TestCrossPlatformValidator:
    def test_default_tolerance(self, validator, vs_mod):
        assert validator.tolerance["reward_correlation"] == 0.9
        assert validator.tolerance["trajectory_divergence"] == 0.05
        assert validator.tolerance["force_similarity"] == 0.85
        assert validator.tolerance["success_rate_diff"] == 0.1

    def test_custom_tolerance(self, vs_mod):
        v = vs_mod.CrossPlatformValidator(tolerance={"reward_correlation": 0.8})
        assert v.tolerance["reward_correlation"] == 0.8

    def test_tasks_defined(self, vs_mod):
        tasks = vs_mod.CrossPlatformValidator.TASKS
        assert "needle_insertion" in tasks
        assert len(tasks) >= 3

    def test_create_environment(self, validator, vs_mod):
        env = validator._create_environment("mujoco", "needle_insertion")
        assert isinstance(env, vs_mod.MockEnvironment)

    def test_reward_correlation_computation(self, validator, vs_mod):
        ref = vs_mod.ValidationMetrics(
            framework="a",
            task="t",
            episodes=10,
            success_rate=0.8,
            mean_reward=10.0,
            std_reward=1.0,
            mean_episode_length=100,
            force_violations=0,
            position_accuracy=0.001,
            execution_time=1.0,
        )
        comp = vs_mod.ValidationMetrics(
            framework="b",
            task="t",
            episodes=10,
            success_rate=0.8,
            mean_reward=10.0,
            std_reward=1.0,
            mean_episode_length=100,
            force_violations=0,
            position_accuracy=0.001,
            execution_time=1.0,
        )
        corr = validator._compute_reward_correlation(ref, comp)
        assert corr == 1.0

    def test_trajectory_divergence_empty(self, validator):
        div = validator._compute_trajectory_divergence([], [])
        assert div == 0.0

    def test_force_similarity_empty(self, validator):
        sim = validator._compute_force_similarity([], [])
        assert sim == 1.0


# ===========================================================================
# TestValidationReport
# ===========================================================================


class TestValidationReport:
    def test_report_construction(self, vs_mod):
        report = vs_mod.ValidationReport(
            policy_path="test.onnx",
            timestamp=time.time(),
            frameworks=["mujoco", "pybullet"],
            metrics={},
            comparisons=[],
            overall_passed=True,
            recommendations=["All good"],
        )
        assert report.overall_passed is True
        assert len(report.recommendations) == 1


# ===========================================================================
# TestRecommendationGeneration
# ===========================================================================


class TestRecommendationGeneration:
    def test_all_passed_recommendation(self, validator, vs_mod):
        recs = validator._generate_recommendations({}, [])
        assert any("passed" in r.lower() for r in recs)

    def test_failed_comparison_recommendation(self, validator, vs_mod):
        comp = vs_mod.CrossFrameworkComparison(
            reference_framework="isaac",
            comparison_framework="pybullet",
            reward_correlation=0.5,
            trajectory_divergence=0.1,
            force_profile_similarity=0.5,
            success_rate_difference=0.2,
            acceptable=False,
            notes=["Low reward correlation: 0.50"],
        )
        recs = validator._generate_recommendations({}, [comp])
        assert any("failed" in r.lower() for r in recs)
