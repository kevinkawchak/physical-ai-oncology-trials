from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("surgical_robot_training", "examples/01_surgical_robot_training.py")


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for SurgicalTask enum."""

    def test_needle_insertion(self, mod):
        assert mod.SurgicalTask.NEEDLE_INSERTION.value == "needle_insertion"

    def test_tissue_grasping(self, mod):
        assert mod.SurgicalTask.TISSUE_GRASPING.value == "tissue_grasping"

    def test_needle_lift(self, mod):
        assert mod.SurgicalTask.NEEDLE_LIFT.value == "needle_lift"

    def test_needle_handover(self, mod):
        assert mod.SurgicalTask.NEEDLE_HANDOVER.value == "needle_handover"

    def test_shunt_insertion(self, mod):
        assert mod.SurgicalTask.SHUNT_INSERTION.value == "shunt_insertion"


# ---------------------------------------------------------------------------
# TestTrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self, mod):
        cfg = mod.TrainingConfig()
        assert cfg.task == mod.SurgicalTask.NEEDLE_LIFT
        assert cfg.num_envs == 1024
        assert cfg.max_iterations == 1000

    def test_learning_rate_default(self, mod):
        cfg = mod.TrainingConfig()
        assert cfg.learning_rate == 3e-4

    def test_reward_weights_present(self, mod):
        cfg = mod.TrainingConfig()
        assert "position_tracking" in cfg.reward_weights
        assert "collision_penalty" in cfg.reward_weights

    def test_domain_randomization_present(self, mod):
        cfg = mod.TrainingConfig()
        assert "tissue_stiffness_range" in cfg.domain_randomization


# ---------------------------------------------------------------------------
# TestEnvironmentConfig
# ---------------------------------------------------------------------------


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_defaults(self, mod):
        cfg = mod.EnvironmentConfig()
        assert cfg.robot_model == "dvrk_psm"
        assert cfg.control_mode == "cartesian"

    def test_anatomy_present(self, mod):
        cfg = mod.EnvironmentConfig()
        assert "include_tumor" in cfg.anatomy
        assert cfg.anatomy["include_tumor"] is True

    def test_physics_dt(self, mod):
        cfg = mod.EnvironmentConfig()
        assert cfg.physics_dt > 0


# ---------------------------------------------------------------------------
# TestOncologySurgicalEnv
# ---------------------------------------------------------------------------


class TestOncologySurgicalEnv:
    """Tests for OncologySurgicalEnv class."""

    def test_init(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        assert env.num_envs == 4

    def test_reset_returns_array(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == 4

    def test_step_returns_tuple(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        obs = env.reset()
        actions = np.zeros((4, env.action_space["shape"][0]), dtype=np.float32)
        result = env.step(actions)
        assert len(result) == 4  # obs, rewards, dones, infos

    def test_observation_space(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        obs_space = env.observation_space
        assert "shape" in obs_space
        assert obs_space["shape"][0] > 0

    def test_action_space(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        act_space = env.action_space
        assert "shape" in act_space
        assert act_space["shape"][0] == 8  # 7 + 1 gripper


# ---------------------------------------------------------------------------
# TestSurgicalPolicyNetwork
# ---------------------------------------------------------------------------


class TestSurgicalPolicyNetwork:
    """Tests for SurgicalPolicyNetwork class."""

    def test_init(self, mod):
        net = mod.SurgicalPolicyNetwork(obs_dim=35, action_dim=8)
        assert net.obs_dim == 35
        assert net.action_dim == 8

    def test_forward_produces_output(self, mod):
        net = mod.SurgicalPolicyNetwork(obs_dim=10, action_dim=4)
        obs = np.random.randn(8, 10).astype(np.float32)
        actions, values, log_probs = net.forward(obs)
        assert actions.shape == (8, 4)
        assert values.shape == (8,)

    def test_get_action(self, mod):
        net = mod.SurgicalPolicyNetwork(obs_dim=10, action_dim=4)
        obs = np.random.randn(10).astype(np.float32)
        action = net.get_action(obs)
        assert action.shape[1] == 4


# ---------------------------------------------------------------------------
# TestPolicyEvaluator
# ---------------------------------------------------------------------------


class TestPolicyEvaluator:
    """Tests for PolicyEvaluator class."""

    def test_init(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        net = mod.SurgicalPolicyNetwork(
            obs_dim=env.observation_space["shape"][0], action_dim=env.action_space["shape"][0]
        )
        evaluator = mod.PolicyEvaluator(env, net)
        assert evaluator.env is env

    def test_evaluate_returns_dict(self, mod):
        cfg = mod.EnvironmentConfig()
        env = mod.OncologySurgicalEnv(task=mod.SurgicalTask.NEEDLE_INSERTION, config=cfg, num_envs=4)
        net = mod.SurgicalPolicyNetwork(
            obs_dim=env.observation_space["shape"][0], action_dim=env.action_space["shape"][0]
        )
        evaluator = mod.PolicyEvaluator(env, net)
        results = evaluator.evaluate(n_episodes=2)
        assert isinstance(results, dict)
        assert "success_rate" in results
        assert "n_episodes" in results

    def test_thresholds_exist(self, mod):
        assert hasattr(mod.PolicyEvaluator, "THRESHOLDS")
        assert "success_rate" in mod.PolicyEvaluator.THRESHOLDS
