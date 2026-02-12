"""Tests for examples/01_surgical_robot_training.py

Covers SurgicalTask enum, environment config, and training pipeline.
"""

from __future__ import annotations

import numpy as np


class TestEnums:
    def test_surgical_tasks(self, surgical_training_mod):
        st = surgical_training_mod.SurgicalTask
        assert st.NEEDLE_INSERTION.value == "needle_insertion"
        assert st.TISSUE_GRASPING.value == "tissue_grasping"


class TestTrainingConfig:
    def test_creation(self, surgical_training_mod):
        cfg = surgical_training_mod.TrainingConfig()
        assert cfg.max_iterations > 0
        assert cfg.learning_rate > 0


class TestOncologySurgicalEnv:
    def test_creation(self, surgical_training_mod):
        config = surgical_training_mod.EnvironmentConfig()
        env = surgical_training_mod.OncologySurgicalEnv(
            task=surgical_training_mod.SurgicalTask.NEEDLE_INSERTION,
            config=config,
            num_envs=4,
        )
        assert env is not None

    def test_reset(self, surgical_training_mod):
        config = surgical_training_mod.EnvironmentConfig()
        env = surgical_training_mod.OncologySurgicalEnv(
            task=surgical_training_mod.SurgicalTask.NEEDLE_INSERTION,
            config=config,
            num_envs=4,
        )
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_step(self, surgical_training_mod):
        config = surgical_training_mod.EnvironmentConfig()
        env = surgical_training_mod.OncologySurgicalEnv(
            task=surgical_training_mod.SurgicalTask.NEEDLE_INSERTION,
            config=config,
            num_envs=4,
        )
        env.reset()
        action_dim = env.action_space["shape"][0]
        action = np.zeros((4, action_dim))
        obs, rewards, dones, infos = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)


class TestPolicyEvaluator:
    def test_creation(self, surgical_training_mod):
        config = surgical_training_mod.EnvironmentConfig()
        env = surgical_training_mod.OncologySurgicalEnv(
            task=surgical_training_mod.SurgicalTask.NEEDLE_INSERTION,
            config=config,
            num_envs=4,
        )
        policy = surgical_training_mod.SurgicalPolicyNetwork(
            obs_dim=env.observation_space["shape"][0],
            action_dim=env.action_space["shape"][0],
        )
        evaluator = surgical_training_mod.PolicyEvaluator(env=env, policy=policy)
        assert evaluator is not None
