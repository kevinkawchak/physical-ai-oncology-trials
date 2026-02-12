"""Unit tests for agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py.

Tests cover SimFramework, ExperimentStatus, SamplingStrategy enums,
ParameterRange, SimulationConfig, SimulationResult, ExperimentDesign dataclasses.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py"


@pytest.fixture()
def mod():
    return load_module("simulation_orchestrator", MOD_PATH)


class TestEnums:
    def test_sim_framework(self, mod):
        names = [e.name for e in mod.SimFramework]
        for f in ("ISAAC_LAB", "MUJOCO", "PYBULLET", "GAZEBO"):
            assert f in names

    def test_experiment_status(self, mod):
        names = [e.name for e in mod.ExperimentStatus]
        for s in ("DESIGNED", "QUEUED", "RUNNING", "COMPLETED", "FAILED"):
            assert s in names

    def test_sampling_strategy(self, mod):
        names = [e.name for e in mod.SamplingStrategy]
        for s in ("GRID", "RANDOM", "LATIN_HYPERCUBE", "SOBOL"):
            assert s in names


class TestParameterRange:
    def test_creation(self, mod):
        pr = mod.ParameterRange(
            name="friction",
            min_value=0.1,
            max_value=1.0,
            num_samples=5,
        )
        assert pr.name == "friction"
        assert pr.min_value < pr.max_value

    def test_sample_values(self, mod):
        pr = mod.ParameterRange(
            name="stiffness",
            min_value=100.0,
            max_value=10000.0,
            num_samples=10,
        )
        samples = pr.sample_values(10)
        assert len(samples) == 10
        assert all(pr.min_value <= s <= pr.max_value for s in samples)


class TestSimulationConfig:
    def test_creation(self, mod):
        config = mod.SimulationConfig(
            config_id="CFG-001",
            framework=mod.SimFramework.MUJOCO,
            policy_path="/models/policy.onnx",
            environment="needle_insertion",
            parameters={"friction": 0.5},
            num_episodes=10,
            max_steps_per_episode=1000,
            random_seed=42,
        )
        assert config.framework == mod.SimFramework.MUJOCO
        assert config.num_episodes == 10


class TestSimulationResult:
    def test_creation(self, mod):
        result = mod.SimulationResult(
            config_id="CFG-001",
            framework=mod.SimFramework.MUJOCO,
            parameters={"friction": 0.5},
            success_rate=0.85,
            mean_reward=150.0,
            std_reward=25.0,
            mean_episode_length=450,
        )
        assert result.success_rate == 0.85
        assert result.mean_reward == 150.0

    def test_result_status_default(self, mod):
        result = mod.SimulationResult(
            config_id="CFG-002",
            framework=mod.SimFramework.PYBULLET,
            parameters={},
            success_rate=0.0,
            mean_reward=0.0,
            std_reward=0.0,
            mean_episode_length=0,
        )
        assert result.status == mod.ExperimentStatus.COMPLETED


class TestExperimentDesign:
    def test_creation(self, mod):
        design = mod.ExperimentDesign(
            experiment_id="EXP-001",
            research_question="How does friction affect needle insertion accuracy?",
            hypothesis="Higher friction increases force but improves accuracy",
            independent_variables=[mod.ParameterRange(name="friction", min_value=0.1, max_value=1.0, num_samples=5)],
            dependent_variables=["success_rate", "mean_force"],
            control_variables={"velocity": 0.01},
            frameworks=[mod.SimFramework.MUJOCO],
            sampling_strategy=mod.SamplingStrategy.LATIN_HYPERCUBE,
            total_configurations=5,
        )
        assert design.experiment_id == "EXP-001"
        assert len(design.independent_variables) == 1
