"""Tests for agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py

Covers experiment design, simulation runner, analysis engine,
and the orchestrator end-to-end.
"""

from __future__ import annotations


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_sim_frameworks(self, orchestrator_mod):
        sf = orchestrator_mod.SimFramework
        assert sf.ISAAC_LAB.value == "isaac_lab"
        assert sf.MUJOCO.value == "mujoco"
        assert sf.PYBULLET.value == "pybullet"
        assert sf.GAZEBO.value == "gazebo"

    def test_experiment_status(self, orchestrator_mod):
        es = orchestrator_mod.ExperimentStatus
        assert es.DESIGNED.value == "designed"
        assert es.COMPLETED.value == "completed"
        assert es.FAILED.value == "failed"

    def test_sampling_strategies(self, orchestrator_mod):
        ss = orchestrator_mod.SamplingStrategy
        assert ss.GRID.value == "grid"
        assert ss.LATIN_HYPERCUBE.value == "latin_hypercube"


# ── ParameterRange dataclass ─────────────────────────────────────────────


class TestParameterRange:
    def test_creation(self, orchestrator_mod):
        pr = orchestrator_mod.ParameterRange(
            name="force_limit",
            min_value=1.0,
            max_value=10.0,
            num_samples=5,
        )
        assert pr.name == "force_limit"

    def test_sample_values(self, orchestrator_mod):
        pr = orchestrator_mod.ParameterRange(
            name="velocity",
            min_value=0.01,
            max_value=0.25,
            num_samples=5,
        )
        values = pr.sample_values()
        assert len(values) == 5
        assert all(0.01 <= v <= 0.25 for v in values)


# ── SimulationConfig dataclass ───────────────────────────────────────────


class TestSimulationConfig:
    def test_creation(self, orchestrator_mod):
        cfg = orchestrator_mod.SimulationConfig(
            config_id="CFG-001",
            framework=orchestrator_mod.SimFramework.MUJOCO,
        )
        assert cfg.config_id == "CFG-001"
        assert cfg.num_episodes == 100
        assert cfg.random_seed == 42


# ── SimulationRunner ─────────────────────────────────────────────────────


class TestSimulationRunner:
    def test_creation(self, orchestrator_mod):
        runner = orchestrator_mod.SimulationRunner()
        assert runner is not None

    def test_run_simulation(self, orchestrator_mod):
        runner = orchestrator_mod.SimulationRunner()
        cfg = orchestrator_mod.SimulationConfig(
            config_id="RUN-001",
            framework=orchestrator_mod.SimFramework.MUJOCO,
            num_episodes=10,
        )
        result = runner.run_simulation(cfg)
        assert isinstance(result, orchestrator_mod.SimulationResult)
        assert result.config_id == "RUN-001"


# ── ExperimentDesigner ───────────────────────────────────────────────────


class TestExperimentDesigner:
    def test_design_from_template(self, orchestrator_mod):
        designer = orchestrator_mod.ExperimentDesigner()
        design = designer.design_experiment(
            research_question="What is the effect of force limits?",
            template="needle_insertion",
        )
        assert isinstance(design, orchestrator_mod.ExperimentDesign)
        assert design.research_question is not None


# ── SimulationOrchestrator ───────────────────────────────────────────────


class TestSimulationOrchestrator:
    def test_creation(self, orchestrator_mod):
        orch = orchestrator_mod.SimulationOrchestrator()
        assert orch is not None

    def test_run_campaign(self, orchestrator_mod):
        orch = orchestrator_mod.SimulationOrchestrator()
        report = orch.run_experiment_campaign(
            research_question="How does velocity affect safety?",
            template="needle_insertion",
            max_iterations=1,
        )
        assert isinstance(report, orchestrator_mod.ExperimentReport)

    def test_reasoning_log(self, orchestrator_mod):
        orch = orchestrator_mod.SimulationOrchestrator()
        orch.run_experiment_campaign(
            research_question="Test question",
            template="needle_insertion",
            max_iterations=1,
        )
        log = orch.get_reasoning_log()
        assert isinstance(log, list)
        assert len(log) > 0
