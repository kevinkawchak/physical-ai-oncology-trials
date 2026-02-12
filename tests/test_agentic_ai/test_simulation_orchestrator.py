"""Tests for agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "autonomous_simulation_orchestrator",
        "agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py",
    )


# ============================================================
# Enum tests
# ============================================================


class TestSimFramework:
    def test_isaac_lab_value(self, mod):
        assert mod.SimFramework.ISAAC_LAB.value == "isaac_lab"

    def test_mujoco_value(self, mod):
        assert mod.SimFramework.MUJOCO.value == "mujoco"

    def test_pybullet_value(self, mod):
        assert mod.SimFramework.PYBULLET.value == "pybullet"

    def test_gazebo_value(self, mod):
        assert mod.SimFramework.GAZEBO.value == "gazebo"

    def test_member_count(self, mod):
        assert len(mod.SimFramework) == 4


class TestExperimentStatus:
    def test_values(self, mod):
        assert mod.ExperimentStatus.DESIGNED.value == "designed"
        assert mod.ExperimentStatus.QUEUED.value == "queued"
        assert mod.ExperimentStatus.RUNNING.value == "running"
        assert mod.ExperimentStatus.COMPLETED.value == "completed"
        assert mod.ExperimentStatus.FAILED.value == "failed"
        assert mod.ExperimentStatus.ANALYZING.value == "analyzing"

    def test_member_count(self, mod):
        assert len(mod.ExperimentStatus) == 6


class TestSamplingStrategy:
    def test_grid_value(self, mod):
        assert mod.SamplingStrategy.GRID.value == "grid"

    def test_latin_hypercube_value(self, mod):
        assert mod.SamplingStrategy.LATIN_HYPERCUBE.value == "latin_hypercube"

    def test_sobol_value(self, mod):
        assert mod.SamplingStrategy.SOBOL.value == "sobol"

    def test_member_count(self, mod):
        assert len(mod.SamplingStrategy) == 4


# ============================================================
# Dataclass tests
# ============================================================


class TestParameterRange:
    def test_creation(self, mod):
        pr = mod.ParameterRange(name="stiffness", min_value=1.0, max_value=50.0, num_samples=5)
        assert pr.name == "stiffness"
        assert pr.min_value == 1.0
        assert pr.max_value == 50.0

    def test_sample_values_count(self, mod):
        pr = mod.ParameterRange(name="stiffness", min_value=1.0, max_value=50.0, num_samples=5)
        vals = pr.sample_values()
        assert len(vals) == 5

    def test_sample_values_in_range(self, mod):
        pr = mod.ParameterRange(name="friction", min_value=0.1, max_value=0.9, num_samples=10)
        vals = pr.sample_values()
        for v in vals:
            assert 0.1 - 1e-9 <= v <= 0.9 + 1e-9

    def test_sample_values_uniform_endpoints(self, mod):
        pr = mod.ParameterRange(name="x", min_value=0.0, max_value=1.0, num_samples=3, distribution="uniform")
        vals = pr.sample_values()
        assert abs(vals[0] - 0.0) < 1e-9
        assert abs(vals[-1] - 1.0) < 1e-9

    def test_sample_values_log_uniform(self, mod):
        pr = mod.ParameterRange(name="x", min_value=1.0, max_value=100.0, num_samples=3, distribution="log_uniform")
        vals = pr.sample_values()
        assert len(vals) == 3
        assert vals[0] >= 1.0 - 1e-9

    def test_sample_values_normal(self, mod):
        pr = mod.ParameterRange(name="x", min_value=0.0, max_value=10.0, num_samples=5, distribution="normal")
        vals = pr.sample_values()
        assert len(vals) == 5
        for v in vals:
            assert 0.0 - 1e-9 <= v <= 10.0 + 1e-9


class TestSimulationConfig:
    def test_creation_defaults(self, mod):
        cfg = mod.SimulationConfig(config_id="CFG-001", framework=mod.SimFramework.MUJOCO)
        assert cfg.config_id == "CFG-001"
        assert cfg.num_episodes == 100
        assert cfg.max_steps_per_episode == 1000
        assert cfg.random_seed == 42

    def test_creation_custom(self, mod):
        cfg = mod.SimulationConfig(
            config_id="CFG-002",
            framework=mod.SimFramework.ISAAC_LAB,
            parameters={"stiffness": 5.0},
            num_episodes=50,
        )
        assert cfg.parameters["stiffness"] == 5.0
        assert cfg.num_episodes == 50


class TestSimulationResult:
    def test_creation_defaults(self, mod):
        r = mod.SimulationResult(config_id="CFG-001", framework=mod.SimFramework.MUJOCO)
        assert r.success_rate == 0.0
        assert r.status == mod.ExperimentStatus.COMPLETED


class TestExperimentDesign:
    def test_creation(self, mod):
        d = mod.ExperimentDesign(
            experiment_id="EXP-001",
            research_question="Test question",
            hypothesis="Test hypothesis",
        )
        assert d.experiment_id == "EXP-001"
        assert d.sampling_strategy == mod.SamplingStrategy.GRID
        assert d.significance_level == 0.05


# ============================================================
# ExperimentDesigner tests
# ============================================================


class TestExperimentDesigner:
    def test_design_experiment_returns_design(self, mod):
        designer = mod.ExperimentDesigner()
        design = designer.design_experiment("Is the policy robust to tissue variation?")
        assert isinstance(design, mod.ExperimentDesign)
        assert design.experiment_id.startswith("EXP-")

    def test_design_with_template(self, mod):
        designer = mod.ExperimentDesigner()
        design = designer.design_experiment(
            "Tissue robustness question",
            template="tissue_robustness",
        )
        assert len(design.independent_variables) >= 1
        assert design.total_configurations > 0

    def test_design_with_frameworks(self, mod):
        designer = mod.ExperimentDesigner()
        design = designer.design_experiment(
            "Cross-framework consistency",
            template="sim_to_sim_transfer",
            frameworks=["mujoco", "pybullet"],
        )
        assert len(design.frameworks) == 2

    def test_standard_parameters_available(self, mod):
        designer = mod.ExperimentDesigner()
        assert "tissue_stiffness_kpa" in designer.STANDARD_PARAMETERS
        assert "tissue_friction" in designer.STANDARD_PARAMETERS

    def test_design_total_configurations(self, mod):
        designer = mod.ExperimentDesigner()
        design = designer.design_experiment(
            "Test question",
            template="tissue_robustness",
            frameworks=["mujoco"],
        )
        # tissue_robustness has stiffness (5 samples) * friction (5 samples) * 1 framework = 25
        assert design.total_configurations == 25


# ============================================================
# SimulationRunner tests
# ============================================================


class TestSimulationRunner:
    def test_run_single_returns_result(self, mod):
        runner = mod.SimulationRunner()
        cfg = mod.SimulationConfig(
            config_id="RUN-001",
            framework=mod.SimFramework.MUJOCO,
            parameters={"tissue_stiffness_kpa": 10.0},
        )
        result = runner.run_simulation(cfg)
        assert isinstance(result, mod.SimulationResult)
        assert result.config_id == "RUN-001"
        assert result.status == mod.ExperimentStatus.COMPLETED

    def test_success_rate_bounded(self, mod):
        runner = mod.SimulationRunner()
        cfg = mod.SimulationConfig(config_id="RUN-002", framework=mod.SimFramework.PYBULLET)
        result = runner.run_simulation(cfg)
        assert 0.0 <= result.success_rate <= 1.0

    def test_batch_run(self, mod):
        runner = mod.SimulationRunner()
        results = []
        for i in range(3):
            cfg = mod.SimulationConfig(
                config_id=f"BATCH-{i:03d}",
                framework=mod.SimFramework.MUJOCO,
                parameters={"tissue_stiffness_kpa": 5.0 + i * 10},
            )
            results.append(runner.run_simulation(cfg))
        assert len(results) == 3
        assert all(r.status == mod.ExperimentStatus.COMPLETED for r in results)


# ============================================================
# AnalysisEngine tests
# ============================================================


class TestAnalysisEngine:
    def test_analyze_returns_report(self, mod):
        engine = mod.AnalysisEngine()
        design = mod.ExperimentDesign(
            experiment_id="EXP-TEST",
            research_question="Test",
            hypothesis="Policy maintains >90% success rate across tissue property variations",
            independent_variables=[
                mod.ParameterRange("tissue_stiffness_kpa", 1.0, 50.0, 3),
            ],
        )
        results = [
            mod.SimulationResult(
                config_id=f"R-{i}",
                framework=mod.SimFramework.MUJOCO,
                parameters={"tissue_stiffness_kpa": float(i * 10 + 5)},
                success_rate=0.92 - i * 0.01,
                status=mod.ExperimentStatus.COMPLETED,
            )
            for i in range(3)
        ]
        report = engine.analyze_results(design, results)
        assert isinstance(report, mod.ExperimentReport)
        assert report.total_runs == 3
        assert report.successful_runs == 3

    def test_analyze_all_failed(self, mod):
        engine = mod.AnalysisEngine()
        design = mod.ExperimentDesign(
            experiment_id="EXP-FAIL",
            research_question="Test",
            hypothesis="Test",
        )
        results = [
            mod.SimulationResult(
                config_id="R-0",
                framework=mod.SimFramework.MUJOCO,
                status=mod.ExperimentStatus.FAILED,
            ),
        ]
        report = engine.analyze_results(design, results)
        assert report.failed_runs == 1
        assert report.successful_runs == 0

    def test_sensitivity_analysis(self, mod):
        engine = mod.AnalysisEngine()
        param = mod.ParameterRange("stiffness", 1.0, 50.0, 3)
        design = mod.ExperimentDesign(
            experiment_id="EXP-SENS",
            research_question="Sensitivity",
            hypothesis="Test",
            independent_variables=[param],
        )
        results = [
            mod.SimulationResult(
                config_id=f"S-{i}",
                framework=mod.SimFramework.MUJOCO,
                parameters={"stiffness": float(v)},
                success_rate=0.95 - i * 0.05,
                status=mod.ExperimentStatus.COMPLETED,
            )
            for i, v in enumerate([1.0, 25.0, 50.0])
        ]
        report = engine.analyze_results(design, results)
        assert "stiffness" in report.parameter_sensitivity


# ============================================================
# SimulationOrchestrator tests
# ============================================================


class TestSimulationOrchestrator:
    def test_init(self, mod):
        orch = mod.SimulationOrchestrator(autonomy="full")
        assert orch.autonomy == "full"

    def test_run_campaign_returns_report(self, mod):
        orch = mod.SimulationOrchestrator()
        report = orch.run_experiment_campaign(
            research_question="Is the policy robust?",
            template="tissue_robustness",
            frameworks=["mujoco"],
            max_iterations=1,
        )
        assert isinstance(report, mod.ExperimentReport)
        assert report.total_runs > 0

    def test_experiment_history_populated(self, mod):
        orch = mod.SimulationOrchestrator()
        orch.run_experiment_campaign(
            research_question="Test campaign",
            frameworks=["mujoco"],
            max_iterations=1,
        )
        history = orch.get_experiment_history()
        assert len(history) == 1

    def test_reasoning_log_populated(self, mod):
        orch = mod.SimulationOrchestrator()
        orch.run_experiment_campaign(
            research_question="Test reasoning",
            frameworks=["mujoco"],
            max_iterations=1,
        )
        log = orch.get_reasoning_log()
        assert len(log) > 0
        assert "phase" in log[0]
        assert "reasoning" in log[0]
