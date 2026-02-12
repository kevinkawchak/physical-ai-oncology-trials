"""Tests for Validation Benchmark Suite Runner.

Validates BenchmarkConfig, BenchmarkResult, BenchmarkReport (including
add_result and to_dict), SCENARIOS definitions, PhysicsBenchmark,
PerformanceBenchmark, CrossFrameworkBenchmark, and BenchmarkRunner.

LICENSE: MIT
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "q1-2026-standards/objective-3-validation-benchmark/benchmark_runner.py"


@pytest.fixture()
def bm_mod():
    return load_module("benchmark_runner", MOD_PATH)


@pytest.fixture()
def default_config(bm_mod):
    return bm_mod.BenchmarkConfig(scenario="peg_transfer", frameworks=["mujoco"])


# ===========================================================================
# TestBenchmarkConfig
# ===========================================================================


class TestBenchmarkConfig:
    def test_defaults(self, bm_mod):
        cfg = bm_mod.BenchmarkConfig(scenario="reach", frameworks=["mujoco"])
        assert cfg.num_episodes == 100
        assert cfg.max_steps == 1000
        assert cfg.seed == 42
        assert cfg.verbose is False

    def test_custom_values(self, bm_mod):
        cfg = bm_mod.BenchmarkConfig(
            scenario="suturing",
            frameworks=["isaac", "mujoco"],
            num_episodes=50,
            seed=0,
        )
        assert cfg.num_episodes == 50
        assert cfg.seed == 0


# ===========================================================================
# TestBenchmarkResult
# ===========================================================================


class TestBenchmarkResult:
    def test_construction(self, bm_mod):
        r = bm_mod.BenchmarkResult(
            name="free_fall",
            category="physics",
            passed=True,
            value=0.005,
            target=0.01,
            unit="m",
        )
        assert r.passed is True
        assert r.details == {}

    def test_with_details(self, bm_mod):
        r = bm_mod.BenchmarkResult(
            name="throughput",
            category="performance",
            passed=False,
            value=50000,
            target=100000,
            unit="steps/sec",
            details={"error": "Too slow"},
        )
        assert r.details["error"] == "Too slow"


# ===========================================================================
# TestBenchmarkReport
# ===========================================================================


class TestBenchmarkReport:
    def test_default_values(self, bm_mod):
        report = bm_mod.BenchmarkReport(scenario="reach", frameworks=["mujoco"])
        assert report.overall_passed is True
        assert report.results == []
        assert report.duration_seconds == 0.0

    def test_add_result_passing(self, bm_mod):
        report = bm_mod.BenchmarkReport(scenario="reach", frameworks=["mujoco"])
        result = bm_mod.BenchmarkResult("test", "physics", True, 0.001, 0.01, "m")
        report.add_result(result)
        assert len(report.results) == 1
        assert report.overall_passed is True

    def test_add_result_failing(self, bm_mod):
        report = bm_mod.BenchmarkReport(scenario="reach", frameworks=["mujoco"])
        result = bm_mod.BenchmarkResult("test", "physics", False, 0.05, 0.01, "m")
        report.add_result(result)
        assert report.overall_passed is False

    def test_to_dict(self, bm_mod):
        report = bm_mod.BenchmarkReport(scenario="reach", frameworks=["mujoco"])
        report.add_result(bm_mod.BenchmarkResult("t1", "physics", True, 0.001, 0.01, "m"))
        d = report.to_dict()
        assert d["scenario"] == "reach"
        assert len(d["results"]) == 1
        assert "timestamp" in d

    def test_timestamp_auto_set(self, bm_mod):
        report = bm_mod.BenchmarkReport(scenario="reach", frameworks=["mujoco"])
        # Timestamp should be an ISO date string from current time
        ts = datetime.fromisoformat(report.timestamp)
        assert ts.year >= 2026


# ===========================================================================
# TestScenarios
# ===========================================================================


class TestScenarios:
    def test_scenarios_defined(self, bm_mod):
        assert len(bm_mod.SCENARIOS) >= 8

    def test_scenario_keys(self, bm_mod):
        required = {"peg_transfer", "needle_insertion", "free_fall", "pendulum", "reach", "push"}
        assert required.issubset(set(bm_mod.SCENARIOS.keys()))

    def test_scenario_structure(self, bm_mod):
        for name, info in bm_mod.SCENARIOS.items():
            assert "description" in info
            assert "category" in info
            assert "difficulty" in info
            assert "metrics" in info


# ===========================================================================
# TestPhysicsBenchmark
# ===========================================================================


class TestPhysicsBenchmark:
    def test_free_fall_without_mujoco(self, bm_mod, default_config):
        pb = bm_mod.PhysicsBenchmark(default_config)
        result = pb.run_free_fall("nonexistent.xml")
        # If MuJoCo not available, should return a result with passed=False
        assert isinstance(result, bm_mod.BenchmarkResult)
        assert result.name == "free_fall"
        assert result.category == "physics"

    def test_pendulum_without_mujoco(self, bm_mod, default_config):
        pb = bm_mod.PhysicsBenchmark(default_config)
        result = pb.run_pendulum("nonexistent.xml")
        assert isinstance(result, bm_mod.BenchmarkResult)
        assert result.name == "pendulum"


# ===========================================================================
# TestPerformanceBenchmark
# ===========================================================================


class TestPerformanceBenchmark:
    def test_throughput_without_mujoco(self, bm_mod, default_config):
        perf = bm_mod.PerformanceBenchmark(default_config)
        result = perf.run_throughput("nonexistent.xml")
        assert isinstance(result, bm_mod.BenchmarkResult)
        assert result.name == "throughput"
        assert result.unit == "steps/sec"


# ===========================================================================
# TestCrossFrameworkBenchmark
# ===========================================================================


class TestCrossFrameworkBenchmark:
    def test_trajectory_match_without_mujoco(self, bm_mod, default_config):
        cfb = bm_mod.CrossFrameworkBenchmark(default_config)
        result = cfb.run_trajectory_match("a.xml", "b.xml")
        assert isinstance(result, bm_mod.BenchmarkResult)
        assert result.name == "trajectory_match"
        assert result.category == "cross_framework"


# ===========================================================================
# TestBenchmarkRunner
# ===========================================================================


class TestBenchmarkRunner:
    def test_default_construction(self, bm_mod):
        runner = bm_mod.BenchmarkRunner()
        assert runner.config.scenario == "peg_transfer"
        assert runner.config.frameworks == ["mujoco"]

    def test_custom_construction(self, bm_mod, default_config):
        runner = bm_mod.BenchmarkRunner(default_config)
        assert runner.config.scenario == "peg_transfer"

    def test_run_unknown_scenario(self, bm_mod, default_config):
        runner = bm_mod.BenchmarkRunner(default_config)
        report = runner.run("nonexistent_scenario")
        assert report.overall_passed is False

    def test_run_returns_report(self, bm_mod, default_config):
        runner = bm_mod.BenchmarkRunner(default_config)
        report = runner.run("peg_transfer")
        assert isinstance(report, bm_mod.BenchmarkReport)
        assert report.scenario == "peg_transfer"
        assert report.duration_seconds >= 0

    def test_run_suite(self, bm_mod, default_config):
        runner = bm_mod.BenchmarkRunner(default_config)
        reports = runner.run_suite(scenarios=["peg_transfer", "reach"])
        assert len(reports) == 2
        assert reports[0].scenario == "peg_transfer"
        assert reports[1].scenario == "reach"

    def test_run_suite_all_scenarios(self, bm_mod, default_config):
        runner = bm_mod.BenchmarkRunner(default_config)
        reports = runner.run_suite()
        assert len(reports) == len(bm_mod.SCENARIOS)
