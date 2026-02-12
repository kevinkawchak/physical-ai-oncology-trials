"""Unit tests for q1-2026-standards/objective-3-validation-benchmark/benchmark_runner.py.

Tests cover BenchmarkConfig, BenchmarkResult, BenchmarkReport dataclasses,
SCENARIOS constant, and PhysicsBenchmark.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "q1-2026-standards/objective-3-validation-benchmark/benchmark_runner.py"


@pytest.fixture()
def mod():
    return load_module("benchmark_runner", MOD_PATH)


class TestConstants:
    def test_scenarios_exist(self, mod):
        assert len(mod.SCENARIOS) >= 6

    def test_scenario_keys(self, mod):
        for key in ("peg_transfer", "needle_insertion", "free_fall", "pendulum"):
            assert key in mod.SCENARIOS


class TestBenchmarkConfig:
    def test_creation(self, mod):
        config = mod.BenchmarkConfig(
            scenario="needle_insertion",
            frameworks=["mujoco", "pybullet"],
            num_episodes=10,
            max_steps=1000,
            seed=42,
        )
        assert config.scenario == "needle_insertion"
        assert len(config.frameworks) == 2


class TestBenchmarkResult:
    def test_passed(self, mod):
        result = mod.BenchmarkResult(
            name="free_fall_accuracy",
            category="physics",
            passed=True,
            value=9.79,
            target=9.81,
            unit="m/s^2",
        )
        assert result.passed is True
        assert abs(result.value - result.target) < 0.1

    def test_failed(self, mod):
        result = mod.BenchmarkResult(
            name="trajectory_consistency",
            category="policy",
            passed=False,
            value=0.3,
            target=0.9,
            unit="correlation",
        )
        assert result.passed is False


class TestBenchmarkReport:
    def test_creation(self, mod):
        report = mod.BenchmarkReport(
            scenario="free_fall",
            frameworks=["mujoco"],
        )
        assert report.scenario == "free_fall"
        assert report.overall_passed is True

    def test_add_result(self, mod):
        report = mod.BenchmarkReport(
            scenario="pendulum",
            frameworks=["mujoco", "pybullet"],
        )
        report.add_result(
            mod.BenchmarkResult(
                name="period_accuracy",
                category="physics",
                passed=True,
                value=2.01,
                target=2.0,
                unit="seconds",
            )
        )
        assert len(report.results) == 1

    def test_to_dict(self, mod):
        report = mod.BenchmarkReport(
            scenario="peg_transfer",
            frameworks=["mujoco"],
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["scenario"] == "peg_transfer"
