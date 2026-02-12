"""Tests for tools/sim-job-runner/sim_job_runner.py

Covers JobResult dataclass, framework detection, and job ID generation.
"""

from __future__ import annotations


class TestJobResult:
    def test_creation_defaults(self, sim_runner_mod):
        result = sim_runner_mod.JobResult(
            job_id="JOB-001",
            framework="mujoco",
            task="needle_insertion",
        )
        assert result.job_id == "JOB-001"
        assert result.status == "pending"
        assert result.metrics == {}

    def test_to_dict(self, sim_runner_mod):
        result = sim_runner_mod.JobResult(
            job_id="JOB-002",
            framework="isaac_lab",
            task="tissue_retraction",
            status="completed",
            duration_s=12.5,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["framework"] == "isaac_lab"
        assert d["duration_s"] == 12.5


class TestFrameworkDetection:
    def test_detect_available_frameworks(self, sim_runner_mod):
        frameworks = sim_runner_mod._detect_available_frameworks()
        assert isinstance(frameworks, list)

    def test_generate_job_id(self, sim_runner_mod):
        job_id = sim_runner_mod._generate_job_id("mujoco", "needle_insertion")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_generate_unique_ids(self, sim_runner_mod):
        id1 = sim_runner_mod._generate_job_id("mujoco", "task_a")
        id2 = sim_runner_mod._generate_job_id("mujoco", "task_b")
        assert id1 != id2


class TestRunSimulation:
    def test_run_returns_result(self, sim_runner_mod):
        result = sim_runner_mod._run_simulation(
            framework="mujoco",
            task="needle_insertion",
            params={"max_steps": 100},
        )
        assert isinstance(result, sim_runner_mod.JobResult)
        assert result.framework == "mujoco"
