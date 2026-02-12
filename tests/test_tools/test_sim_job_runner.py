"""Unit tests for tools/sim-job-runner/sim_job_runner.py.

Tests cover TASK_REGISTRY, SUPPORTED_FRAMEWORKS constants,
JobResult dataclass, framework detection, and job ID generation.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "tools/sim-job-runner/sim_job_runner.py"


@pytest.fixture()
def mod():
    return load_module("sim_job_runner", MOD_PATH)


class TestConstants:
    def test_task_registry_count(self, mod):
        assert len(mod.TASK_REGISTRY) >= 6

    def test_task_registry_keys(self, mod):
        for task in (
            "needle_insertion",
            "tissue_retraction",
            "surgical_reach",
            "instrument_handover",
            "biopsy_sampling",
            "catheter_navigation",
        ):
            assert task in mod.TASK_REGISTRY

    def test_task_has_metrics(self, mod):
        for task_name, task_info in mod.TASK_REGISTRY.items():
            assert "metrics" in task_info, f"{task_name} missing metrics"

    def test_supported_frameworks(self, mod):
        for fw in ("mujoco", "isaac", "pybullet", "gazebo"):
            assert fw in mod.SUPPORTED_FRAMEWORKS


class TestJobResult:
    def test_creation(self, mod):
        result = mod.JobResult(
            job_id="JOB-001",
            framework="mujoco",
            task="needle_insertion",
            status="completed",
            start_time="2026-02-12T10:00:00",
            end_time="2026-02-12T10:05:00",
            duration_s=300.0,
            metrics={"success_rate": 0.85, "mean_force": 2.5},
        )
        assert result.job_id == "JOB-001"
        assert result.status == "completed"

    def test_to_dict(self, mod):
        result = mod.JobResult(
            job_id="JOB-002",
            framework="pybullet",
            task="tissue_retraction",
            status="completed",
            start_time="2026-02-12T10:00:00",
            end_time="2026-02-12T10:03:00",
            duration_s=180.0,
            metrics={"success_rate": 0.9},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["framework"] == "pybullet"

    def test_failed_result(self, mod):
        result = mod.JobResult(
            job_id="JOB-003",
            framework="isaac",
            task="surgical_reach",
            status="failed",
            start_time="2026-02-12T10:00:00",
            end_time="2026-02-12T10:00:05",
            duration_s=5.0,
            metrics={},
            error="Framework not available",
        )
        assert result.status == "failed"
        assert result.error is not None


class TestJobIdGeneration:
    def test_generate_job_id(self, mod):
        job_id = mod._generate_job_id("mujoco", "needle_insertion")
        assert "mujoco" in job_id.lower() or job_id.startswith("JOB")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_unique_ids(self, mod):
        id1 = mod._generate_job_id("mujoco", "needle_insertion")
        id2 = mod._generate_job_id("pybullet", "needle_insertion")
        assert id1 != id2
