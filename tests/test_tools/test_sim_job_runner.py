from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("sim_job_runner", "tools/sim-job-runner/sim_job_runner.py")


# ---------------------------------------------------------------------------
# TestJobResult
# ---------------------------------------------------------------------------


class TestJobResult:
    """Tests for the JobResult dataclass."""

    def test_creation_required_fields(self, mod):
        result = mod.JobResult(job_id="job_001", framework="mujoco", task="needle_insertion")
        assert result.job_id == "job_001"
        assert result.framework == "mujoco"
        assert result.task == "needle_insertion"

    def test_default_status_pending(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        assert result.status == "pending"

    def test_default_duration_zero(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        assert result.duration_s == 0.0

    def test_default_metrics_empty(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        assert result.metrics == {}

    def test_default_error_empty(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        assert result.error == ""

    def test_to_dict_returns_dict(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_keys(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        d = result.to_dict()
        expected = {
            "job_id",
            "framework",
            "task",
            "status",
            "start_time",
            "end_time",
            "duration_s",
            "metrics",
            "config",
            "error",
        }
        assert expected == set(d.keys())

    def test_to_dict_duration_rounded(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t", duration_s=1.23456)
        d = result.to_dict()
        assert d["duration_s"] == 1.23

    def test_custom_status(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t", status="completed")
        assert result.status == "completed"

    def test_metrics_writable(self, mod):
        result = mod.JobResult(job_id="j", framework="f", task="t")
        result.metrics["accuracy"] = 0.95
        assert result.metrics["accuracy"] == 0.95


# ---------------------------------------------------------------------------
# TestTaskRegistry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    """Tests for the TASK_REGISTRY constant."""

    def test_task_registry_exists(self, mod):
        assert hasattr(mod, "TASK_REGISTRY")
        assert isinstance(mod.TASK_REGISTRY, dict)

    def test_task_registry_nonempty(self, mod):
        assert len(mod.TASK_REGISTRY) > 0

    def test_needle_insertion_in_registry(self, mod):
        assert "needle_insertion" in mod.TASK_REGISTRY

    def test_tissue_retraction_in_registry(self, mod):
        assert "tissue_retraction" in mod.TASK_REGISTRY

    def test_task_has_description(self, mod):
        for task_id, task_def in mod.TASK_REGISTRY.items():
            assert "description" in task_def, f"Task '{task_id}' missing description"

    def test_task_has_frameworks(self, mod):
        for task_id, task_def in mod.TASK_REGISTRY.items():
            assert "frameworks" in task_def, f"Task '{task_id}' missing frameworks"
            assert isinstance(task_def["frameworks"], list)
            assert len(task_def["frameworks"]) > 0

    def test_task_has_metrics(self, mod):
        for task_id, task_def in mod.TASK_REGISTRY.items():
            assert "metrics" in task_def, f"Task '{task_id}' missing metrics"
            assert isinstance(task_def["metrics"], list)

    def test_task_has_default_params(self, mod):
        for task_id, task_def in mod.TASK_REGISTRY.items():
            assert "default_params" in task_def, f"Task '{task_id}' missing default_params"
            assert isinstance(task_def["default_params"], dict)


# ---------------------------------------------------------------------------
# TestFrameworkDetection
# ---------------------------------------------------------------------------


class TestFrameworkDetection:
    """Tests for framework detection logic."""

    def test_supported_frameworks_exists(self, mod):
        assert hasattr(mod, "SUPPORTED_FRAMEWORKS")
        assert isinstance(mod.SUPPORTED_FRAMEWORKS, list)

    def test_supported_frameworks_has_mujoco(self, mod):
        assert "mujoco" in mod.SUPPORTED_FRAMEWORKS

    def test_supported_frameworks_has_isaac(self, mod):
        assert "isaac" in mod.SUPPORTED_FRAMEWORKS

    def test_detect_framework_function_exists(self, mod):
        assert callable(mod._detect_framework)

    def test_detect_unknown_framework_returns_false(self, mod):
        assert mod._detect_framework("nonexistent_framework_xyz") is False

    def test_generate_job_id_contains_task(self, mod):
        job_id = mod._generate_job_id("mujoco", "needle_insertion")
        assert "needle_insertion" in job_id
        assert "mujoco" in job_id

    def test_resolve_runner_script_function_exists(self, mod):
        assert callable(mod._resolve_runner_script)
