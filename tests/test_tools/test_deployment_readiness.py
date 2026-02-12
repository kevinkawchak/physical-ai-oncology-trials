from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("deployment_readiness", "tools/deployment-readiness/deployment_readiness.py")


# ---------------------------------------------------------------------------
# TestReadinessReport
# ---------------------------------------------------------------------------


class TestReadinessReport:
    """Tests for the ReadinessReport dataclass."""

    def test_creation_minimal(self, mod):
        report = mod.ReadinessReport(model_path="model.onnx")
        assert report.model_path == "model.onnx"

    def test_default_overall_status(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx")
        assert report.overall_status == "unknown"

    def test_default_blockers_empty(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx")
        assert report.blockers == []

    def test_default_fields_are_dicts(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx")
        for attr in ("model_info", "compatibility", "benchmark", "safety", "checklist", "validation"):
            assert isinstance(getattr(report, attr), dict)

    def test_to_dict_returns_dict(self, mod):
        report = mod.ReadinessReport(model_path="m.onnx")
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_model_path(self, mod):
        report = mod.ReadinessReport(model_path="abc.onnx")
        d = report.to_dict()
        assert d["model_path"] == "abc.onnx"

    def test_to_dict_contains_all_keys(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx")
        d = report.to_dict()
        expected_keys = {
            "model_path",
            "timestamp",
            "model_info",
            "compatibility",
            "benchmark",
            "safety",
            "checklist",
            "validation",
            "overall_status",
            "blockers",
        }
        assert expected_keys == set(d.keys())

    def test_timestamp_field(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx", timestamp="2026-01-01T00:00:00")
        assert report.timestamp == "2026-01-01T00:00:00"

    def test_blockers_mutable(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx")
        report.blockers.append("test blocker")
        assert "test blocker" in report.blockers

    def test_overall_status_custom(self, mod):
        report = mod.ReadinessReport(model_path="x.onnx", overall_status="READY")
        assert report.overall_status == "READY"


# ---------------------------------------------------------------------------
# TestRegulatoryChecklist
# ---------------------------------------------------------------------------


class TestRegulatoryChecklist:
    """Tests for the REGULATORY_CHECKLIST constant and _generate_checklist."""

    def test_regulatory_checklist_exists(self, mod):
        assert hasattr(mod, "REGULATORY_CHECKLIST")

    def test_regulatory_checklist_has_iec_62304(self, mod):
        assert "iec_62304" in mod.REGULATORY_CHECKLIST

    def test_regulatory_checklist_has_fda_aiml(self, mod):
        assert "fda_aiml" in mod.REGULATORY_CHECKLIST

    def test_regulatory_checklist_has_iso_14971(self, mod):
        assert "iso_14971" in mod.REGULATORY_CHECKLIST

    def test_generate_checklist_returns_dict(self, mod):
        checklist = mod._generate_checklist()
        assert isinstance(checklist, dict)

    def test_generate_checklist_has_standards(self, mod):
        checklist = mod._generate_checklist()
        assert len(checklist) == 3

    def test_generate_checklist_items_have_status(self, mod):
        checklist = mod._generate_checklist()
        for standard_id, standard in checklist.items():
            for item in standard["items"]:
                assert item["status"] == "not_assessed"


# ---------------------------------------------------------------------------
# TestSafetyCategories
# ---------------------------------------------------------------------------


class TestSafetyCategories:
    """Tests for the SAFETY_CATEGORIES constant."""

    def test_safety_categories_exists(self, mod):
        assert hasattr(mod, "SAFETY_CATEGORIES")

    def test_safety_categories_has_force_limits(self, mod):
        assert "force_limits" in mod.SAFETY_CATEGORIES

    def test_safety_categories_has_latency_budget(self, mod):
        assert "latency_budget" in mod.SAFETY_CATEGORIES

    def test_safety_categories_entries_have_description(self, mod):
        for cat_id, cat in mod.SAFETY_CATEGORIES.items():
            assert "description" in cat
            assert "unit" in cat
            assert "reference" in cat


# ---------------------------------------------------------------------------
# TestModuleFunctions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    """Verify key functions exist and behave correctly with edge-case inputs."""

    def test_check_model_compatibility_exists(self, mod):
        assert callable(mod._check_model_compatibility)

    def test_check_model_compatibility_missing_file(self, mod):
        result = mod._check_model_compatibility("/nonexistent/model.onnx")
        assert result["onnx_valid"] is False
        assert len(result["issues"]) > 0

    def test_benchmark_inference_exists(self, mod):
        assert callable(mod._benchmark_inference)

    def test_benchmark_inference_missing_file(self, mod):
        result = mod._benchmark_inference("/nonexistent/model.onnx", iterations=1)
        assert len(result["issues"]) > 0

    def test_check_safety_constraints_no_file(self, mod):
        result = mod._check_safety_constraints("x.onnx", None)
        assert result["constraints_checked"] == 0
        assert len(result["issues"]) > 0

    def test_validate_against_reference_missing_ref(self, mod):
        result = mod._validate_against_reference("x.onnx", "/nonexistent/ref.npz", 1e-4)
        assert len(result["issues"]) > 0

    def test_validate_against_reference_tolerance_stored(self, mod):
        result = mod._validate_against_reference("x.onnx", "/nonexistent/ref.npz", 0.001)
        assert result["tolerance"] == 0.001
