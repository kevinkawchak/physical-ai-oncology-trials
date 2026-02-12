"""Tests for tools/deployment-readiness/deployment_readiness.py

Covers ReadinessReport dataclass, checklist generation, and safety checks.
"""

from __future__ import annotations


class TestReadinessReport:
    def test_creation_defaults(self, deploy_mod):
        report = deploy_mod.ReadinessReport(model_path="/tmp/model.onnx")
        assert report.model_path == "/tmp/model.onnx"
        assert report.overall_status == "unknown"
        assert report.blockers == []

    def test_to_dict(self, deploy_mod):
        report = deploy_mod.ReadinessReport(model_path="/tmp/model.onnx")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["model_path"] == "/tmp/model.onnx"
        assert "overall_status" in d


class TestGenerateChecklist:
    def test_checklist_returns_dict(self, deploy_mod):
        checklist = deploy_mod._generate_checklist()
        assert isinstance(checklist, dict)
        assert len(checklist) > 0

    def test_checklist_has_categories(self, deploy_mod):
        checklist = deploy_mod._generate_checklist()
        # Should have regulatory compliance categories
        keys = list(checklist.keys())
        assert len(keys) >= 3


class TestCheckSafetyConstraints:
    def test_returns_dict(self, deploy_mod):
        result = deploy_mod._check_safety_constraints(
            model_path="/tmp/nonexistent.onnx",
            constraints_path=None,
        )
        assert isinstance(result, dict)
