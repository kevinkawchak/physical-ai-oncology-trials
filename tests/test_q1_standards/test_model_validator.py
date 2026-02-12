"""Unit tests for q1-2026-standards/objective-2-robot-model-repository/model_validator.py.

Tests cover ValidationResult, ModelValidationReport dataclasses,
and FormatValidator.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "q1-2026-standards/objective-2-robot-model-repository/model_validator.py"


@pytest.fixture()
def mod():
    return load_module("model_validator", MOD_PATH)


class TestValidationResult:
    def test_passed(self, mod):
        result = mod.ValidationResult(
            check_name="joint_limits",
            passed=True,
            level="info",
            message="All joint limits within range",
        )
        assert result.passed is True

    def test_failed(self, mod):
        result = mod.ValidationResult(
            check_name="mass_positive",
            passed=False,
            level="error",
            message="Link 'arm_3' has negative mass",
        )
        assert result.passed is False


class TestModelValidationReport:
    def test_creation(self, mod):
        report = mod.ModelValidationReport(
            model_path="/models/robot.urdf",
            model_name="test_robot",
            validation_level="basic",
        )
        assert report.model_name == "test_robot"
        assert report.overall_passed is True  # No failures yet

    def test_add_result(self, mod):
        report = mod.ModelValidationReport(
            model_path="/models/robot.urdf",
            model_name="test_robot",
            validation_level="full",
        )
        report.add_result(
            mod.ValidationResult(
                check_name="test_check",
                passed=True,
                level="info",
                message="OK",
            )
        )
        assert len(report.results) == 1

    def test_add_failure_changes_overall(self, mod):
        report = mod.ModelValidationReport(
            model_path="/models/robot.urdf",
            model_name="test_robot",
            validation_level="full",
        )
        report.add_result(
            mod.ValidationResult(
                check_name="failed_check",
                passed=False,
                level="error",
                message="Failed",
            )
        )
        assert report.overall_passed is False

    def test_to_dict(self, mod):
        report = mod.ModelValidationReport(
            model_path="/models/robot.urdf",
            model_name="test_robot",
            validation_level="basic",
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "test_robot"
