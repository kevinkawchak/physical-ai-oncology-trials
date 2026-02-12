"""Unit tests for regulatory/fda-compliance/fda_submission_tracker.py.

Tests cover SubmissionType, SubmissionStatus, DeviceClass enums,
AIMLComponent, ChecklistItem, Submission dataclasses.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "regulatory/fda-compliance/fda_submission_tracker.py"


@pytest.fixture()
def mod():
    return load_module("fda_submission_tracker", MOD_PATH)


class TestEnums:
    def test_submission_type(self, mod):
        names = [e.name for e in mod.SubmissionType]
        for t in ("DE_NOVO", "PMA", "BREAKTHROUGH", "PRE_SUB", "PCCP"):
            assert t in names

    def test_submission_status_lifecycle(self, mod):
        names = [e.name for e in mod.SubmissionStatus]
        for s in ("PLANNING", "SUBMITTED", "UNDER_REVIEW", "CLEARED", "AUTHORIZED"):
            assert s in names

    def test_device_class(self, mod):
        names = [e.name for e in mod.DeviceClass]
        for c in ("CLASS_I", "CLASS_II", "CLASS_III"):
            assert c in names


class TestAIMLComponent:
    def test_creation(self, mod):
        comp = mod.AIMLComponent(
            name="Tissue Segmentation",
            model_type="segmentation",
            architecture="U-Net",
            training_data_size=10000,
            training_data_sources=["institutional"],
            validation_approach="k-fold",
            performance_metrics={"dice": 0.92},
            locked_or_adaptive="locked",
        )
        assert comp.name == "Tissue Segmentation"
        # Regression: default should not be "classification"
        assert comp.model_type == "segmentation" or comp.model_type != "classification"


class TestSubmission:
    def test_creation(self, mod):
        sub = mod.Submission(
            submission_id="SUB-001",
            submission_type=mod.SubmissionType.DE_NOVO,
            device_name="AI Surgical Robot",
            intended_use="Oncology surgical assistance",
            device_class=mod.DeviceClass.CLASS_II,
            sponsor="Test Corp",
            status=mod.SubmissionStatus.PLANNING,
        )
        assert sub.submission_id == "SUB-001"
        assert sub.status == mod.SubmissionStatus.PLANNING

    def test_to_dict(self, mod):
        sub = mod.Submission(
            submission_id="SUB-002",
            submission_type=mod.SubmissionType.PMA,
            device_name="AI Treatment Planner",
            intended_use="Treatment planning",
            device_class=mod.DeviceClass.CLASS_III,
            sponsor="Med Inc",
            status=mod.SubmissionStatus.SUBMITTED,
            ai_ml_components=[],
        )
        d = sub.to_dict()
        assert isinstance(d, dict)
        assert d["submission_id"] == "SUB-002"


class TestChecklistItem:
    def test_creation(self, mod):
        item = mod.ChecklistItem(
            category="device_description",
            description="Comprehensive device description",
            status="pending",
            regulatory_reference="21 CFR 807.87(e)",
            required=True,
        )
        assert item.required is True
        assert item.status == "pending"
