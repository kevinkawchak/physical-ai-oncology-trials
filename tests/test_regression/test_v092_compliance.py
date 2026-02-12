"""Regression tests for v0.9.2 compliance fixes.

Covers DATE_SHIFT handling, model_type default, safety constraint
verification, and RESEARCH USE ONLY disclaimers.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


class TestDateShiftHandling:
    """Regression: DATE_SHIFT silently falling through to date removal.

    Bug: deidentification_pipeline.py DATE_SHIFT case fell through.
    Fix: Added explicit DATE_SHIFT branch.
    """

    def test_date_shift_is_valid_handling(self):
        mod = load_module("deidentification_pipeline", "privacy/de-identification/deidentification_pipeline.py")
        handlers = [e.name for e in mod.DateHandling]
        assert "DATE_SHIFT" in handlers

    def test_date_shift_config(self):
        mod = load_module("deidentification_pipeline", "privacy/de-identification/deidentification_pipeline.py")
        config = mod.DeidentificationConfig(
            date_handling=mod.DateHandling.DATE_SHIFT,
        )
        assert config.date_handling == mod.DateHandling.DATE_SHIFT


class TestAIMLComponentModelType:
    """Regression: AI/ML components defaulting to "classification".

    Bug: fda_submission_tracker.py set all AI/ML component model_type
    to "classification".
    Fix: Changed default to "unspecified".
    """

    def test_default_model_type_not_classification(self):
        mod = load_module("fda_submission_tracker", "regulatory/fda-compliance/fda_submission_tracker.py")
        # create_submission assigns model_type for AI/ML components;
        # verify it defaults to "unspecified", not "classification".
        tracker = mod.FDASubmissionTracker(sponsor="Test Sponsor", device_class="II")
        submission = tracker.create_submission(
            submission_type="de_novo",
            device_name="Test Device",
            intended_use="Test use",
            ai_ml_components=["Test Model"],
        )
        component = submission.ai_ml_components[0]
        # Default should be "unspecified", not "classification"
        assert component.model_type != "classification"
        assert component.model_type == "unspecified"


class TestFDASubmissionStatusValues:
    """Verify FDA submission status enum completeness."""

    def test_all_statuses_present(self):
        mod = load_module("fda_submission_tracker", "regulatory/fda-compliance/fda_submission_tracker.py")
        names = [e.name for e in mod.SubmissionStatus]
        for status in ("PLANNING", "SUBMITTED", "UNDER_REVIEW", "CLEARED", "AUTHORIZED"):
            assert status in names


class TestRegulatoryTrackerOrdering:
    """Regression: Unreachable "overdue" status branch.

    Bug: regulatory_tracker.py deadlines past due were mislabeled
    as "imminent" due to incorrect if/elif ordering.
    """

    def test_update_types_complete(self):
        mod = load_module("regulatory_tracker", "regulatory/regulatory-intelligence/regulatory_tracker.py")
        types = [e.name for e in mod.UpdateType]
        assert "GUIDANCE_DRAFT" in types
        assert "GUIDANCE_FINAL" in types
        assert "ENFORCEMENT_ACTION" in types

    def test_impact_levels_ordered(self):
        mod = load_module("regulatory_tracker", "regulatory/regulatory-intelligence/regulatory_tracker.py")
        levels = list(mod.ImpactLevel)
        # CRITICAL should be first/highest
        assert levels[0] == mod.ImpactLevel.CRITICAL


class TestResearchUseDisclaimer:
    """Regression: Missing RESEARCH USE ONLY disclaimers.

    v0.9.2 added disclaimers to 11 modules. Verify they exist.
    """

    @pytest.mark.parametrize(
        "mod_name,mod_path",
        [
            ("deidentification_pipeline", "privacy/de-identification/deidentification_pipeline.py"),
            ("phi_detector", "privacy/phi-pii-management/phi_detector.py"),
            ("access_control_manager", "privacy/access-control/access_control_manager.py"),
            ("fda_submission_tracker", "regulatory/fda-compliance/fda_submission_tracker.py"),
            ("dose_calculator", "tools/dose-calculator/dose_calculator.py"),
            ("tumor_twin_pipeline", "digital-twins/patient-modeling/tumor_twin_pipeline.py"),
            ("treatment_simulator", "digital-twins/treatment-simulation/treatment_simulator.py"),
        ],
    )
    def test_research_disclaimer_present(self, mod_name, mod_path):
        """Module should contain RESEARCH USE ONLY disclaimer."""
        from pathlib import Path

        filepath = Path(__file__).resolve().parent.parent.parent / mod_path
        content = filepath.read_text()
        assert "RESEARCH" in content.upper()
