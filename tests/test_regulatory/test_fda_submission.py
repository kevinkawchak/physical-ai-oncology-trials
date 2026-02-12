"""Tests for FDA Submission Tracker.

Validates submission creation, checklist generation, status updates,
milestone tracking, and the v0.9.2 model_type default fix.

LICENSE: MIT
"""

from __future__ import annotations


import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "regulatory/fda-compliance/fda_submission_tracker.py"


@pytest.fixture()
def fda_mod():
    return load_module("fda_submission_tracker", MOD_PATH)


@pytest.fixture()
def tracker(fda_mod):
    return fda_mod.FDASubmissionTracker(sponsor="Test Sponsor", device_class="II", review_division="CDRH")


@pytest.fixture()
def sample_submission(tracker):
    return tracker.create_submission(
        submission_type="de_novo",
        device_name="AI Surgical Planner",
        intended_use="AI-assisted planning for tumor resection",
        ai_ml_components=["segmentation_model", "path_optimizer"],
        breakthrough_designation=False,
        oncology_indication="NSCLC",
    )


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_submission_type_values(self, fda_mod):
        names = {e.name for e in fda_mod.SubmissionType}
        assert "FIVE_TEN_K" in names
        assert "DE_NOVO" in names
        assert "PMA" in names
        assert "PCCP" in names

    def test_submission_status_values(self, fda_mod):
        statuses = {e.name for e in fda_mod.SubmissionStatus}
        assert "PLANNING" in statuses
        assert "CLEARED" in statuses
        assert "WITHDRAWN" in statuses
        assert len(statuses) == 11

    def test_device_class_values(self, fda_mod):
        assert fda_mod.DeviceClass.CLASS_I.value == "I"
        assert fda_mod.DeviceClass.CLASS_II.value == "II"
        assert fda_mod.DeviceClass.CLASS_III.value == "III"


# ===========================================================================
# TestAIMLComponent
# ===========================================================================


class TestAIMLComponent:
    def test_default_fields(self, fda_mod):
        c = fda_mod.AIMLComponent(name="seg", model_type="classification")
        assert c.locked_or_adaptive == "locked"
        assert c.pccp_planned is False
        assert c.architecture == ""

    def test_v092_model_type_defaults_to_unspecified(self, tracker, sample_submission):
        """v0.9.2 fix: components created via tracker default model_type to 'unspecified'."""
        for comp in sample_submission.ai_ml_components:
            assert comp.model_type == "unspecified"


# ===========================================================================
# TestSubmissionCreation
# ===========================================================================


class TestSubmissionCreation:
    def test_submission_id_format(self, sample_submission):
        assert sample_submission.submission_id.startswith("SUB-")

    def test_initial_status_is_planning(self, sample_submission, fda_mod):
        assert sample_submission.status == fda_mod.SubmissionStatus.PLANNING

    def test_components_created(self, sample_submission):
        assert len(sample_submission.ai_ml_components) == 2
        names = [c.name for c in sample_submission.ai_ml_components]
        assert "segmentation_model" in names
        assert "path_optimizer" in names

    def test_milestones_initialized(self, sample_submission):
        assert len(sample_submission.milestones) >= 1
        assert sample_submission.milestones[0]["event"] == "submission_created"

    def test_submission_stored(self, tracker, sample_submission):
        assert sample_submission.submission_id in tracker._submissions

    def test_to_dict(self, sample_submission):
        d = sample_submission.to_dict()
        assert "submission_id" in d
        assert d["submission_type"] == "de_novo"
        assert d["ai_ml_components"] == 2

    def test_oncology_indication(self, sample_submission):
        assert sample_submission.oncology_indication == "NSCLC"


# ===========================================================================
# TestChecklistGeneration
# ===========================================================================


class TestChecklistGeneration:
    def test_checklist_has_items(self, tracker, sample_submission):
        cl = tracker.generate_presub_checklist(sample_submission)
        assert len(cl.items) > 0

    def test_de_novo_specific_items(self, tracker, sample_submission):
        cl = tracker.generate_presub_checklist(sample_submission)
        categories = {item.category for item in cl.items}
        assert "de_novo_specific" in categories

    def test_breakthrough_items_absent_when_not_designated(self, tracker, sample_submission):
        cl = tracker.generate_presub_checklist(sample_submission)
        categories = {item.category for item in cl.items}
        assert "breakthrough_designation" not in categories

    def test_breakthrough_items_present_when_designated(self, tracker):
        sub = tracker.create_submission(
            submission_type="510k",
            device_name="Device",
            intended_use="Testing",
            breakthrough_designation=True,
        )
        cl = tracker.generate_presub_checklist(sub)
        categories = {item.category for item in cl.items}
        assert "breakthrough_designation" in categories

    def test_completion_starts_at_zero(self, tracker, sample_submission):
        cl = tracker.generate_presub_checklist(sample_submission)
        assert cl.completion_percentage == 0.0

    def test_update_completion(self, tracker, sample_submission, fda_mod):
        cl = tracker.generate_presub_checklist(sample_submission)
        # Mark all items complete
        for item in cl.items:
            item.status = "complete"
        cl.update_completion()
        assert cl.completion_percentage == 100.0

    def test_completion_partial(self, tracker, sample_submission, fda_mod):
        cl = tracker.generate_presub_checklist(sample_submission)
        required = [i for i in cl.items if i.required]
        if len(required) > 1:
            required[0].status = "complete"
            cl.update_completion()
            assert 0 < cl.completion_percentage < 100


# ===========================================================================
# TestStatusUpdates
# ===========================================================================


class TestStatusUpdates:
    def test_update_status(self, tracker, sample_submission, fda_mod):
        tracker.update_status(sample_submission.submission_id, "submitted", notes="Filed with CDRH")
        assert sample_submission.status == fda_mod.SubmissionStatus.SUBMITTED

    def test_milestone_added_on_update(self, tracker, sample_submission):
        initial_count = len(sample_submission.milestones)
        tracker.update_status(sample_submission.submission_id, "under_review")
        assert len(sample_submission.milestones) == initial_count + 1

    def test_update_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            tracker.update_status("FAKE-ID", "submitted")

    def test_get_submission_summary(self, tracker, sample_submission):
        summary = tracker.get_submission_summary(sample_submission.submission_id)
        assert "AI Surgical Planner" in summary
        assert "de_novo" in summary

    def test_summary_for_nonexistent_id(self, tracker):
        result = tracker.get_submission_summary("DOES-NOT-EXIST")
        assert "not found" in result.lower()
