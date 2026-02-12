"""Tests for regulatory/fda-compliance/fda_submission_tracker.py

Covers SubmissionType, DeviceClass, FDASubmissionTracker, and checklist generation.
"""

from __future__ import annotations


class TestEnums:
    def test_submission_types(self, fda_mod):
        st = fda_mod.SubmissionType
        assert st.FIVE_TEN_K.value == "510k"
        assert st.DE_NOVO.value == "de_novo"
        assert st.PMA.value == "pma"

    def test_submission_status(self, fda_mod):
        ss = fda_mod.SubmissionStatus
        assert ss.PLANNING.value == "planning"
        assert ss.CLEARED.value == "cleared"

    def test_device_classes(self, fda_mod):
        dc = fda_mod.DeviceClass
        assert dc.CLASS_I.value == "I"
        assert dc.CLASS_II.value == "II"
        assert dc.CLASS_III.value == "III"


class TestAIMLComponent:
    def test_creation(self, fda_mod):
        comp = fda_mod.AIMLComponent(
            name="DosePredictor",
            model_type="unspecified",
            architecture="CNN",
        )
        assert comp.name == "DosePredictor"
        assert comp.locked_or_adaptive == "locked"


class TestFDASubmissionTracker:
    def test_creation(self, fda_mod):
        tracker = fda_mod.FDASubmissionTracker(sponsor="TestCorp")
        assert tracker.sponsor == "TestCorp"

    def test_create_submission(self, fda_mod):
        tracker = fda_mod.FDASubmissionTracker(sponsor="MedTech")
        submission = tracker.create_submission(
            submission_type="510k",
            device_name="AI Dose Calculator",
            intended_use="Treatment planning aid",
        )
        assert isinstance(submission, fda_mod.Submission)
        assert submission.sponsor == "MedTech"
        assert submission.status == fda_mod.SubmissionStatus.PLANNING

    def test_create_submission_with_ai(self, fda_mod):
        tracker = fda_mod.FDASubmissionTracker(sponsor="MedTech")
        submission = tracker.create_submission(
            submission_type="de_novo",
            device_name="AI Tumor Segmenter",
            intended_use="Automated tumor segmentation",
            ai_ml_components=["segmentation_model"],
        )
        assert len(submission.ai_ml_components) >= 1

    def test_generate_checklist(self, fda_mod):
        tracker = fda_mod.FDASubmissionTracker(sponsor="MedTech")
        submission = tracker.create_submission(
            submission_type="510k",
            device_name="Test Device",
            intended_use="Clinical decision support",
        )
        checklist = tracker.generate_presub_checklist(submission)
        assert isinstance(checklist, fda_mod.SubmissionChecklist)
        assert len(checklist.items) > 0
