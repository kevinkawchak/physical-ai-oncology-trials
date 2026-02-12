"""Tests for regulatory/irb-management/irb_protocol_manager.py

Covers IRB protocol lifecycle, AI review checklist, and consent generation.
"""

from __future__ import annotations


class TestEnums:
    def test_protocol_status(self, irb_mod):
        ps = irb_mod.ProtocolStatus
        assert ps.DRAFT.value == "draft"
        assert ps.APPROVED.value == "approved"
        assert ps.TERMINATED.value == "terminated"

    def test_review_types(self, irb_mod):
        rt = irb_mod.ReviewType
        assert rt.FULL_BOARD.value == "full_board"
        assert rt.EXPEDITED.value == "expedited"
        assert rt.EXEMPT.value == "exempt"

    def test_risk_categories(self, irb_mod):
        rc = irb_mod.RiskCategory
        assert rc.MINIMAL_RISK.value == "minimal_risk"


class TestAIComponent:
    def test_creation(self, irb_mod):
        comp = irb_mod.AIComponent(
            name="TumorSegmenter",
            function="Automated tumor segmentation",
        )
        assert comp.name == "TumorSegmenter"
        assert comp.autonomy_level == "advisory"
        assert comp.human_override is True


class TestIRBProtocolManager:
    def test_creation(self, irb_mod):
        mgr = irb_mod.IRBProtocolManager(institution="Test University")
        assert mgr.institution == "Test University"

    def test_create_protocol(self, irb_mod):
        mgr = irb_mod.IRBProtocolManager(institution="Test Hospital")
        protocol = mgr.create_protocol(
            title="AI-Guided Surgical Trial",
            pi_name="Dr. Test",
            ai_components=["tumor_segmenter", "dose_calculator"],
        )
        assert isinstance(protocol, irb_mod.Protocol)
        assert protocol.pi_name == "Dr. Test"
        assert len(protocol.ai_components) >= 2

    def test_generate_ai_review_checklist(self, irb_mod):
        mgr = irb_mod.IRBProtocolManager(institution="Test Hospital")
        protocol = mgr.create_protocol(
            title="AI Trial",
            pi_name="Dr. Smith",
            ai_components=["model_a"],
        )
        checklist = mgr.generate_ai_review_checklist(protocol)
        assert isinstance(checklist, irb_mod.SubmissionChecklist)
        assert len(checklist.items) > 0

    def test_generate_consent_template(self, irb_mod):
        mgr = irb_mod.IRBProtocolManager(institution="Test Hospital")
        protocol = mgr.create_protocol(
            title="AI Surgical Trial",
            pi_name="Dr. Jones",
            ai_components=["segmenter"],
        )
        consent = mgr.generate_consent_template(protocol)
        assert isinstance(consent, str)
        assert len(consent) > 0
