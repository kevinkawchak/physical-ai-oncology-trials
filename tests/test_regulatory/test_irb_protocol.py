"""Tests for IRB Protocol Manager.

Validates protocol creation, AI-specific consent generation,
review checklist assembly, and amendment tracking.

LICENSE: MIT
"""

from __future__ import annotations


import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "regulatory/irb-management/irb_protocol_manager.py"


@pytest.fixture()
def irb_mod():
    return load_module("irb_protocol_manager", MOD_PATH)


@pytest.fixture()
def manager(irb_mod):
    return irb_mod.IRBProtocolManager(
        institution="Test Medical Center",
        irb_type="central",
        fwa_number="FWA00012345",
    )


@pytest.fixture()
def sample_protocol(manager):
    return manager.create_protocol(
        title="AI-Guided Tumor Resection Trial",
        pi_name="Dr. Smith",
        ai_components=["tumor_segmentation", "path_planning", "outcome_prediction"],
        participant_count=200,
        trial_phase="pivotal",
    )


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_protocol_status_values(self, irb_mod):
        names = {e.name for e in irb_mod.ProtocolStatus}
        assert "DRAFT" in names
        assert "APPROVED" in names
        assert "SUSPENDED" in names
        assert len(names) == 9

    def test_review_type_values(self, irb_mod):
        names = {e.name for e in irb_mod.ReviewType}
        assert "FULL_BOARD" in names
        assert "EXPEDITED" in names
        assert "ADVERSE_EVENT" in names

    def test_risk_category_values(self, irb_mod):
        names = {e.name for e in irb_mod.RiskCategory}
        assert names == {"MINIMAL_RISK", "GREATER_THAN_MINIMAL"}


# ===========================================================================
# TestAIComponent
# ===========================================================================


class TestAIComponent:
    def test_default_values(self, irb_mod):
        comp = irb_mod.AIComponent(name="test", function="diagnostic")
        assert comp.autonomy_level == "advisory"
        assert comp.human_override is True
        assert comp.model_locked is True

    def test_custom_values(self, irb_mod):
        comp = irb_mod.AIComponent(
            name="auto_planner",
            function="treatment_planning",
            autonomy_level="semi_autonomous",
            human_override=True,
            model_locked=False,
        )
        assert comp.autonomy_level == "semi_autonomous"
        assert comp.model_locked is False


# ===========================================================================
# TestProtocolCreation
# ===========================================================================


class TestProtocolCreation:
    def test_protocol_id_format(self, sample_protocol):
        assert sample_protocol.protocol_id.startswith("IRB-")

    def test_initial_status_is_draft(self, sample_protocol, irb_mod):
        assert sample_protocol.status == irb_mod.ProtocolStatus.DRAFT

    def test_ai_components_created(self, sample_protocol):
        assert len(sample_protocol.ai_components) == 3

    def test_ai_components_have_human_override(self, sample_protocol):
        for comp in sample_protocol.ai_components:
            assert comp.human_override is True

    def test_consent_elements_generated(self, sample_protocol):
        assert len(sample_protocol.consent_elements) > 0

    def test_consent_elements_are_ai_specific(self, sample_protocol):
        for elem in sample_protocol.consent_elements:
            assert elem.ai_specific is True

    def test_to_dict(self, sample_protocol):
        d = sample_protocol.to_dict()
        assert d["protocol_id"] == sample_protocol.protocol_id
        assert d["ai_components"] == 3
        assert d["participant_count"] == 200

    def test_protocol_stored(self, manager, sample_protocol):
        assert sample_protocol.protocol_id in manager._protocols

    def test_review_type_defaults_to_full_board(self, sample_protocol, irb_mod):
        assert sample_protocol.review_type == irb_mod.ReviewType.FULL_BOARD

    def test_risk_category_greater_than_minimal(self, sample_protocol, irb_mod):
        assert sample_protocol.risk_category == irb_mod.RiskCategory.GREATER_THAN_MINIMAL

    def test_minimal_risk_protocol(self, manager, irb_mod):
        proto = manager.create_protocol(
            title="Minimal Risk AI Study",
            pi_name="Dr. Lee",
            ai_components=["screening_tool"],
            risk_category="minimal_risk",
        )
        assert proto.risk_category == irb_mod.RiskCategory.MINIMAL_RISK


# ===========================================================================
# TestReviewChecklist
# ===========================================================================


class TestReviewChecklist:
    def test_checklist_generated(self, manager, sample_protocol):
        cl = manager.generate_ai_review_checklist(sample_protocol)
        assert len(cl.items) > 0

    def test_checklist_categories(self, manager, sample_protocol):
        cl = manager.generate_ai_review_checklist(sample_protocol)
        categories = {item.category for item in cl.items}
        assert "ai_transparency" in categories
        assert "algorithmic_bias" in categories
        assert "consent" in categories
        assert "human_oversight" in categories

    def test_all_items_ai_specific(self, manager, sample_protocol):
        cl = manager.generate_ai_review_checklist(sample_protocol)
        for item in cl.items:
            assert item.ai_specific is True

    def test_checklist_submission_id_matches(self, manager, sample_protocol):
        cl = manager.generate_ai_review_checklist(sample_protocol)
        assert cl.submission_id == sample_protocol.protocol_id


# ===========================================================================
# TestConsentTemplate
# ===========================================================================


class TestConsentTemplate:
    def test_consent_template_generated(self, manager, sample_protocol):
        text = manager.generate_consent_template(sample_protocol)
        assert "INFORMED CONSENT DOCUMENT" in text

    def test_consent_includes_protocol_info(self, manager, sample_protocol):
        text = manager.generate_consent_template(sample_protocol)
        assert sample_protocol.protocol_id in text
        assert sample_protocol.pi_name in text

    def test_consent_includes_ai_components(self, manager, sample_protocol):
        text = manager.generate_consent_template(sample_protocol)
        assert "tumor_segmentation" in text

    def test_consent_includes_rights(self, manager, sample_protocol):
        text = manager.generate_consent_template(sample_protocol)
        assert "withdraw" in text.lower()

    def test_consent_includes_ai_disclosure(self, manager, sample_protocol):
        text = manager.generate_consent_template(sample_protocol)
        assert "artificial intelligence" in text.lower() or "ai" in text.lower()


# ===========================================================================
# TestAmendments
# ===========================================================================


class TestAmendments:
    def test_file_amendment(self, manager, sample_protocol):
        manager.file_amendment(
            sample_protocol.protocol_id,
            amendment_type="major",
            description="Updated AI model architecture",
            ai_model_changed=True,
        )
        assert len(sample_protocol.amendments) == 1
        assert sample_protocol.amendments[0]["ai_model_changed"] is True

    def test_amendment_status(self, manager, sample_protocol):
        manager.file_amendment(
            sample_protocol.protocol_id,
            amendment_type="minor",
            description="Updated labeling",
        )
        assert sample_protocol.amendments[0]["status"] == "submitted"

    def test_amendment_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.file_amendment("FAKE-ID", "minor", "test")

    def test_multiple_amendments(self, manager, sample_protocol):
        manager.file_amendment(sample_protocol.protocol_id, "minor", "Fix typo")
        manager.file_amendment(sample_protocol.protocol_id, "major", "New AI model", ai_model_changed=True)
        assert len(sample_protocol.amendments) == 2


# ===========================================================================
# TestConsentElements
# ===========================================================================


class TestConsentElements:
    def test_consent_element_defaults(self, irb_mod):
        elem = irb_mod.ConsentElement(element="test", description="desc")
        assert elem.required is True
        assert elem.ai_specific is False

    def test_ai_consent_elements_count(self, irb_mod):
        elems = irb_mod.AI_CONSENT_ELEMENTS
        assert len(elems) >= 6
        required_count = sum(1 for e in elems if e["required"])
        assert required_count >= 5
