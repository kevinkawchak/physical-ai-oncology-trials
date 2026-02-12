"""Unit tests for regulatory/irb-management/irb_protocol_manager.py.

Tests cover ProtocolStatus, ReviewType, RiskCategory enums,
AIComponent, ConsentElement, Protocol dataclasses.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "regulatory/irb-management/irb_protocol_manager.py"


@pytest.fixture()
def mod():
    return load_module("irb_protocol_manager", MOD_PATH)


class TestEnums:
    def test_protocol_status(self, mod):
        names = [e.name for e in mod.ProtocolStatus]
        for s in ("DRAFT", "SUBMITTED", "APPROVED", "SUSPENDED", "TERMINATED"):
            assert s in names

    def test_review_type(self, mod):
        names = [e.name for e in mod.ReviewType]
        for r in ("FULL_BOARD", "EXPEDITED", "EXEMPT"):
            assert r in names

    def test_risk_category(self, mod):
        names = [e.name for e in mod.RiskCategory]
        assert "MINIMAL_RISK" in names
        assert "GREATER_THAN_MINIMAL" in names


class TestAIComponent:
    def test_creation(self, mod):
        comp = mod.AIComponent(
            name="Dose Optimization AI",
            function="Optimizes radiation dose distribution",
            autonomy_level="advisory",
            training_data_description="10,000 treatment plans",
            human_override=True,
            model_locked=True,
        )
        assert comp.name == "Dose Optimization AI"
        assert comp.human_override is True


class TestConsentElement:
    def test_creation(self, mod):
        elem = mod.ConsentElement(
            element="AI disclosure",
            description="Participant informed about AI use in treatment planning",
            required=True,
            ai_specific=True,
            regulatory_basis="Common Rule 45 CFR 46.116",
        )
        assert elem.ai_specific is True
        assert elem.required is True


class TestProtocol:
    def test_creation(self, mod):
        protocol = mod.Protocol(
            protocol_id="IRB-001",
            title="AI-Guided Surgical Robot Trial",
            pi_name="Dr. Smith",
            institution="Memorial Hospital",
            status=mod.ProtocolStatus.DRAFT,
            review_type=mod.ReviewType.FULL_BOARD,
            risk_category=mod.RiskCategory.GREATER_THAN_MINIMAL,
            participant_count=100,
            trial_phase="II",
        )
        assert protocol.protocol_id == "IRB-001"
        assert protocol.status == mod.ProtocolStatus.DRAFT

    def test_to_dict(self, mod):
        protocol = mod.Protocol(
            protocol_id="IRB-002",
            title="Test Protocol",
            pi_name="Dr. Jones",
            institution="University Hospital",
            status=mod.ProtocolStatus.APPROVED,
            review_type=mod.ReviewType.EXPEDITED,
            risk_category=mod.RiskCategory.MINIMAL_RISK,
        )
        d = protocol.to_dict()
        assert isinstance(d, dict)
        assert d["protocol_id"] == "IRB-002"
