"""Unit tests for regulatory/regulatory-intelligence/regulatory_tracker.py.

Tests cover RegulatoryJurisdiction, UpdateType, ImpactLevel enums,
RegulatoryUpdate, ComplianceDeadline dataclasses, and REGULATORY_TIMELINE.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "regulatory/regulatory-intelligence/regulatory_tracker.py"


@pytest.fixture()
def mod():
    return load_module("regulatory_tracker", MOD_PATH)


class TestEnums:
    def test_jurisdiction(self, mod):
        names = [e.name for e in mod.RegulatoryJurisdiction]
        for j in ("FDA", "EMA", "ICH", "WHO", "HHS_OCR"):
            assert j in names

    def test_update_type(self, mod):
        names = [e.name for e in mod.UpdateType]
        for t in ("GUIDANCE_DRAFT", "GUIDANCE_FINAL", "ENFORCEMENT_ACTION"):
            assert t in names

    def test_impact_level(self, mod):
        names = [e.name for e in mod.ImpactLevel]
        for l in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"):
            assert l in names


class TestRegulatoryUpdate:
    def test_creation(self, mod):
        update = mod.RegulatoryUpdate(
            update_id="RU-001",
            jurisdiction=mod.RegulatoryJurisdiction.FDA,
            title="AI/ML PCCP Guidance Final",
            date="2025-08-15",
            update_type=mod.UpdateType.GUIDANCE_FINAL,
            impact_level=mod.ImpactLevel.CRITICAL,
            summary="FDA finalizes PCCP guidance for AI/ML-enabled devices",
            affected_areas=["AI/ML", "SaMD"],
        )
        assert update.update_id == "RU-001"
        assert update.impact_level == mod.ImpactLevel.CRITICAL

    def test_to_dict(self, mod):
        update = mod.RegulatoryUpdate(
            update_id="RU-002",
            jurisdiction=mod.RegulatoryJurisdiction.EMA,
            title="EU AI Act Implementation",
            date="2026-02-01",
            update_type=mod.UpdateType.REGULATION_FINAL,
            impact_level=mod.ImpactLevel.HIGH,
            summary="EU AI Act enforcement begins",
            affected_areas=["AI", "Medical Devices"],
        )
        d = update.to_dict()
        assert isinstance(d, dict)
        assert d["update_id"] == "RU-002"


class TestRegulatoryTimeline:
    def test_timeline_exists(self, mod):
        assert len(mod.REGULATORY_TIMELINE) > 0

    def test_timeline_entries_have_fields(self, mod):
        for entry in mod.REGULATORY_TIMELINE:
            assert "date" in entry or "title" in entry
