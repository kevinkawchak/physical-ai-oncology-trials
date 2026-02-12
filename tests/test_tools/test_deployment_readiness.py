"""Unit tests for tools/deployment-readiness/deployment_readiness.py.

Tests cover REGULATORY_CHECKLIST, SAFETY_CATEGORIES constants,
and pre-deployment validation logic.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "tools/deployment-readiness/deployment_readiness.py"


@pytest.fixture()
def mod():
    return load_module("deployment_readiness", MOD_PATH)


class TestConstants:
    def test_regulatory_checklist_sections(self, mod):
        checklist = mod.REGULATORY_CHECKLIST
        for section in ("iec_62304", "fda_aiml", "iso_14971"):
            assert section in checklist

    def test_checklist_items_nonempty(self, mod):
        for section, items in mod.REGULATORY_CHECKLIST.items():
            assert len(items) > 0, f"{section} has no checklist items"

    def test_safety_categories(self, mod):
        cats = mod.SAFETY_CATEGORIES
        for cat in ("force_limits", "workspace_bounds", "velocity_limits", "latency_budget"):
            assert cat in cats

    def test_safety_categories_have_values(self, mod):
        for cat, val in mod.SAFETY_CATEGORIES.items():
            assert val is not None, f"{cat} has None value"


class TestChecklistGeneration:
    def test_generate_checklist(self, mod):
        checklist = mod._generate_checklist()
        total_items = sum(len(section["items"]) for section in checklist.values())
        assert total_items >= 10

    def test_iec_62304_requirements(self, mod):
        checklist = mod._generate_checklist()
        iec_items = checklist["iec_62304"]["items"]
        assert len(iec_items) >= 3

    def test_fda_aiml_requirements(self, mod):
        checklist = mod._generate_checklist()
        fda_items = checklist["fda_aiml"]["items"]
        assert len(fda_items) >= 3
