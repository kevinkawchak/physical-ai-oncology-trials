"""Tests for regulatory/regulatory-intelligence/regulatory_tracker.py

Covers regulatory updates, deadlines, and landscape reporting.
"""

from __future__ import annotations


class TestEnums:
    def test_jurisdictions(self, reg_tracker_mod):
        rj = reg_tracker_mod.RegulatoryJurisdiction
        assert rj.FDA.value == "fda"
        assert rj.EMA.value == "ema"
        assert rj.ICH.value == "ich"

    def test_update_types(self, reg_tracker_mod):
        ut = reg_tracker_mod.UpdateType
        assert ut.GUIDANCE_FINAL.value == "guidance_final"
        assert ut.DEVICE_CLEARANCE.value == "device_clearance"

    def test_impact_levels(self, reg_tracker_mod):
        il = reg_tracker_mod.ImpactLevel
        assert il.CRITICAL.value == "critical"
        assert il.INFORMATIONAL.value == "informational"


class TestRegulatoryUpdate:
    def test_creation(self, reg_tracker_mod):
        update = reg_tracker_mod.RegulatoryUpdate(
            update_id="RU-001",
            jurisdiction="FDA",
            title="New AI Guidance",
            date="2026-01-15",
            update_type="guidance_final",
            impact_level="high",
            summary="FDA releases final guidance on AI/ML.",
        )
        assert update.update_id == "RU-001"

    def test_to_dict(self, reg_tracker_mod):
        update = reg_tracker_mod.RegulatoryUpdate(
            update_id="RU-002",
            jurisdiction="EMA",
            title="Update",
            date="2026-02-01",
            update_type="guidance_draft",
            impact_level="medium",
            summary="Draft guidance.",
        )
        d = update.to_dict()
        assert isinstance(d, dict)
        assert d["jurisdiction"] == "EMA"


class TestRegulatoryTracker:
    def test_creation(self, reg_tracker_mod):
        tracker = reg_tracker_mod.RegulatoryTracker()
        assert tracker is not None

    def test_get_recent_updates(self, reg_tracker_mod):
        tracker = reg_tracker_mod.RegulatoryTracker()
        updates = tracker.get_recent_updates(days=180)
        assert isinstance(updates, list)

    def test_get_upcoming_deadlines(self, reg_tracker_mod):
        tracker = reg_tracker_mod.RegulatoryTracker()
        deadlines = tracker.get_upcoming_deadlines(days_ahead=365)
        assert isinstance(deadlines, list)

    def test_generate_action_items(self, reg_tracker_mod):
        tracker = reg_tracker_mod.RegulatoryTracker()
        actions = tracker.generate_action_items()
        assert isinstance(actions, list)

    def test_generate_landscape_report(self, reg_tracker_mod):
        tracker = reg_tracker_mod.RegulatoryTracker()
        report = tracker.generate_landscape_report()
        assert isinstance(report, str)
        assert len(report) > 0
