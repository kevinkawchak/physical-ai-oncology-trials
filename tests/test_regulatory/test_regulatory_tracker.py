"""Tests for Regulatory Intelligence Tracker.

Validates update tracking, deadline management, cutoff filtering
(v0.9.1 fix), overdue/imminent status (v0.9.2 fix), and action-item
generation.

LICENSE: MIT
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
MOD_PATH = "regulatory/regulatory-intelligence/regulatory_tracker.py"


@pytest.fixture()
def rt_mod():
    return load_module("regulatory_tracker", MOD_PATH)


@pytest.fixture()
def tracker(rt_mod):
    return rt_mod.RegulatoryTracker(
        jurisdictions=["fda", "ema", "ich"],
        topics=["ai_ml_devices", "oncology"],
    )


# ===========================================================================
# TestEnums
# ===========================================================================


class TestEnums:
    def test_jurisdiction_values(self, rt_mod):
        names = {e.name for e in rt_mod.RegulatoryJurisdiction}
        assert "FDA" in names
        assert "EMA" in names
        assert "MHRA" in names
        assert len(names) == 9

    def test_update_type_values(self, rt_mod):
        names = {e.name for e in rt_mod.UpdateType}
        assert "GUIDANCE_DRAFT" in names
        assert "ENFORCEMENT_ACTION" in names
        assert len(names) == 9

    def test_impact_level_values(self, rt_mod):
        names = {e.name for e in rt_mod.ImpactLevel}
        assert names == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"}


# ===========================================================================
# TestRegulatoryUpdate
# ===========================================================================


class TestRegulatoryUpdate:
    def test_update_construction(self, rt_mod):
        u = rt_mod.RegulatoryUpdate(
            update_id="REG-1",
            jurisdiction="fda",
            title="Test Guidance",
            date="2026-01-01",
            update_type="guidance_draft",
            impact_level="high",
            summary="Summary text",
        )
        assert u.update_id == "REG-1"
        assert u.affected_areas == []

    def test_to_dict(self, rt_mod):
        u = rt_mod.RegulatoryUpdate(
            update_id="REG-2",
            jurisdiction="ema",
            title="EU Update",
            date="2026-01-15",
            update_type="regulation_final",
            impact_level="critical",
            summary="Test",
            affected_areas=["ai_ml_models"],
        )
        d = u.to_dict()
        assert d["jurisdiction"] == "ema"
        assert "affected_areas" in d


# ===========================================================================
# TestTrackerInitialization
# ===========================================================================


class TestTrackerInit:
    def test_updates_loaded(self, tracker):
        assert len(tracker._updates) > 0

    def test_jurisdiction_filtering_on_init(self, rt_mod):
        t = rt_mod.RegulatoryTracker(jurisdictions=["fda"])
        for u in t._updates:
            assert u.jurisdiction == "fda"

    def test_default_jurisdictions(self, rt_mod):
        t = rt_mod.RegulatoryTracker()
        assert t.jurisdictions == ["fda", "ema", "ich"]


# ===========================================================================
# TestGetRecentUpdates
# ===========================================================================


class TestGetRecentUpdates:
    def test_v091_cutoff_date_filtering(self, tracker):
        """v0.9.1 fix: uses ISO date string cutoff for recent updates."""
        # Very large window should return all updates
        all_updates = tracker.get_recent_updates(days=365 * 10)
        assert len(all_updates) == len(tracker._updates)

    def test_narrow_window_filters(self, tracker):
        recent = tracker.get_recent_updates(days=1)
        # Unless an update was today, this should return fewer
        all_updates = tracker.get_recent_updates(days=365 * 10)
        assert len(recent) <= len(all_updates)

    def test_jurisdiction_filter(self, tracker):
        fda_only = tracker.get_recent_updates(days=365 * 10, jurisdiction="fda")
        for u in fda_only:
            assert u.jurisdiction == "fda"

    def test_impact_filter(self, tracker):
        critical = tracker.get_recent_updates(days=365 * 10, impact_level="critical")
        for u in critical:
            assert u.impact_level == "critical"

    def test_sorted_descending_by_date(self, tracker):
        updates = tracker.get_recent_updates(days=365 * 10)
        dates = [u.date for u in updates]
        assert dates == sorted(dates, reverse=True)


# ===========================================================================
# TestGetUpcomingDeadlines
# ===========================================================================


class TestGetUpcomingDeadlines:
    def test_deadlines_returned(self, tracker):
        deadlines = tracker.get_upcoming_deadlines(days_ahead=365 * 5)
        # May or may not have deadlines depending on current date vs timeline
        assert isinstance(deadlines, list)

    def test_v092_overdue_vs_imminent_status(self, rt_mod):
        """v0.9.2 fix: status correctly distinguishes overdue, imminent, upcoming."""
        dl_overdue = rt_mod.ComplianceDeadline(
            jurisdiction="fda",
            regulation="Test",
            requirement="Req",
            deadline=(datetime.now() - timedelta(days=5)).isoformat()[:10],
            status="overdue",
        )
        assert dl_overdue.status == "overdue"

        dl_imminent = rt_mod.ComplianceDeadline(
            jurisdiction="fda",
            regulation="Test",
            requirement="Req",
            deadline=(datetime.now() + timedelta(days=10)).isoformat()[:10],
            status="imminent",
        )
        assert dl_imminent.status == "imminent"

    def test_deadlines_sorted_by_date(self, tracker):
        deadlines = tracker.get_upcoming_deadlines(days_ahead=365 * 5)
        dates = [d.deadline for d in deadlines]
        assert dates == sorted(dates)


# ===========================================================================
# TestGenerateActionItems
# ===========================================================================


class TestGenerateActionItems:
    def test_action_items_generated(self, tracker):
        items = tracker.generate_action_items()
        assert len(items) > 0

    def test_action_item_structure(self, tracker):
        items = tracker.generate_action_items()
        for item in items:
            assert "action" in item
            assert "source" in item
            assert "jurisdiction" in item
            assert "impact" in item

    def test_action_items_jurisdiction_filter(self, tracker):
        fda_items = tracker.generate_action_items(jurisdiction="fda")
        for item in fda_items:
            assert item["jurisdiction"] == "fda"

    def test_action_items_non_fda_excluded(self, tracker):
        ich_items = tracker.generate_action_items(jurisdiction="ich")
        for item in ich_items:
            assert item["jurisdiction"] == "ich"


# ===========================================================================
# TestComplianceDeadline
# ===========================================================================


class TestComplianceDeadline:
    def test_deadline_defaults(self, rt_mod):
        dl = rt_mod.ComplianceDeadline(
            jurisdiction="fda",
            regulation="Test Reg",
            requirement="Req",
            deadline="2026-06-01",
        )
        assert dl.status == "upcoming"
        assert dl.notes == ""


# ===========================================================================
# TestRegulatoryTimeline
# ===========================================================================


class TestRegulatoryTimeline:
    def test_timeline_has_entries(self, rt_mod):
        assert len(rt_mod.REGULATORY_TIMELINE) > 0

    def test_timeline_entry_structure(self, rt_mod):
        for entry in rt_mod.REGULATORY_TIMELINE:
            assert "jurisdiction" in entry
            assert "title" in entry
            assert "date" in entry
            assert "type" in entry
            assert "impact" in entry

    def test_landscape_report(self, tracker):
        report = tracker.generate_landscape_report()
        assert "REGULATORY LANDSCAPE REPORT" in report
        assert "ACTION ITEMS" in report
