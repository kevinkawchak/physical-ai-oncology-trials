"""Unit tests for tools/trial-site-monitor/trial_site_monitor.py.

Tests cover DEFAULT_THRESHOLDS, SITE_STATUS constants,
SiteMetrics dataclass, and metric computation.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "tools/trial-site-monitor/trial_site_monitor.py"


@pytest.fixture()
def mod():
    return load_module("trial_site_monitor", MOD_PATH)


class TestConstants:
    def test_default_thresholds(self, mod):
        t = mod.DEFAULT_THRESHOLDS
        assert "min_data_completeness_pct" in t
        assert "max_ae_reporting_delay_days" in t
        assert t["min_data_completeness_pct"] > 0

    def test_site_status_levels(self, mod):
        for level in ("green", "yellow", "red"):
            assert level in mod.SITE_STATUS


class TestSiteMetrics:
    def test_creation(self, mod):
        sm = mod.SiteMetrics(
            site_id="SITE-001",
            site_name="Memorial Cancer Center",
            status="green",
            total_screened=100,
            total_enrolled=45,
            total_completed=30,
            total_withdrawn=5,
        )
        assert sm.site_id == "SITE-001"
        assert sm.status == "green"

    def test_to_dict(self, mod):
        sm = mod.SiteMetrics(
            site_id="SITE-002",
            site_name="University Hospital",
            status="yellow",
            total_screened=80,
            total_enrolled=35,
            total_completed=20,
            total_withdrawn=3,
        )
        d = sm.to_dict()
        assert isinstance(d, dict)
        assert d["site_id"] == "SITE-002"

    def test_screen_failure_rate(self, mod):
        sm = mod.SiteMetrics(
            site_id="SITE-003",
            site_name="Test Site",
            status="green",
            total_screened=100,
            total_enrolled=25,
            total_completed=20,
            total_withdrawn=2,
            screen_failure_rate=0.75,
        )
        assert sm.screen_failure_rate == 0.75

    def test_enrollment_rate(self, mod):
        sm = mod.SiteMetrics(
            site_id="SITE-004",
            site_name="Research Center",
            status="green",
            total_screened=60,
            total_enrolled=30,
            total_completed=15,
            total_withdrawn=1,
            enrollment_rate_monthly=5.0,
        )
        assert sm.enrollment_rate_monthly == 5.0


class TestMetricComputation:
    def test_compute_site_metrics(self, mod):
        site_data = {
            "site_id": "SITE-COMP",
            "site_name": "Computation Test",
            "total_screened": 100,
            "total_enrolled": 50,
            "total_completed": 30,
            "total_withdrawn": 5,
            "data_completeness_pct": 92.0,
            "query_rate_per_subject": 0.5,
            "protocol_deviations": 3,
            "mean_ae_reporting_delay_days": 2.0,
            "days_since_last_enrollment": 10,
            "enrollment_rate_monthly": 5.0,
        }
        metrics = mod.compute_site_metrics(site_data, mod.DEFAULT_THRESHOLDS)
        assert isinstance(metrics, mod.SiteMetrics)
        assert metrics.site_id == "SITE-COMP"
