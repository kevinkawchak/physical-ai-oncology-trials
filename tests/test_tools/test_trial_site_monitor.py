"""Tests for tools/trial-site-monitor/trial_site_monitor.py

Covers SiteMetrics dataclass and site metric computation.
"""

from __future__ import annotations


class TestSiteMetrics:
    def test_creation_defaults(self, site_monitor_mod):
        metrics = site_monitor_mod.SiteMetrics(
            site_id="SITE-001",
            site_name="Test Site",
        )
        assert metrics.site_id == "SITE-001"
        assert metrics.status == "green"
        assert metrics.data_completeness_pct == 100.0

    def test_to_dict(self, site_monitor_mod):
        metrics = site_monitor_mod.SiteMetrics(
            site_id="SITE-002",
            site_name="Clinical Center A",
            total_enrolled=50,
            total_screened=100,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["site_id"] == "SITE-002"
        assert d["total_enrolled"] == 50

    def test_with_quality_flags(self, site_monitor_mod):
        metrics = site_monitor_mod.SiteMetrics(
            site_id="SITE-003",
            site_name="Flagged Site",
            status="red",
            flags=["low_enrollment", "high_query_rate"],
        )
        assert metrics.status == "red"
        assert len(metrics.flags) == 2


class TestComputeSiteMetrics:
    def test_compute_from_manifest(self, site_monitor_mod):
        site_data = {
            "site_id": "SITE-010",
            "site_name": "Test Hospital",
            "enrollment": {
                "screened": 80,
                "enrolled": 40,
                "completed": 30,
                "withdrawn": 5,
            },
            "data_quality": {
                "completeness_pct": 95.0,
                "open_queries": 12,
            },
            "protocol_deviations": [
                {"id": "PD-001", "description": "deviation 1"},
                {"id": "PD-002", "description": "deviation 2"},
            ],
            "adverse_events": {
                "mean_reporting_delay_days": 1.5,
            },
        }
        thresholds = site_monitor_mod.DEFAULT_THRESHOLDS
        metrics = site_monitor_mod.compute_site_metrics(site_data, thresholds)
        assert isinstance(metrics, site_monitor_mod.SiteMetrics)
        assert metrics.site_id == "SITE-010"
