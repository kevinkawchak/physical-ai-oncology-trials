from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("trial_site_monitor", "tools/trial-site-monitor/trial_site_monitor.py")


@pytest.fixture()
def sample_site():
    """A sample site dictionary for testing compute_site_metrics."""
    return {
        "site_id": "SITE-001",
        "site_name": "Test Cancer Center",
        "activation_date": "2025-01-01",
        "enrollment": {
            "screened": 50,
            "enrolled": 20,
            "completed": 10,
            "withdrawn": 2,
            "last_enrollment_date": "2026-01-15",
        },
        "data_quality": {
            "completeness_pct": 95.0,
            "open_queries": 10,
        },
        "protocol_deviations": [
            {"category": "timing", "severity": "minor", "description": "Late visit", "date": "2025-06-01"},
            {"category": "dosing", "severity": "major", "description": "Wrong dose", "date": "2025-07-15"},
        ],
        "adverse_events": {
            "total_reported": 5,
            "serious": 1,
            "mean_reporting_delay_days": 2.0,
        },
    }


# ---------------------------------------------------------------------------
# TestSiteMetrics
# ---------------------------------------------------------------------------


class TestSiteMetrics:
    """Tests for the SiteMetrics dataclass."""

    def test_creation_required_fields(self, mod):
        metrics = mod.SiteMetrics(site_id="SITE-001", site_name="Test Site")
        assert metrics.site_id == "SITE-001"
        assert metrics.site_name == "Test Site"

    def test_default_status_green(self, mod):
        metrics = mod.SiteMetrics(site_id="S", site_name="N")
        assert metrics.status == "green"

    def test_default_enrollment_zeros(self, mod):
        metrics = mod.SiteMetrics(site_id="S", site_name="N")
        assert metrics.total_screened == 0
        assert metrics.total_enrolled == 0
        assert metrics.total_completed == 0
        assert metrics.total_withdrawn == 0

    def test_default_flags_empty(self, mod):
        metrics = mod.SiteMetrics(site_id="S", site_name="N")
        assert metrics.flags == []

    def test_to_dict_returns_dict(self, mod):
        metrics = mod.SiteMetrics(site_id="S", site_name="N")
        d = metrics.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_site_id(self, mod):
        metrics = mod.SiteMetrics(site_id="SITE-042", site_name="N")
        d = metrics.to_dict()
        assert d["site_id"] == "SITE-042"

    def test_to_dict_screen_failure_rate_rounded(self, mod):
        metrics = mod.SiteMetrics(site_id="S", site_name="N", screen_failure_rate=0.333333)
        d = metrics.to_dict()
        assert d["screen_failure_rate"] == 0.333

    def test_to_dict_all_keys(self, mod):
        metrics = mod.SiteMetrics(site_id="S", site_name="N")
        d = metrics.to_dict()
        expected = {
            "site_id",
            "site_name",
            "status",
            "total_screened",
            "total_enrolled",
            "total_completed",
            "total_withdrawn",
            "screen_failure_rate",
            "enrollment_rate_monthly",
            "data_completeness_pct",
            "query_rate_per_subject",
            "protocol_deviations",
            "deviation_rate_per_subject",
            "mean_ae_reporting_delay_days",
            "days_since_last_enrollment",
            "flags",
        }
        assert expected == set(d.keys())


# ---------------------------------------------------------------------------
# TestComputeSiteMetrics
# ---------------------------------------------------------------------------


class TestComputeSiteMetrics:
    """Tests for the compute_site_metrics function."""

    def test_compute_basic(self, mod, sample_site):
        thresholds = mod.DEFAULT_THRESHOLDS
        metrics = mod.compute_site_metrics(sample_site, thresholds)
        assert metrics.site_id == "SITE-001"
        assert metrics.total_screened == 50
        assert metrics.total_enrolled == 20

    def test_screen_failure_rate(self, mod, sample_site):
        thresholds = mod.DEFAULT_THRESHOLDS
        metrics = mod.compute_site_metrics(sample_site, thresholds)
        expected_rate = 1.0 - (20 / 50)
        assert abs(metrics.screen_failure_rate - expected_rate) < 0.01

    def test_protocol_deviations_counted(self, mod, sample_site):
        thresholds = mod.DEFAULT_THRESHOLDS
        metrics = mod.compute_site_metrics(sample_site, thresholds)
        assert metrics.protocol_deviations == 2

    def test_data_completeness(self, mod, sample_site):
        thresholds = mod.DEFAULT_THRESHOLDS
        metrics = mod.compute_site_metrics(sample_site, thresholds)
        assert metrics.data_completeness_pct == 95.0

    def test_query_rate(self, mod, sample_site):
        thresholds = mod.DEFAULT_THRESHOLDS
        metrics = mod.compute_site_metrics(sample_site, thresholds)
        assert metrics.query_rate_per_subject == 10 / 20

    def test_ae_reporting_delay(self, mod, sample_site):
        thresholds = mod.DEFAULT_THRESHOLDS
        metrics = mod.compute_site_metrics(sample_site, thresholds)
        assert metrics.mean_ae_reporting_delay_days == 2.0


# ---------------------------------------------------------------------------
# TestSiteStatusClassification
# ---------------------------------------------------------------------------


class TestSiteStatusClassification:
    """Tests for site status classification logic."""

    def test_green_status_no_flags(self, mod):
        site = {
            "site_id": "S",
            "site_name": "N",
            "activation_date": "2025-01-01",
            "enrollment": {"screened": 100, "enrolled": 50, "completed": 30, "withdrawn": 0},
            "data_quality": {"completeness_pct": 99.0, "open_queries": 0},
            "protocol_deviations": [],
            "adverse_events": {"mean_reporting_delay_days": 1.0},
        }
        metrics = mod.compute_site_metrics(site, mod.DEFAULT_THRESHOLDS)
        # If enrollment rate is acceptable, status should be green or yellow
        # Depends on how many flags are triggered
        assert metrics.status in ("green", "yellow")

    def test_red_status_many_issues(self, mod):
        site = {
            "site_id": "S",
            "site_name": "N",
            "activation_date": "2025-01-01",
            "enrollment": {
                "screened": 100,
                "enrolled": 5,
                "completed": 0,
                "withdrawn": 0,
                "last_enrollment_date": "2024-01-01",
            },
            "data_quality": {"completeness_pct": 50.0, "open_queries": 100},
            "protocol_deviations": [1, 2, 3, 4, 5],
            "adverse_events": {"mean_reporting_delay_days": 10.0},
        }
        metrics = mod.compute_site_metrics(site, mod.DEFAULT_THRESHOLDS)
        assert metrics.status == "red"

    def test_site_status_dict_exists(self, mod):
        assert hasattr(mod, "SITE_STATUS")
        assert "green" in mod.SITE_STATUS
        assert "yellow" in mod.SITE_STATUS
        assert "red" in mod.SITE_STATUS


# ---------------------------------------------------------------------------
# TestDefaultThresholds
# ---------------------------------------------------------------------------


class TestDefaultThresholds:
    """Tests for DEFAULT_THRESHOLDS constant."""

    def test_thresholds_exist(self, mod):
        assert hasattr(mod, "DEFAULT_THRESHOLDS")
        assert isinstance(mod.DEFAULT_THRESHOLDS, dict)

    def test_thresholds_have_expected_keys(self, mod):
        expected = {
            "min_screening_to_enrollment_ratio",
            "max_screen_failure_rate",
            "max_protocol_deviation_rate_per_subject",
            "min_data_completeness_pct",
            "max_query_rate_per_subject",
            "max_days_enrollment_gap",
            "min_monthly_enrollment_rate",
            "max_ae_reporting_delay_days",
        }
        assert expected == set(mod.DEFAULT_THRESHOLDS.keys())

    def test_thresholds_values_positive(self, mod):
        for key, value in mod.DEFAULT_THRESHOLDS.items():
            assert value > 0, f"Threshold '{key}' should be positive"
