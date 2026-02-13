"""Tests for federation.privacy_analytics module.

Covers SiteSurvivalAggregator, FederatedSurvivalAnalyzer,
KaplanMeierResult, CoxPHResult, and response rate computation.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.privacy_analytics import (
    CoxPHResult,
    FederatedSurvivalAnalyzer,
    KaplanMeierResult,
    PrivacyAnalyticsConfig,
    SiteSurvivalAggregator,
    SurvivalRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(site_id: str, n: int = 20) -> list:
    """Generate synthetic survival records."""
    records = []
    for i in range(n):
        records.append(
            SurvivalRecord(
                patient_id=f"PT-{site_id}-{i:03d}",
                site_id=site_id,
                time=np.random.exponential(12) + 0.5,
                event=np.random.random() > 0.3,
                treatment_arm="experimental" if i % 2 == 0 else "control",
                covariates={"age": 50 + np.random.randn() * 10, "stage": float(np.random.randint(2, 5))},
            )
        )
    return records


# ---------------------------------------------------------------------------
# SiteSurvivalAggregator tests
# ---------------------------------------------------------------------------


class TestSiteSurvivalAggregator:
    def test_add_records(self):
        agg = SiteSurvivalAggregator("s1")
        records = _make_records("s1", 10)
        agg.add_records(records)
        assert len(agg.records) == 10

    def test_compute_local_statistics(self):
        agg = SiteSurvivalAggregator("s1")
        agg.add_records(_make_records("s1", 20))
        time_points = [1.0, 5.0, 10.0, 20.0]
        stats = agg.compute_local_statistics(time_points)
        assert stats["site_id"] == "s1"
        assert len(stats["at_risk"]) == 4
        assert len(stats["events"]) == 4
        assert stats["total_patients"] == 20

    def test_compute_covariate_statistics(self):
        agg = SiteSurvivalAggregator("s1")
        agg.add_records(_make_records("s1", 15))
        stats = agg.compute_local_covariate_statistics()
        assert stats["n"] == 15
        assert "age" in stats["covariate_names"]
        assert "stage" in stats["covariate_names"]
        assert len(stats["means"]) == 2

    def test_empty_records(self):
        agg = SiteSurvivalAggregator("s1")
        stats = agg.compute_local_statistics([1.0, 5.0])
        assert stats["total_patients"] == 0
        assert all(v == 0 for v in stats["at_risk"])


# ---------------------------------------------------------------------------
# FederatedSurvivalAnalyzer tests
# ---------------------------------------------------------------------------


class TestFederatedSurvivalAnalyzer:
    def _setup_analyzer(self, n_sites=3, n_patients=20):
        config = PrivacyAnalyticsConfig(epsilon=1.0, delta=1e-5, min_cell_size=2)
        analyzer = FederatedSurvivalAnalyzer(config)
        for i in range(n_sites):
            site_id = f"site_{i}"
            agg = analyzer.register_site(site_id)
            agg.add_records(_make_records(site_id, n_patients))
        return analyzer

    def test_register_site(self):
        config = PrivacyAnalyticsConfig()
        analyzer = FederatedSurvivalAnalyzer(config)
        agg = analyzer.register_site("s1")
        assert isinstance(agg, SiteSurvivalAggregator)
        assert "s1" in analyzer.site_aggregators

    def test_common_time_points(self):
        analyzer = self._setup_analyzer()
        time_points = analyzer.compute_common_time_points()
        assert len(time_points) > 0
        assert time_points == sorted(time_points)

    def test_kaplan_meier(self):
        analyzer = self._setup_analyzer()
        result = analyzer.federated_kaplan_meier("all")
        assert isinstance(result, KaplanMeierResult)
        assert result.analysis_group == "all"
        assert len(result.time_points) > 0
        assert len(result.survival_probs) == len(result.time_points)
        assert all(0 <= s <= 1 for s in result.survival_probs)

    def test_kaplan_meier_arm_filter(self):
        analyzer = self._setup_analyzer()
        result = analyzer.federated_kaplan_meier("exp", arm_filter="experimental")
        assert result.analysis_group == "exp"
        assert len(result.time_points) > 0

    def test_kaplan_meier_survival_decreasing(self):
        analyzer = self._setup_analyzer(n_patients=50)
        result = analyzer.federated_kaplan_meier("test")
        probs = result.survival_probs
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i - 1] + 0.01

    def test_kaplan_meier_confidence_intervals(self):
        analyzer = self._setup_analyzer()
        result = analyzer.federated_kaplan_meier("test")
        for i in range(len(result.ci_lower)):
            assert result.ci_lower[i] <= result.survival_probs[i] + 0.001
            assert result.ci_upper[i] >= result.survival_probs[i] - 0.001

    def test_cox_ph(self):
        analyzer = self._setup_analyzer()
        result = analyzer.federated_cox_ph(["age", "stage"])
        assert isinstance(result, CoxPHResult)
        assert "age" in result.coefficients
        assert "stage" in result.coefficients
        assert "age" in result.hazard_ratios
        assert all(hr > 0 for hr in result.hazard_ratios.values())
        assert 0 <= result.concordance_index <= 1

    def test_cox_ph_no_data(self):
        config = PrivacyAnalyticsConfig()
        analyzer = FederatedSurvivalAnalyzer(config)
        result = analyzer.federated_cox_ph()
        assert isinstance(result, CoxPHResult)

    def test_response_rate(self):
        analyzer = self._setup_analyzer(n_patients=30)
        rr = analyzer.compute_response_rate("experimental")
        assert not rr.get("suppressed", True)
        assert 0 <= rr["response_rate"] <= 1
        assert rr["ci_lower"] <= rr["response_rate"]
        assert rr["ci_upper"] >= rr["response_rate"]

    def test_response_rate_suppression(self):
        config = PrivacyAnalyticsConfig(min_cell_size=1000)
        analyzer = FederatedSurvivalAnalyzer(config)
        agg = analyzer.register_site("s1")
        agg.add_records(_make_records("s1", 5))
        rr = analyzer.compute_response_rate("experimental")
        assert rr.get("suppressed", False)

    def test_empty_analyzer(self):
        config = PrivacyAnalyticsConfig()
        analyzer = FederatedSurvivalAnalyzer(config)
        result = analyzer.federated_kaplan_meier("empty")
        assert len(result.time_points) == 0
