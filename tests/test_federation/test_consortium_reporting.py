"""Tests for federation.consortium_reporting module.

Covers EnrollmentDashboardGenerator, AdverseEventReporter,
SitePerformanceReporter, and DSMBPackageGenerator.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.consortium_reporting import (
    AdverseEvent,
    AdverseEventGrade,
    AdverseEventReporter,
    AERelationship,
    DSMBPackageGenerator,
    EnrollmentDashboardGenerator,
    SitePerformanceMetrics,
    SitePerformanceReporter,
    SiteRiskLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_site_metrics(n=3):
    """Create sample site metrics."""
    metrics = {}
    for i in range(n):
        sid = f"site_{i:03d}"
        metrics[sid] = SitePerformanceMetrics(
            site_id=sid,
            site_name=f"Hospital {chr(65 + i)}",
            enrollment_count=10 + i * 5,
            enrollment_target=30,
            enrollment_rate=2.0 + i * 0.5,
            data_completeness=0.95 - i * 0.02,
            query_rate=5 + i * 3,
            protocol_deviation_count=i,
        )
    return metrics


def _make_adverse_events(n=10):
    """Create sample adverse events."""
    aes = []
    for i in range(n):
        ae = AdverseEvent(
            ae_id=f"AE-{i:04d}",
            patient_id=f"PT-{i:04d}",
            site_id=f"site_{i % 3:03d}",
            term="Nausea" if i % 3 == 0 else ("Fatigue" if i % 3 == 1 else "Neutropenia"),
            system_organ_class="Gastrointestinal" if i % 3 == 0 else "General",
            grade=AdverseEventGrade(min(i % 4 + 1, 4)),
            relationship=AERelationship.POSSIBLE if i % 2 == 0 else AERelationship.UNRELATED,
            serious=(i % 5 == 0),
        )
        aes.append(ae)
    return aes


# ---------------------------------------------------------------------------
# EnrollmentDashboardGenerator tests
# ---------------------------------------------------------------------------


class TestEnrollmentDashboardGenerator:
    def test_generate_dashboard(self):
        gen = EnrollmentDashboardGenerator()
        metrics = _make_site_metrics()
        dashboard = gen.generate("TEST-001", metrics)
        assert dashboard.trial_id == "TEST-001"
        assert dashboard.total_enrolled == sum(m.enrollment_count for m in metrics.values())
        assert dashboard.total_target == sum(m.enrollment_target for m in metrics.values())

    def test_site_enrollments_populated(self):
        gen = EnrollmentDashboardGenerator()
        metrics = _make_site_metrics()
        dashboard = gen.generate("TEST", metrics)
        assert len(dashboard.site_enrollments) == 3

    def test_enrollment_curve(self):
        gen = EnrollmentDashboardGenerator()
        metrics = _make_site_metrics()
        dashboard = gen.generate("TEST", metrics)
        assert len(dashboard.enrollment_curve) > 0

    def test_projected_completion(self):
        gen = EnrollmentDashboardGenerator()
        metrics = _make_site_metrics()
        dashboard = gen.generate("TEST", metrics)
        assert len(dashboard.projected_completion) > 0

    def test_arm_distribution(self):
        gen = EnrollmentDashboardGenerator()
        metrics = _make_site_metrics()
        arms = {"experimental": 15, "control": 15}
        dashboard = gen.generate("TEST", metrics, arm_distribution=arms)
        assert dashboard.arm_distribution == arms


# ---------------------------------------------------------------------------
# AdverseEventReporter tests
# ---------------------------------------------------------------------------


class TestAdverseEventReporter:
    def test_summarize(self):
        reporter = AdverseEventReporter()
        aes = _make_adverse_events(10)
        summary = reporter.summarize(aes)
        assert summary["total_aes"] == 10
        assert "grade_distribution" in summary
        assert "soc_distribution" in summary

    def test_sae_count(self):
        reporter = AdverseEventReporter()
        aes = _make_adverse_events(10)
        summary = reporter.summarize(aes)
        expected_saes = sum(1 for ae in aes if ae.serious)
        assert summary["total_saes"] == expected_saes

    def test_treatment_related_count(self):
        reporter = AdverseEventReporter()
        aes = _make_adverse_events(10)
        summary = reporter.summarize(aes)
        assert summary["treatment_related"] >= 0

    def test_site_filter(self):
        reporter = AdverseEventReporter()
        aes = _make_adverse_events(10)
        summary = reporter.summarize(aes, sites=["site_000"])
        assert summary["total_aes"] <= 10

    def test_empty_aes(self):
        reporter = AdverseEventReporter()
        summary = reporter.summarize([])
        assert summary["total_aes"] == 0
        assert summary["sae_rate"] == 0.0


# ---------------------------------------------------------------------------
# SitePerformanceReporter tests
# ---------------------------------------------------------------------------


class TestSitePerformanceReporter:
    def test_compute_risk_low(self):
        reporter = SitePerformanceReporter()
        metrics = SitePerformanceMetrics(
            enrollment_count=25,
            enrollment_target=30,
            data_completeness=0.98,
            query_rate=3,
            protocol_deviation_count=0,
            ae_reporting_delay_days=1.0,
        )
        risk = reporter.compute_risk_level(metrics)
        assert risk == SiteRiskLevel.LOW

    def test_compute_risk_high(self):
        reporter = SitePerformanceReporter()
        metrics = SitePerformanceMetrics(
            enrollment_count=5,
            enrollment_target=50,
            data_completeness=0.75,
            query_rate=25,
            protocol_deviation_count=6,
            ae_reporting_delay_days=10,
        )
        risk = reporter.compute_risk_level(metrics)
        assert risk in (SiteRiskLevel.HIGH, SiteRiskLevel.CRITICAL)

    def test_generate_report(self):
        reporter = SitePerformanceReporter()
        metrics = _make_site_metrics()
        report = reporter.generate_report(metrics)
        assert "sites" in report
        assert "overall" in report
        assert report["overall"]["total_sites"] == 3


# ---------------------------------------------------------------------------
# DSMBPackageGenerator tests
# ---------------------------------------------------------------------------


class TestDSMBPackageGenerator:
    def test_generate_package(self):
        gen = DSMBPackageGenerator()
        metrics = _make_site_metrics()
        aes = _make_adverse_events(15)
        package = gen.generate(
            trial_id="DSMB-TEST",
            review_number=1,
            site_metrics=metrics,
            adverse_events=aes,
        )
        assert package.trial_id == "DSMB-TEST"
        assert package.review_number == 1
        assert package.enrollment_summary is not None
        assert package.ae_summary["total_aes"] == 15

    def test_recommendations_present(self):
        gen = DSMBPackageGenerator()
        metrics = _make_site_metrics()
        aes = _make_adverse_events(10)
        package = gen.generate("TEST", 1, metrics, aes)
        assert len(package.recommendations) > 0

    def test_safety_signals_detection(self):
        gen = DSMBPackageGenerator()
        metrics = _make_site_metrics()
        aes = []
        for i in range(10):
            aes.append(
                AdverseEvent(
                    ae_id=f"AE-{i}",
                    site_id="site_000",
                    grade=AdverseEventGrade.GRADE_4,
                    serious=True,
                )
            )
        package = gen.generate("TEST", 1, metrics, aes)
        assert len(package.safety_signals) > 0
