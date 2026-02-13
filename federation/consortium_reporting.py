#!/usr/bin/env python3
"""Consortium reporting engine for multi-site oncology trials.

Generates site-level and aggregate trial status reports, enrollment
dashboards, adverse event summaries, and Data Safety Monitoring Board
(DSMB) packages for federated oncology trial coordination.

Framework Dependencies:
    - NumPy 1.24.0+

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    All reports must be reviewed by qualified biostatisticians and
    clinical investigators before regulatory submission.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReportType(Enum):
    """Types of consortium reports."""

    ENROLLMENT_DASHBOARD = "enrollment_dashboard"
    ADVERSE_EVENT_SUMMARY = "adverse_event_summary"
    DSMB_PACKAGE = "dsmb_package"
    SITE_PERFORMANCE = "site_performance"
    DATA_QUALITY = "data_quality"
    INTERIM_ANALYSIS = "interim_analysis"


class AdverseEventGrade(Enum):
    """CTCAE v5.0 adverse event grades."""

    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    GRADE_5 = 5


class AERelationship(Enum):
    """Relationship of adverse event to study treatment."""

    UNRELATED = "unrelated"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    PROBABLE = "probable"
    DEFINITE = "definite"


class SiteRiskLevel(Enum):
    """Risk-based monitoring site classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class AdverseEvent:
    """Individual adverse event record.

    Attributes:
        ae_id: Unique adverse event identifier.
        patient_id: De-identified patient ID.
        site_id: Reporting site.
        term: MedDRA preferred term.
        system_organ_class: System Organ Class.
        grade: CTCAE grade.
        relationship: Relationship to study treatment.
        serious: Whether this is a Serious AE (SAE).
        onset_day: Study day of onset.
        resolution_day: Study day of resolution (0 if ongoing).
        action_taken: Action taken (dose reduction, discontinuation, etc.).
        outcome: Outcome (resolved, ongoing, fatal, etc.).
    """

    ae_id: str = ""
    patient_id: str = ""
    site_id: str = ""
    term: str = ""
    system_organ_class: str = ""
    grade: AdverseEventGrade = AdverseEventGrade.GRADE_1
    relationship: AERelationship = AERelationship.UNRELATED
    serious: bool = False
    onset_day: int = 0
    resolution_day: int = 0
    action_taken: str = ""
    outcome: str = "resolved"


@dataclass
class SitePerformanceMetrics:
    """Performance metrics for a single site.

    Attributes:
        site_id: Site identifier.
        site_name: Human-readable site name.
        enrollment_count: Current enrollment.
        enrollment_target: Target enrollment.
        enrollment_rate: Weekly enrollment rate.
        screen_fail_rate: Screen failure percentage.
        data_completeness: CRF data completeness (0–1).
        query_rate: Data queries per 100 CRFs.
        protocol_deviation_count: Number of protocol deviations.
        ae_reporting_delay_days: Mean AE reporting delay.
        risk_level: Computed risk level.
    """

    site_id: str = ""
    site_name: str = ""
    enrollment_count: int = 0
    enrollment_target: int = 50
    enrollment_rate: float = 0.0
    screen_fail_rate: float = 0.0
    data_completeness: float = 0.95
    query_rate: float = 0.0
    protocol_deviation_count: int = 0
    ae_reporting_delay_days: float = 0.0
    risk_level: SiteRiskLevel = SiteRiskLevel.LOW


@dataclass
class EnrollmentDashboard:
    """Enrollment dashboard data model.

    Attributes:
        dashboard_id: Unique dashboard identifier.
        trial_id: Trial identifier.
        total_enrolled: Total patients enrolled.
        total_target: Total enrollment target.
        site_enrollments: Per-site enrollment data.
        arm_distribution: Enrollment by treatment arm.
        enrollment_curve: Cumulative enrollment over time.
        projected_completion: Projected completion date.
    """

    dashboard_id: str = ""
    trial_id: str = ""
    total_enrolled: int = 0
    total_target: int = 0
    site_enrollments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arm_distribution: Dict[str, int] = field(default_factory=dict)
    enrollment_curve: List[Dict[str, Any]] = field(default_factory=list)
    projected_completion: str = ""


@dataclass
class DSMBPackage:
    """Data Safety Monitoring Board report package.

    Attributes:
        package_id: Unique package identifier.
        trial_id: Trial identifier.
        review_number: Sequential DSMB review number.
        enrollment_summary: Enrollment dashboard.
        ae_summary: Adverse event summary.
        efficacy_data: Interim efficacy data.
        safety_signals: Identified safety signals.
        recommendations: DSMB recommendations.
        statistical_analysis: Statistical analysis results.
    """

    package_id: str = ""
    trial_id: str = ""
    review_number: int = 1
    enrollment_summary: Optional[EnrollmentDashboard] = None
    ae_summary: Dict[str, Any] = field(default_factory=dict)
    efficacy_data: Dict[str, Any] = field(default_factory=dict)
    safety_signals: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report Generators
# ---------------------------------------------------------------------------


class EnrollmentDashboardGenerator:
    """Generates enrollment dashboard reports.

    Creates visual-ready enrollment data including cumulative
    enrollment curves, site-level breakdowns, and projections.
    """

    def generate(
        self,
        trial_id: str,
        site_data: Dict[str, SitePerformanceMetrics],
        arm_distribution: Optional[Dict[str, int]] = None,
    ) -> EnrollmentDashboard:
        """Generate an enrollment dashboard.

        Args:
            trial_id: Trial identifier.
            site_data: Per-site performance metrics.
            arm_distribution: Optional arm distribution override.

        Returns:
            EnrollmentDashboard with current enrollment state.
        """
        dashboard = EnrollmentDashboard(
            dashboard_id=str(uuid.uuid4())[:8],
            trial_id=trial_id,
        )

        for site_id, metrics in site_data.items():
            dashboard.total_enrolled += metrics.enrollment_count
            dashboard.total_target += metrics.enrollment_target
            pct = (metrics.enrollment_count / metrics.enrollment_target * 100) if metrics.enrollment_target > 0 else 0
            dashboard.site_enrollments[site_id] = {
                "name": metrics.site_name,
                "enrolled": metrics.enrollment_count,
                "target": metrics.enrollment_target,
                "pct": round(pct, 1),
                "rate": metrics.enrollment_rate,
                "risk": metrics.risk_level.value,
            }

        if arm_distribution is not None:
            dashboard.arm_distribution = dict(arm_distribution)

        remaining = dashboard.total_target - dashboard.total_enrolled
        total_rate = sum(m.enrollment_rate for m in site_data.values())
        if total_rate > 0 and remaining > 0:
            weeks_remaining = remaining / total_rate
            dashboard.projected_completion = f"{weeks_remaining:.0f} weeks remaining"
        elif remaining <= 0:
            dashboard.projected_completion = "Target reached"
        else:
            dashboard.projected_completion = "Insufficient enrollment rate data"

        dashboard.enrollment_curve = self._generate_enrollment_curve(dashboard.total_enrolled)

        logger.info(
            "Enrollment dashboard: %d/%d (%.1f%%)",
            dashboard.total_enrolled,
            dashboard.total_target,
            (dashboard.total_enrolled / dashboard.total_target * 100) if dashboard.total_target > 0 else 0,
        )
        return dashboard

    def _generate_enrollment_curve(self, current_total: int) -> List[Dict[str, Any]]:
        """Generate simulated cumulative enrollment curve."""
        if current_total == 0:
            return []

        curve = []
        for week in range(1, 13):
            fraction = min(1.0, week / 12.0)
            cumulative = int(current_total * fraction)
            curve.append({"week": week, "cumulative": cumulative})
        return curve


class AdverseEventReporter:
    """Generates adverse event summary reports.

    Produces tabulated AE summaries by grade, system organ class,
    treatment arm, and site for DSMB review.
    """

    def summarize(self, adverse_events: List[AdverseEvent], sites: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate AE summary statistics.

        Args:
            adverse_events: List of adverse events.
            sites: Optional filter for specific sites.

        Returns:
            Dictionary with AE summary tables.
        """
        filtered = adverse_events
        if sites is not None:
            filtered = [ae for ae in adverse_events if ae.site_id in sites]

        total = len(filtered)
        sae_count = sum(1 for ae in filtered if ae.serious)

        grade_dist = {}
        for ae in filtered:
            grade_key = f"grade_{ae.grade.value}"
            grade_dist[grade_key] = grade_dist.get(grade_key, 0) + 1

        soc_dist: Dict[str, int] = {}
        for ae in filtered:
            soc_dist[ae.system_organ_class] = soc_dist.get(ae.system_organ_class, 0) + 1

        relationship_dist: Dict[str, int] = {}
        for ae in filtered:
            relationship_dist[ae.relationship.value] = relationship_dist.get(ae.relationship.value, 0) + 1

        site_dist: Dict[str, int] = {}
        for ae in filtered:
            site_dist[ae.site_id] = site_dist.get(ae.site_id, 0) + 1

        related_count = sum(
            1
            for ae in filtered
            if ae.relationship in (AERelationship.POSSIBLE, AERelationship.PROBABLE, AERelationship.DEFINITE)
        )

        return {
            "total_aes": total,
            "total_saes": sae_count,
            "treatment_related": related_count,
            "grade_distribution": grade_dist,
            "soc_distribution": soc_dist,
            "relationship_distribution": relationship_dist,
            "site_distribution": site_dist,
            "sae_rate": round(sae_count / total * 100, 1) if total > 0 else 0.0,
        }


class SitePerformanceReporter:
    """Generates site performance and risk-based monitoring reports."""

    def compute_risk_level(self, metrics: SitePerformanceMetrics) -> SiteRiskLevel:
        """Compute risk level for a site based on performance metrics.

        Args:
            metrics: Site performance metrics.

        Returns:
            Computed SiteRiskLevel.
        """
        risk_score = 0

        if metrics.enrollment_target > 0:
            enrollment_pct = metrics.enrollment_count / metrics.enrollment_target
            if enrollment_pct < 0.25:
                risk_score += 3
            elif enrollment_pct < 0.5:
                risk_score += 1

        if metrics.data_completeness < 0.80:
            risk_score += 3
        elif metrics.data_completeness < 0.90:
            risk_score += 1

        if metrics.query_rate > 20:
            risk_score += 2
        elif metrics.query_rate > 10:
            risk_score += 1

        if metrics.protocol_deviation_count > 5:
            risk_score += 2
        elif metrics.protocol_deviation_count > 2:
            risk_score += 1

        if metrics.ae_reporting_delay_days > 7:
            risk_score += 2
        elif metrics.ae_reporting_delay_days > 3:
            risk_score += 1

        if risk_score >= 8:
            return SiteRiskLevel.CRITICAL
        elif risk_score >= 5:
            return SiteRiskLevel.HIGH
        elif risk_score >= 3:
            return SiteRiskLevel.MEDIUM
        return SiteRiskLevel.LOW

    def generate_report(self, site_metrics: Dict[str, SitePerformanceMetrics]) -> Dict[str, Any]:
        """Generate a site performance report for all sites.

        Args:
            site_metrics: Per-site performance metrics.

        Returns:
            Performance report with risk assessments.
        """
        report = {"sites": {}, "overall": {}}

        for site_id, metrics in site_metrics.items():
            risk = self.compute_risk_level(metrics)
            metrics.risk_level = risk
            report["sites"][site_id] = {
                "name": metrics.site_name,
                "enrollment": f"{metrics.enrollment_count}/{metrics.enrollment_target}",
                "data_completeness": f"{metrics.data_completeness * 100:.1f}%",
                "query_rate": metrics.query_rate,
                "deviations": metrics.protocol_deviation_count,
                "risk_level": risk.value,
            }

        risk_counts = {}
        for site_id, metrics in site_metrics.items():
            level = metrics.risk_level.value
            risk_counts[level] = risk_counts.get(level, 0) + 1

        report["overall"] = {
            "total_sites": len(site_metrics),
            "risk_distribution": risk_counts,
            "mean_completeness": np.mean([m.data_completeness for m in site_metrics.values()]) if site_metrics else 0.0,
        }

        return report


# ---------------------------------------------------------------------------
# DSMB Package Generator
# ---------------------------------------------------------------------------


class DSMBPackageGenerator:
    """Generates Data Safety Monitoring Board report packages.

    Combines enrollment, safety, and efficacy data into standardized
    DSMB review packages.
    """

    def __init__(self):
        self.enrollment_gen = EnrollmentDashboardGenerator()
        self.ae_reporter = AdverseEventReporter()
        self.site_reporter = SitePerformanceReporter()

    def generate(
        self,
        trial_id: str,
        review_number: int,
        site_metrics: Dict[str, SitePerformanceMetrics],
        adverse_events: List[AdverseEvent],
        arm_distribution: Optional[Dict[str, int]] = None,
        efficacy_data: Optional[Dict[str, Any]] = None,
    ) -> DSMBPackage:
        """Generate a complete DSMB review package.

        Args:
            trial_id: Trial identifier.
            review_number: Sequential DSMB review number.
            site_metrics: Per-site performance metrics.
            adverse_events: All adverse events.
            arm_distribution: Treatment arm distribution.
            efficacy_data: Interim efficacy data.

        Returns:
            DSMBPackage ready for review.
        """
        enrollment = self.enrollment_gen.generate(trial_id, site_metrics, arm_distribution)
        ae_summary = self.ae_reporter.summarize(adverse_events)

        safety_signals = self._detect_safety_signals(adverse_events, ae_summary)

        recommendations = self._generate_recommendations(enrollment, ae_summary, safety_signals)

        package = DSMBPackage(
            package_id=str(uuid.uuid4())[:8],
            trial_id=trial_id,
            review_number=review_number,
            enrollment_summary=enrollment,
            ae_summary=ae_summary,
            efficacy_data=efficacy_data or {},
            safety_signals=safety_signals,
            recommendations=recommendations,
        )

        logger.info("DSMB package generated: review #%d for trial %s", review_number, trial_id)
        return package

    def _detect_safety_signals(
        self, adverse_events: List[AdverseEvent], ae_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect potential safety signals from AE data."""
        signals = []

        grade_dist = ae_summary.get("grade_distribution", {})
        severe_count = grade_dist.get("grade_4", 0) + grade_dist.get("grade_5", 0)
        total = ae_summary.get("total_aes", 0)
        if total > 0 and severe_count / total > 0.1:
            signals.append(
                {
                    "signal": "high_severe_ae_rate",
                    "description": f"Severe AE rate ({severe_count}/{total}) exceeds 10% threshold",
                    "severity": "high",
                }
            )

        if ae_summary.get("total_saes", 0) > 0:
            sae_count = ae_summary["total_saes"]
            signals.append(
                {
                    "signal": "sae_detected",
                    "description": f"{sae_count} serious adverse events reported",
                    "severity": "medium",
                }
            )

        return signals

    def _generate_recommendations(
        self,
        enrollment: EnrollmentDashboard,
        ae_summary: Dict[str, Any],
        safety_signals: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate DSMB recommendations based on data."""
        recommendations = []

        if enrollment.total_target > 0:
            pct = enrollment.total_enrolled / enrollment.total_target
            if pct < 0.3:
                recommendations.append("Consider enrollment enhancement strategies — current rate below 30% of target")

        high_signals = [s for s in safety_signals if s.get("severity") == "high"]
        if high_signals:
            recommendations.append("Review high-severity safety signals before proceeding")

        if not recommendations:
            recommendations.append("No significant safety concerns identified — recommend continuation")

        return recommendations


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = DSMBPackageGenerator()

    site_metrics = {}
    for i in range(4):
        site_id = f"site_{i:03d}"
        site_metrics[site_id] = SitePerformanceMetrics(
            site_id=site_id,
            site_name=f"Hospital {chr(65 + i)}",
            enrollment_count=10 + i * 5,
            enrollment_target=30,
            enrollment_rate=2.0 + i * 0.5,
            data_completeness=0.92 - i * 0.03,
            query_rate=5 + i * 2,
            protocol_deviation_count=i,
        )

    aes = []
    for i in range(20):
        ae = AdverseEvent(
            ae_id=f"AE-{i:04d}",
            patient_id=f"PT-{i:04d}",
            site_id=f"site_{i % 4:03d}",
            term=np.random.choice(["Nausea", "Fatigue", "Neutropenia", "Diarrhea", "Rash"]),
            system_organ_class=np.random.choice(["Gastrointestinal", "General", "Blood", "Skin"]),
            grade=AdverseEventGrade(np.random.randint(1, 4)),
            relationship=AERelationship(np.random.choice(["unrelated", "possible", "probable"])),
            serious=(i % 7 == 0),
        )
        aes.append(ae)

    package = gen.generate(
        trial_id="ONCO-FED-001",
        review_number=1,
        site_metrics=site_metrics,
        adverse_events=aes,
        arm_distribution={"experimental": 15, "control": 15},
    )

    print(f"\n=== DSMB Package (Review #{package.review_number}) ===")
    print(f"  Enrollment: {package.enrollment_summary.total_enrolled}/{package.enrollment_summary.total_target}")
    print(f"  Total AEs: {package.ae_summary['total_aes']}")
    print(f"  SAEs: {package.ae_summary['total_saes']}")
    print(f"  Safety signals: {len(package.safety_signals)}")
    print("  Recommendations:")
    for rec in package.recommendations:
        print(f"    - {rec}")
