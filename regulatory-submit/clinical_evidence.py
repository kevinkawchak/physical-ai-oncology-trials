"""
=============================================================================
Clinical Evidence Report Builder for Physical AI Oncology Trials
=============================================================================

Links simulation benchmark results and digital twin validation data to
clinical performance claims, generating structured clinical evidence
reports with statistical analysis and confidence intervals for FDA
regulatory submissions.

CLINICAL CONTEXT:
-----------------
FDA submissions for AI/ML-enabled oncology devices require clinical evidence
demonstrating safety and effectiveness. This module bridges the gap between
computational validation (simulation benchmarks, digital twin accuracy) and
clinical performance claims by:
  - Aggregating benchmark results into evidence summaries
  - Computing statistical metrics with confidence intervals
  - Performing subgroup analyses across demographic categories
  - Generating SPIRIT-AI / CONSORT-AI compliant reporting sections
  - Linking simulation fidelity to clinical relevance arguments

The evidence hierarchy for AI devices typically includes:
  1. Bench testing / software verification
  2. Simulation and digital twin validation
  3. Retrospective clinical validation
  4. Prospective clinical performance studies

USE CASES COVERED:
    1. Evidence reports linking simulation benchmarks to clinical claims
    2. Statistical analysis of model performance with CIs
    3. Subgroup analysis across demographic categories
    4. Clinical evidence summary for 510(k) / De Novo submissions

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - FDA Guidance: Clinical Performance Assessment (2025)
    - FDA Draft Guidance: AI-Enabled Device Software Functions (Jan 2025)
    - SPIRIT-AI / CONSORT-AI Reporting Extensions (2020)
    - STARD-AI Reporting Guidelines
    - TRIPOD-AI Prediction Model Reporting

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Clinical evidence reports require biostatistician and regulatory review
    before inclusion in FDA submissions.

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================


class EvidenceLevel(Enum):
    """Evidence hierarchy levels for AI device validation."""

    BENCH_TESTING = "bench_testing"
    SIMULATION = "simulation_validation"
    DIGITAL_TWIN = "digital_twin_validation"
    RETROSPECTIVE = "retrospective_clinical"
    PROSPECTIVE = "prospective_clinical"


class MetricType(Enum):
    """Types of performance metrics."""

    ACCURACY = "accuracy"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    AUC = "auc_roc"
    DICE = "dice_coefficient"
    HAUSDORFF = "hausdorff_distance"
    MAE = "mean_absolute_error"
    RMSE = "root_mean_square_error"
    CONCORDANCE = "concordance_index"
    CUSTOM = "custom"


@dataclass
class BenchmarkResult:
    """Individual benchmark or validation result."""

    benchmark_name: str
    evidence_level: EvidenceLevel
    metric_type: MetricType
    metric_name: str
    value: float
    sample_size: int
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    confidence_level: float = 0.95
    dataset_description: str = ""
    methodology: str = ""
    date_collected: str = ""


@dataclass
class SubgroupResult:
    """Performance result for a demographic subgroup."""

    subgroup_category: str  # "age", "sex", "race_ethnicity", "tumor_stage", etc.
    subgroup_value: str  # "18-40", "Female", "Black/African American", etc.
    metric_name: str
    value: float
    sample_size: int
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0


@dataclass
class ClinicalClaim:
    """Clinical performance claim supported by evidence."""

    claim_id: str
    claim_text: str
    intended_use_alignment: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    evidence_strength: str = "moderate"  # "strong", "moderate", "limited"
    limitations: list[str] = field(default_factory=list)


@dataclass
class ClinicalEvidenceReport:
    """Complete clinical evidence report."""

    report_id: str
    device_name: str
    intended_use: str
    sponsor: str = ""
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)
    subgroup_results: list[SubgroupResult] = field(default_factory=list)
    clinical_claims: list[ClinicalClaim] = field(default_factory=list)
    study_limitations: list[str] = field(default_factory=list)
    generated_date: str = ""
    generated_markdown: str = ""


# =============================================================================
# SECTION 2: STATISTICAL UTILITIES
# =============================================================================


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    # Approximate z-value for common confidence levels
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    denominator = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def normal_ci(value: float, std_error: float, confidence: float = 0.95) -> tuple[float, float]:
    """
    Compute normal approximation confidence interval.

    Args:
        value: Point estimate.
        std_error: Standard error of the estimate.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)
    return (value - z * std_error, value + z * std_error)


def compute_confidence_interval(
    metric_value: float,
    sample_size: int,
    metric_type: MetricType,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute confidence interval appropriate for the metric type.

    Args:
        metric_value: Observed metric value.
        sample_size: Sample size.
        metric_type: Type of metric (determines CI method).
        confidence: Confidence level.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if sample_size <= 0:
        return (0.0, 0.0)

    proportion_metrics = {
        MetricType.ACCURACY,
        MetricType.SENSITIVITY,
        MetricType.SPECIFICITY,
        MetricType.AUC,
        MetricType.DICE,
    }

    if metric_type in proportion_metrics:
        successes = int(round(metric_value * sample_size))
        successes = max(0, min(successes, sample_size))
        return wilson_score_interval(successes, sample_size, confidence)
    else:
        # For continuous metrics, use normal approximation
        # Approximate SE from sample size
        se = metric_value * 0.1 / math.sqrt(sample_size) if sample_size > 1 else metric_value * 0.1
        return normal_ci(metric_value, se, confidence)


# =============================================================================
# SECTION 3: CLINICAL EVIDENCE BUILDER
# =============================================================================


class ClinicalEvidenceBuilder:
    """
    Builds clinical evidence reports linking simulation and validation
    results to clinical performance claims for FDA submissions.

    Produces structured Markdown reports with statistical analysis,
    confidence intervals, subgroup analyses, and evidence summaries
    suitable for inclusion in 510(k), De Novo, or PMA submissions.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Aggregate benchmark results from multiple validation sources
    2. Compute confidence intervals for all performance metrics
    3. Perform demographic subgroup analysis
    4. Generate evidence-to-claim linkage documentation
    """

    def __init__(self, sponsor: str = "") -> None:
        """
        Initialize clinical evidence builder.

        Args:
            sponsor: Sponsoring organization name.
        """
        self.sponsor = sponsor
        self._report_counter = 0
        logger.info("ClinicalEvidenceBuilder initialized")

    def create_report(
        self,
        device_name: str,
        intended_use: str,
    ) -> ClinicalEvidenceReport:
        """
        Create a new clinical evidence report.

        Args:
            device_name: Name of the device.
            intended_use: Intended use statement.

        Returns:
            ClinicalEvidenceReport with initialized structure.
        """
        self._report_counter += 1
        report_id = f"CER-{datetime.now().strftime('%Y%m%d')}-{self._report_counter:04d}"

        report = ClinicalEvidenceReport(
            report_id=report_id,
            device_name=device_name,
            intended_use=intended_use,
            sponsor=self.sponsor,
            generated_date=datetime.now().isoformat(),
        )

        logger.info(f"Clinical evidence report created: {report_id}")
        return report

    def add_benchmark_result(
        self,
        report: ClinicalEvidenceReport,
        benchmark_name: str,
        evidence_level: EvidenceLevel,
        metric_type: MetricType,
        metric_name: str,
        value: float,
        sample_size: int,
        dataset_description: str = "",
        methodology: str = "",
        confidence_level: float = 0.95,
    ) -> BenchmarkResult:
        """
        Add a benchmark result with auto-computed confidence interval.

        Args:
            report: Report to add the result to.
            benchmark_name: Name of the benchmark.
            evidence_level: Level in the evidence hierarchy.
            metric_type: Type of performance metric.
            metric_name: Display name of the metric.
            value: Observed metric value.
            sample_size: Sample size.
            dataset_description: Description of the evaluation dataset.
            methodology: Evaluation methodology description.
            confidence_level: Confidence level for CI computation.

        Returns:
            BenchmarkResult with computed confidence interval.
        """
        lower, upper = compute_confidence_interval(value, sample_size, metric_type, confidence_level)

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            evidence_level=evidence_level,
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            sample_size=sample_size,
            confidence_interval_lower=lower,
            confidence_interval_upper=upper,
            confidence_level=confidence_level,
            dataset_description=dataset_description,
            methodology=methodology,
            date_collected=datetime.now().strftime("%Y-%m-%d"),
        )

        report.benchmark_results.append(result)
        logger.info(f"Benchmark result added: {metric_name}={value:.4f} [{lower:.4f}, {upper:.4f}]")
        return result

    def add_subgroup_result(
        self,
        report: ClinicalEvidenceReport,
        subgroup_category: str,
        subgroup_value: str,
        metric_name: str,
        value: float,
        sample_size: int,
        metric_type: MetricType = MetricType.ACCURACY,
    ) -> SubgroupResult:
        """
        Add a subgroup analysis result.

        Args:
            report: Report to add the result to.
            subgroup_category: Category (e.g., "age", "sex").
            subgroup_value: Specific subgroup (e.g., "Female").
            metric_name: Metric name.
            value: Observed value.
            sample_size: Sample size for this subgroup.
            metric_type: Metric type for CI computation.

        Returns:
            SubgroupResult with computed confidence interval.
        """
        lower, upper = compute_confidence_interval(value, sample_size, metric_type)

        result = SubgroupResult(
            subgroup_category=subgroup_category,
            subgroup_value=subgroup_value,
            metric_name=metric_name,
            value=value,
            sample_size=sample_size,
            confidence_interval_lower=lower,
            confidence_interval_upper=upper,
        )

        report.subgroup_results.append(result)
        return result

    def add_clinical_claim(
        self,
        report: ClinicalEvidenceReport,
        claim_text: str,
        intended_use_alignment: str = "",
        supporting_evidence: list[str] | None = None,
        evidence_strength: str = "moderate",
        limitations: list[str] | None = None,
    ) -> ClinicalClaim:
        """
        Add a clinical performance claim with supporting evidence links.

        Args:
            report: Report to add the claim to.
            claim_text: The clinical claim text.
            intended_use_alignment: How the claim aligns with intended use.
            supporting_evidence: List of evidence references.
            evidence_strength: Strength of supporting evidence.
            limitations: Known limitations of the claim.

        Returns:
            ClinicalClaim object.
        """
        claim_id = f"CLM-{len(report.clinical_claims) + 1:03d}"

        claim = ClinicalClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            intended_use_alignment=intended_use_alignment,
            supporting_evidence=supporting_evidence or [],
            evidence_strength=evidence_strength,
            limitations=limitations or [],
        )

        report.clinical_claims.append(claim)
        return claim

    def generate_markdown(self, report: ClinicalEvidenceReport) -> str:
        """
        Generate the complete clinical evidence report as Markdown.

        Args:
            report: Report to render.

        Returns:
            Complete report as Markdown string.
        """
        sections: list[str] = []

        sections.append("# Clinical Evidence Report\n")
        sections.append(f"**Report ID:** {report.report_id}  ")
        sections.append(f"**Device:** {report.device_name}  ")
        if report.sponsor:
            sections.append(f"**Sponsor:** {report.sponsor}  ")
        sections.append(f"**Date:** {report.generated_date[:10]}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** This report is generated for planning and analysis. ")
        sections.append("> Clinical evidence requires biostatistician and regulatory review before submission.\n")

        # Intended Use
        sections.append("## 1. Intended Use\n")
        sections.append(f"{report.intended_use}\n")

        # Performance Results
        if report.benchmark_results:
            sections.append("## 2. Performance Results\n")

            # Group by evidence level
            by_level: dict[str, list[BenchmarkResult]] = {}
            for r in report.benchmark_results:
                key = r.evidence_level.value
                by_level.setdefault(key, []).append(r)

            for level, results in by_level.items():
                level_title = level.replace("_", " ").title()
                sections.append(f"### {level_title}\n")
                sections.append("| Benchmark | Metric | Value | 95% CI | N |")
                sections.append("|-----------|--------|-------|--------|---|")
                for r in results:
                    ci_str = f"[{r.confidence_interval_lower:.3f}, {r.confidence_interval_upper:.3f}]"
                    sections.append(
                        f"| {r.benchmark_name} | {r.metric_name} | {r.value:.4f} | {ci_str} | {r.sample_size:,} |"
                    )
                sections.append("")

                for r in results:
                    if r.dataset_description or r.methodology:
                        sections.append(f"**{r.benchmark_name} — {r.metric_name}:**")
                        if r.dataset_description:
                            sections.append(f"- Dataset: {r.dataset_description}")
                        if r.methodology:
                            sections.append(f"- Methodology: {r.methodology}")
                        sections.append("")

        # Subgroup Analysis
        if report.subgroup_results:
            sections.append("## 3. Subgroup Analysis\n")
            sections.append("Demographic subgroup performance per FDA AI/ML bias assessment guidance:\n")

            by_category: dict[str, list[SubgroupResult]] = {}
            for sg in report.subgroup_results:
                by_category.setdefault(sg.subgroup_category, []).append(sg)

            for category, results in by_category.items():
                sections.append(f"### {category.replace('_', ' ').title()}\n")
                sections.append("| Subgroup | Metric | Value | 95% CI | N |")
                sections.append("|----------|--------|-------|--------|---|")
                for r in results:
                    ci_str = f"[{r.confidence_interval_lower:.3f}, {r.confidence_interval_upper:.3f}]"
                    sections.append(
                        f"| {r.subgroup_value} | {r.metric_name} | {r.value:.4f} | {ci_str} | {r.sample_size:,} |"
                    )
                sections.append("")

            # Parity check
            sections.append("### Subgroup Parity Assessment\n")
            for category, results in by_category.items():
                if len(results) >= 2:
                    values = [r.value for r in results]
                    max_diff = max(values) - min(values)
                    status = "PASS" if max_diff <= 0.05 else "REVIEW NEEDED"
                    sections.append(
                        f"- **{category.replace('_', ' ').title()}**: Max difference = {max_diff:.4f} — {status}"
                    )
            sections.append("")

        # Clinical Claims
        if report.clinical_claims:
            sections.append("## 4. Clinical Performance Claims\n")
            for claim in report.clinical_claims:
                sections.append(f"### {claim.claim_id}: {claim.claim_text}\n")
                sections.append(f"**Evidence Strength:** {claim.evidence_strength.title()}\n")
                if claim.intended_use_alignment:
                    sections.append(f"**Intended Use Alignment:** {claim.intended_use_alignment}\n")
                if claim.supporting_evidence:
                    sections.append("**Supporting Evidence:**\n")
                    for ev in claim.supporting_evidence:
                        sections.append(f"- {ev}")
                    sections.append("")
                if claim.limitations:
                    sections.append("**Limitations:**\n")
                    for lim in claim.limitations:
                        sections.append(f"- {lim}")
                    sections.append("")

        # Study Limitations
        if report.study_limitations:
            sections.append("## 5. Study Limitations\n")
            for lim in report.study_limitations:
                sections.append(f"- {lim}")
            sections.append("")

        # Evidence Summary
        sections.append("## Evidence Summary\n")
        total_benchmarks = len(report.benchmark_results)
        total_subgroups = len(report.subgroup_results)
        total_claims = len(report.clinical_claims)
        sections.append("| Metric | Count |")
        sections.append("|--------|-------|")
        sections.append(f"| Benchmark Results | {total_benchmarks} |")
        sections.append(f"| Subgroup Analyses | {total_subgroups} |")
        sections.append(f"| Clinical Claims | {total_claims} |")
        sections.append("")

        sections.append("---\n")
        sections.append("*Generated by Physical AI Oncology Trials — Clinical Evidence Builder*  ")
        sections.append(f"*Report ID: {report.report_id} | Generated: {report.generated_date[:10]}*  ")
        sections.append("*RESEARCH USE ONLY — Requires biostatistician review before submission*\n")

        markdown = "\n".join(sections)
        report.generated_markdown = markdown

        logger.info(
            f"Clinical evidence report generated for {report.report_id}: "
            f"{total_benchmarks} results, {total_subgroups} subgroups, {total_claims} claims"
        )

        return markdown


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_clinical_evidence_demo() -> dict[str, Any]:
    """
    Demonstrate clinical evidence report generation.

    Returns:
        Dictionary with demo results.
    """
    logger.info("=" * 60)
    logger.info("CLINICAL EVIDENCE REPORT BUILDER DEMO")
    logger.info("=" * 60)

    builder = ClinicalEvidenceBuilder(sponsor="Physical AI Oncology Consortium")

    report = builder.create_report(
        device_name="OncoTwin AI Surgical Planning System",
        intended_use=(
            "AI-assisted surgical planning for solid tumor resection in adult "
            "patients with non-small cell lung cancer (NSCLC)"
        ),
    )

    # Add benchmark results across evidence levels
    builder.add_benchmark_result(
        report,
        benchmark_name="TumorSeg-3D Validation",
        evidence_level=EvidenceLevel.SIMULATION,
        metric_type=MetricType.DICE,
        metric_name="Dice Coefficient",
        value=0.912,
        sample_size=500,
        dataset_description="Multi-institutional CT dataset from 5 NCI-designated cancer centers",
        methodology="5-fold cross-validation with external hold-out test set",
    )
    builder.add_benchmark_result(
        report,
        benchmark_name="TumorSeg-3D Validation",
        evidence_level=EvidenceLevel.SIMULATION,
        metric_type=MetricType.SENSITIVITY,
        metric_name="Sensitivity (voxel-level)",
        value=0.945,
        sample_size=500,
        methodology="Per-voxel sensitivity on 3D tumor boundary",
    )
    builder.add_benchmark_result(
        report,
        benchmark_name="PathOptimizer Clinical Review",
        evidence_level=EvidenceLevel.RETROSPECTIVE,
        metric_type=MetricType.ACCURACY,
        metric_name="Expert Agreement Rate",
        value=0.856,
        sample_size=200,
        dataset_description="Retrospective review of 200 NSCLC surgical cases",
        methodology="Multi-reader multi-case study with 3 blinded thoracic surgeons",
    )
    builder.add_benchmark_result(
        report,
        benchmark_name="Digital Twin Fidelity",
        evidence_level=EvidenceLevel.DIGITAL_TWIN,
        metric_type=MetricType.MAE,
        metric_name="Tumor Volume MAE (cm3)",
        value=1.23,
        sample_size=300,
        methodology="Predicted vs. actual post-operative tumor volume",
    )

    # Add subgroup results
    subgroups = [
        ("age", "18-49", "Dice Coefficient", 0.908, 85),
        ("age", "50-64", "Dice Coefficient", 0.915, 180),
        ("age", "65-79", "Dice Coefficient", 0.910, 175),
        ("age", "80+", "Dice Coefficient", 0.904, 60),
        ("sex", "Female", "Dice Coefficient", 0.914, 220),
        ("sex", "Male", "Dice Coefficient", 0.910, 280),
        ("race_ethnicity", "White", "Dice Coefficient", 0.913, 300),
        ("race_ethnicity", "Black/African American", "Dice Coefficient", 0.908, 80),
        ("race_ethnicity", "Asian", "Dice Coefficient", 0.916, 70),
        ("race_ethnicity", "Hispanic/Latino", "Dice Coefficient", 0.905, 50),
    ]
    for cat, val, metric, value, n in subgroups:
        builder.add_subgroup_result(report, cat, val, metric, value, n, MetricType.DICE)

    # Add clinical claims
    builder.add_clinical_claim(
        report,
        claim_text="TumorSeg-3D achieves clinically acceptable tumor boundary delineation for NSCLC surgical planning",
        intended_use_alignment="Directly supports intended use of AI-assisted tumor delineation",
        supporting_evidence=[
            "Dice coefficient 0.912 (95% CI: [0.886, 0.934]) on external validation set",
            "Voxel-level sensitivity 0.945 (95% CI: [0.921, 0.963])",
            "Performance consistent across demographic subgroups (max difference 0.011)",
        ],
        evidence_strength="strong",
        limitations=[
            "Validation limited to NSCLC; generalization to other tumor types not established",
            "Performance on tumors < 1cm not independently validated",
        ],
    )
    builder.add_clinical_claim(
        report,
        claim_text="PathOptimizer-RL recommendations achieve substantial agreement with expert thoracic surgeon consensus",
        intended_use_alignment="Supports surgical path optimization component of intended use",
        supporting_evidence=[
            "Expert agreement rate 85.6% (95% CI: [80.0%, 89.9%]) on retrospective case review",
            "3-surgeon consensus panel, blinded to AI recommendations",
        ],
        evidence_strength="moderate",
        limitations=[
            "Retrospective study design — prospective validation recommended",
            "Agreement metric does not directly measure clinical outcomes",
        ],
    )

    report.study_limitations = [
        "All validation data is retrospective; prospective clinical performance study is planned",
        "Digital twin fidelity validated on NSCLC only — other indications require separate validation",
        "Subgroup sizes for some demographic categories are limited (N < 100)",
        "Long-term clinical outcome correlation (survival, recurrence) not yet assessed",
    ]

    markdown = builder.generate_markdown(report)

    print(f"\nReport: {report.report_id}")
    print(f"Benchmarks: {len(report.benchmark_results)}")
    print(f"Subgroups: {len(report.subgroup_results)}")
    print(f"Claims: {len(report.clinical_claims)}")
    print(f"Document length: {len(markdown):,} characters")

    return {
        "report_id": report.report_id,
        "benchmark_count": len(report.benchmark_results),
        "subgroup_count": len(report.subgroup_results),
        "claim_count": len(report.clinical_claims),
        "document_length": len(markdown),
        "status": "demo_complete",
    }


if __name__ == "__main__":
    result = run_clinical_evidence_demo()
    print(f"\nDemo completed: {result}")
