#!/usr/bin/env python3
"""Example 05: Clinical Evidence Report Building.

Demonstrates building a clinical evidence report that links simulation
benchmark results, digital twin validation data, and retrospective
clinical studies to clinical performance claims with statistical
analysis and confidence intervals.

Key Concepts:
    - Multi-level evidence aggregation (simulation, digital twin, clinical)
    - Confidence interval computation for performance metrics
    - Demographic subgroup analysis for bias assessment
    - Clinical claim documentation with evidence linkage
    - SPIRIT-AI / CONSORT-AI reporting alignment

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_evidence import (
    ClinicalEvidenceBuilder,
    EvidenceLevel,
    MetricType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark Data
# ---------------------------------------------------------------------------

BENCHMARK_RESULTS = [
    {
        "name": "TumorSeg-3D Software Verification",
        "level": EvidenceLevel.BENCH_TESTING,
        "metric_type": MetricType.DICE,
        "metric_name": "Dice Coefficient (software verification)",
        "value": 0.925,
        "n": 1000,
        "dataset": "Synthetic tumor phantoms with ground truth labels",
        "method": "Automated unit and integration testing on synthetic data",
    },
    {
        "name": "TumorSeg-3D Simulation Validation",
        "level": EvidenceLevel.SIMULATION,
        "metric_type": MetricType.DICE,
        "metric_name": "Dice Coefficient (simulation)",
        "value": 0.912,
        "n": 500,
        "dataset": "Multi-institutional CT dataset from 5 NCI-designated cancer centers",
        "method": "5-fold cross-validation with external hold-out test set",
    },
    {
        "name": "TumorSeg-3D Simulation Validation",
        "level": EvidenceLevel.SIMULATION,
        "metric_type": MetricType.SENSITIVITY,
        "metric_name": "Sensitivity (voxel-level)",
        "value": 0.945,
        "n": 500,
        "dataset": "Multi-institutional CT dataset from 5 NCI-designated cancer centers",
        "method": "Per-voxel sensitivity on 3D tumor boundary",
    },
    {
        "name": "Digital Twin Fidelity Assessment",
        "level": EvidenceLevel.DIGITAL_TWIN,
        "metric_type": MetricType.CONCORDANCE,
        "metric_name": "C-index (predicted vs. actual volume)",
        "value": 0.847,
        "n": 300,
        "dataset": "Matched pre/post-operative imaging from 3 institutions",
        "method": "Concordance index between predicted and actual residual tumor volume",
    },
    {
        "name": "Digital Twin Fidelity Assessment",
        "level": EvidenceLevel.DIGITAL_TWIN,
        "metric_type": MetricType.MAE,
        "metric_name": "Volume MAE (cm3)",
        "value": 1.23,
        "n": 300,
        "dataset": "Matched pre/post-operative imaging from 3 institutions",
        "method": "Mean absolute error between predicted and actual post-operative volume",
    },
    {
        "name": "PathOptimizer Retrospective Study",
        "level": EvidenceLevel.RETROSPECTIVE,
        "metric_type": MetricType.ACCURACY,
        "metric_name": "Expert Agreement Rate",
        "value": 0.856,
        "n": 200,
        "dataset": "Retrospective review of 200 NSCLC surgical cases",
        "method": "Multi-reader multi-case study with 3 blinded thoracic surgeons",
    },
    {
        "name": "PathOptimizer Retrospective Study",
        "level": EvidenceLevel.RETROSPECTIVE,
        "metric_type": MetricType.ACCURACY,
        "metric_name": "Margin Adequacy (R0 rate)",
        "value": 0.920,
        "n": 200,
        "dataset": "Pathology-confirmed margin status for AI-planned vs. standard cases",
        "method": "Retrospective comparison with propensity-score matching",
    },
]

SUBGROUP_DATA = [
    # Age subgroups
    ("age", "18-49", "Dice Coefficient", 0.908, 85, MetricType.DICE),
    ("age", "50-64", "Dice Coefficient", 0.915, 180, MetricType.DICE),
    ("age", "65-79", "Dice Coefficient", 0.910, 175, MetricType.DICE),
    ("age", "80+", "Dice Coefficient", 0.904, 60, MetricType.DICE),
    # Sex subgroups
    ("sex", "Female", "Dice Coefficient", 0.914, 220, MetricType.DICE),
    ("sex", "Male", "Dice Coefficient", 0.910, 280, MetricType.DICE),
    # Race/ethnicity subgroups
    ("race_ethnicity", "White", "Dice Coefficient", 0.913, 300, MetricType.DICE),
    ("race_ethnicity", "Black/African American", "Dice Coefficient", 0.908, 80, MetricType.DICE),
    ("race_ethnicity", "Asian", "Dice Coefficient", 0.916, 70, MetricType.DICE),
    ("race_ethnicity", "Hispanic/Latino", "Dice Coefficient", 0.905, 50, MetricType.DICE),
    # Tumor stage subgroups
    ("tumor_stage", "Stage I", "Dice Coefficient", 0.921, 200, MetricType.DICE),
    ("tumor_stage", "Stage II", "Dice Coefficient", 0.909, 180, MetricType.DICE),
    ("tumor_stage", "Stage IIIA", "Dice Coefficient", 0.895, 120, MetricType.DICE),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build a clinical evidence report linking benchmarks to claims."""
    print("=" * 70)
    print("Example 05: Clinical Evidence Report Building")
    print("=" * 70)

    builder = ClinicalEvidenceBuilder(sponsor="Physical AI Oncology Consortium")

    report = builder.create_report(
        device_name="OncoTwin AI Surgical Planning System",
        intended_use=(
            "AI-assisted surgical planning for solid tumor resection in adult patients with NSCLC, stages I-IIIA"
        ),
    )

    # Add benchmark results
    for br in BENCHMARK_RESULTS:
        builder.add_benchmark_result(
            report,
            benchmark_name=br["name"],
            evidence_level=br["level"],
            metric_type=br["metric_type"],
            metric_name=br["metric_name"],
            value=br["value"],
            sample_size=br["n"],
            dataset_description=br["dataset"],
            methodology=br["method"],
        )

    # Add subgroup results
    for cat, val, metric, value, n, mt in SUBGROUP_DATA:
        builder.add_subgroup_result(report, cat, val, metric, value, n, mt)

    # Add clinical claims
    builder.add_clinical_claim(
        report,
        claim_text=(
            "TumorSeg-3D achieves clinically acceptable tumor boundary delineation "
            "for NSCLC surgical planning across the intended use population"
        ),
        intended_use_alignment="Directly supports tumor delineation component of intended use",
        supporting_evidence=[
            "Dice coefficient 0.912 on multi-institutional external validation set (N=500)",
            "Voxel-level sensitivity 0.945 (N=500)",
            "Performance consistent across demographic subgroups (max difference 0.011)",
            "Software verification Dice 0.925 on synthetic phantoms (N=1,000)",
        ],
        evidence_strength="strong",
        limitations=[
            "Validation limited to NSCLC; generalization to other tumor types not established",
            "Performance on tumors < 1cm not independently validated",
            "Hispanic/Latino subgroup size limited (N=50)",
        ],
    )

    builder.add_clinical_claim(
        report,
        claim_text=(
            "PathOptimizer-RL surgical path recommendations achieve substantial agreement "
            "with expert thoracic surgeon consensus"
        ),
        intended_use_alignment="Supports path optimization component of intended use",
        supporting_evidence=[
            "Expert agreement rate 85.6% (N=200) on retrospective case review",
            "3-surgeon consensus panel, blinded to AI recommendations",
            "Margin adequacy (R0 rate) 92.0% for AI-planned cases",
        ],
        evidence_strength="moderate",
        limitations=[
            "Retrospective study design — prospective validation planned",
            "Agreement metric does not directly measure long-term outcomes",
        ],
    )

    builder.add_clinical_claim(
        report,
        claim_text=(
            "Digital twin simulation provides clinically relevant outcome predictions "
            "for surgical planning decision support"
        ),
        intended_use_alignment="Supports simulation-based planning feature",
        supporting_evidence=[
            "C-index 0.847 for predicted vs. actual residual tumor volume (N=300)",
            "Volume MAE 1.23 cm3 — within clinically acceptable range for planning",
        ],
        evidence_strength="moderate",
        limitations=[
            "Digital twin validation on NSCLC only",
            "Prospective real-time simulation accuracy not yet assessed",
        ],
    )

    # Add study limitations
    report.study_limitations = [
        "All validation data is retrospective; prospective clinical study is planned",
        "Digital twin fidelity validated on NSCLC only",
        "Some demographic subgroup sizes are limited (N < 100)",
        "Long-term outcome correlation (survival, recurrence) not yet assessed",
        "Multi-center prospective study needed for post-market evidence generation",
    ]

    # Generate report
    markdown = builder.generate_markdown(report)

    # Output summary
    print(f"\nReport ID: {report.report_id}")
    print(f"Benchmark Results: {len(report.benchmark_results)}")
    print(f"Subgroup Analyses: {len(report.subgroup_results)}")
    print(f"Clinical Claims: {len(report.clinical_claims)}")
    print(f"Document length: {len(markdown):,} characters")

    # Show evidence level distribution
    levels = {}
    for br in report.benchmark_results:
        levels[br.evidence_level.value] = levels.get(br.evidence_level.value, 0) + 1
    print("\nEvidence Level Distribution:")
    for level, count in sorted(levels.items()):
        print(f"  {level}: {count} results")

    print("\nClinical evidence report generation complete.")


if __name__ == "__main__":
    main()
