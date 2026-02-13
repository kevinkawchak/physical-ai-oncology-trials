#!/usr/bin/env python3
"""Example 06: Complete Multi-Component Regulatory Submission Strategy.

Demonstrates a full regulatory submission workflow combining all
regulatory-submit components: Pre-Sub package, PCCP plan, pathway
classification, IEC 62304 lifecycle documentation, clinical evidence
report, and 21 CFR Part 11 audit trail for a comprehensive De Novo
submission strategy.

Key Concepts:
    - End-to-end regulatory submission document generation
    - Cross-component integration (Pre-Sub -> PCCP -> Evidence -> Audit)
    - Multi-model device regulatory strategy
    - Complete document package assembly
    - Regulatory lifecycle management

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

from audit_trail import (
    AuditEventType,
    AuditTrailGenerator,
)
from classification_advisor import (
    AIDeviceType,
    ClassificationAdvisor,
    DeviceCharacteristics,
)
from clinical_evidence import (
    ClinicalEvidenceBuilder,
    EvidenceLevel,
    MetricType,
)
from iec62304_generator import (
    IEC62304Generator,
    SafetyClass,
    SoftwareItem,
)
from pccp_engine import PCCPEngine
from presub_generator import (
    AIModelDescription,
    DeviceCategory,
    PreSubGenerator,
    RegulatoryPathway,
    TestingProtocol,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Device Configuration
# ---------------------------------------------------------------------------

DEVICE_NAME = "OncoTwin AI Surgical Planning System"
SPONSOR = "Physical AI Oncology Consortium"
INDICATION = "Non-small cell lung cancer (NSCLC), stages I-IIIA"
INTENDED_USE = (
    "AI-assisted surgical planning for solid tumor resection in adult patients "
    "with non-small cell lung cancer (NSCLC), using patient-specific digital "
    "twin models derived from pre-operative CT and MRI imaging"
)


# ---------------------------------------------------------------------------
# Step Functions
# ---------------------------------------------------------------------------


def step_1_classification() -> str:
    """Step 1: Determine regulatory pathway."""
    print("\n" + "=" * 60)
    print("STEP 1: Regulatory Pathway Classification")
    print("=" * 60)

    advisor = ClassificationAdvisor()
    characteristics = DeviceCharacteristics(
        device_name=DEVICE_NAME,
        device_type=AIDeviceType.TREATMENT_PLANNING,
        intended_use=INTENDED_USE,
        oncology_indication=INDICATION,
        software_only=True,
        clinician_in_loop=True,
        novel_algorithm=True,
        training_data_size=12500,
        life_threatening_condition=True,
    )

    recommendation = advisor.analyze(characteristics)
    markdown = advisor.generate_markdown(recommendation)

    print(f"  Pathway: {recommendation.recommended_pathway.value.upper()}")
    print(f"  Class: {recommendation.recommended_risk_class.value}")
    print(f"  Confidence: {recommendation.confidence}")
    print(f"  Document: {len(markdown):,} chars")

    return markdown


def step_2_presub() -> str:
    """Step 2: Generate Pre-Sub package."""
    print("\n" + "=" * 60)
    print("STEP 2: Pre-Submission Package Generation")
    print("=" * 60)

    generator = PreSubGenerator(
        sponsor=SPONSOR,
        contact_name="Regulatory Affairs Office",
    )

    package = generator.create_package(
        device_name=DEVICE_NAME,
        device_category=DeviceCategory.SURGICAL_PLANNING,
        intended_use=INTENDED_USE,
        proposed_pathway=RegulatoryPathway.DE_NOVO,
        oncology_indication=INDICATION,
    )

    generator.add_ai_model(
        package,
        AIModelDescription(
            model_name="TumorSeg-3D",
            model_type="segmentation",
            architecture="3D U-Net with attention gates",
            input_modalities=["CT volumetric", "MRI T1-weighted"],
            training_data_size=12500,
            performance_metrics={"dice_coefficient": 0.912, "sensitivity": 0.945},
            locked_or_adaptive="locked",
        ),
    )
    generator.add_ai_model(
        package,
        AIModelDescription(
            model_name="PathOptimizer-RL",
            model_type="reinforcement_learning",
            architecture="PPO with anatomical constraints",
            input_modalities=["3D segmentation", "vascular anatomy"],
            training_data_size=50000,
            performance_metrics={"expert_agreement": 0.856},
            locked_or_adaptive="adaptive",
        ),
    )

    generator.add_testing_protocol(
        package,
        TestingProtocol(
            protocol_name="Retrospective Clinical Validation",
            objective="Validate AI planning accuracy against expert consensus",
            study_design="Multi-reader multi-case retrospective study",
            endpoints=["Margin adequacy (R0 rate)", "Expert agreement"],
        ),
    )

    generator.populate_risk_considerations(package)
    generator.auto_generate_questions(package)
    markdown = generator.generate_markdown(package)

    print(f"  Package: {package.package_id}")
    print(f"  Models: {len(package.ai_models)}")
    print(f"  Questions: {len(package.questions)}")
    print(f"  Document: {len(markdown):,} chars")

    return markdown


def step_3_pccp() -> str:
    """Step 3: Generate PCCP document."""
    print("\n" + "=" * 60)
    print("STEP 3: Predetermined Change Control Plan")
    print("=" * 60)

    engine = PCCPEngine(sponsor=SPONSOR)
    document = engine.create_pccp(
        device_name=DEVICE_NAME,
        submission_reference="DN-2026-0042",
        ai_model_names=["TumorSeg-3D", "PathOptimizer-RL"],
    )

    engine.populate_default_boundaries(document)
    engine.populate_default_validation(document)
    engine.set_transparency_plan(document)
    markdown = engine.generate_markdown(document)

    print(f"  Document: {document.document_id}")
    print(f"  Boundaries: {len(document.modification_boundaries)}")
    print(f"  Document: {len(markdown):,} chars")

    return markdown


def step_4_iec62304() -> str:
    """Step 4: Generate IEC 62304 documentation."""
    print("\n" + "=" * 60)
    print("STEP 4: IEC 62304 Software Lifecycle Documentation")
    print("=" * 60)

    generator = IEC62304Generator(sponsor=SPONSOR)
    doc_set = generator.create_document_set(
        project_name="OncoTwin-SW",
        device_name=DEVICE_NAME,
        safety_class=SafetyClass.CLASS_C,
    )

    items = [
        SoftwareItem(
            item_id="SWI-001",
            name="AI Inference Engine",
            description="Core AI/ML inference pipeline",
            safety_class=SafetyClass.CLASS_C,
            soup_components=["PyTorch", "ONNX Runtime", "NumPy"],
        ),
        SoftwareItem(
            item_id="SWI-002",
            name="Input Validator",
            description="DICOM data validation",
            safety_class=SafetyClass.CLASS_B,
            soup_components=["pydicom"],
        ),
        SoftwareItem(
            item_id="SWI-003",
            name="Digital Twin Engine",
            description="Patient-specific simulation",
            safety_class=SafetyClass.CLASS_C,
            soup_components=["SciPy", "NumPy"],
        ),
    ]
    for item in items:
        generator.add_software_item(doc_set, item)

    generator.populate_default_requirements(doc_set)
    generator.populate_sample_risks(doc_set)
    docs = generator.generate_all(doc_set)

    total_chars = sum(len(v) for v in docs.values())
    print(f"  Documents: {len(docs)}")
    print(f"  Requirements: {len(doc_set.requirements)}")
    print(f"  Risks: {len(doc_set.risks)}")
    print(f"  Total: {total_chars:,} chars")

    # Combine all IEC 62304 docs into one section
    combined = "\n\n---\n\n".join(docs.values())
    return combined


def step_5_evidence() -> str:
    """Step 5: Build clinical evidence report."""
    print("\n" + "=" * 60)
    print("STEP 5: Clinical Evidence Report")
    print("=" * 60)

    builder = ClinicalEvidenceBuilder(sponsor=SPONSOR)
    report = builder.create_report(device_name=DEVICE_NAME, intended_use=INTENDED_USE)

    # Key benchmark results
    benchmarks = [
        ("TumorSeg-3D", EvidenceLevel.SIMULATION, MetricType.DICE, "Dice Coefficient", 0.912, 500),
        ("TumorSeg-3D", EvidenceLevel.SIMULATION, MetricType.SENSITIVITY, "Sensitivity", 0.945, 500),
        ("Digital Twin", EvidenceLevel.DIGITAL_TWIN, MetricType.CONCORDANCE, "C-index", 0.847, 300),
        ("PathOptimizer", EvidenceLevel.RETROSPECTIVE, MetricType.ACCURACY, "Expert Agreement", 0.856, 200),
        ("PathOptimizer", EvidenceLevel.RETROSPECTIVE, MetricType.ACCURACY, "R0 Rate", 0.920, 200),
    ]
    for name, level, mt, mn, val, n in benchmarks:
        builder.add_benchmark_result(report, name, level, mt, mn, val, n)

    # Key subgroups
    subgroups = [
        ("sex", "Female", "Dice", 0.914, 220),
        ("sex", "Male", "Dice", 0.910, 280),
        ("age", "18-64", "Dice", 0.913, 265),
        ("age", "65+", "Dice", 0.908, 235),
    ]
    for cat, val, metric, v, n in subgroups:
        builder.add_subgroup_result(report, cat, val, metric, v, n, MetricType.DICE)

    builder.add_clinical_claim(
        report,
        claim_text="AI surgical planning achieves clinically acceptable accuracy for NSCLC",
        supporting_evidence=["Dice 0.912", "Sensitivity 0.945", "Expert agreement 85.6%"],
        evidence_strength="strong",
    )

    markdown = builder.generate_markdown(report)

    print(f"  Report: {report.report_id}")
    print(f"  Benchmarks: {len(report.benchmark_results)}")
    print(f"  Claims: {len(report.clinical_claims)}")
    print(f"  Document: {len(markdown):,} chars")

    return markdown


def step_6_audit() -> str:
    """Step 6: Generate audit trail."""
    print("\n" + "=" * 60)
    print("STEP 6: 21 CFR Part 11 Audit Trail")
    print("=" * 60)

    generator = AuditTrailGenerator(system_name="OncoTwin AI Platform")

    # Record lifecycle events
    generator.record_training_run(
        operator="ml_engineer_01",
        model_name="TumorSeg-3D",
        model_version="1.0.0",
        dataset_id="DS-NSCLC-2026-Q1",
        dataset_size=12500,
        hyperparameters={"lr": 0.001, "epochs": 100},
        final_metrics={"dice": 0.912, "sensitivity": 0.945},
    )

    generator.record_validation(
        operator="validation_team",
        model_name="TumorSeg-3D",
        model_version="1.0.0",
        test_dataset_id="TS-EXT-001",
        test_dataset_size=500,
        metrics={"dice": 0.908, "sensitivity": 0.941},
        acceptance_criteria={"dice": ">= 0.85", "sensitivity": ">= 0.90"},
        pass_fail="pass",
        reviewer="Dr. Regulatory Scientist",
    )

    generator.record_event(
        event_type=AuditEventType.DEPLOYMENT,
        operator="devops_team",
        action="Deploy TumorSeg-3D v1.0.0",
        description="Production deployment after successful validation",
    )

    generator.record_event(
        event_type=AuditEventType.ADMINISTRATIVE,
        operator="regulatory_affairs",
        action="Submission package assembled",
        description="All regulatory documents generated and assembled for De Novo submission",
    )

    report = generator.generate_report()

    print(f"  Report: {report.report_id}")
    print(f"  Records: {len(report.records)}")
    print(f"  Training Runs: {len(report.training_records)}")
    print(f"  Validations: {len(report.validation_records)}")
    print(f"  Document: {len(report.generated_markdown):,} chars")

    return report.generated_markdown


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run complete regulatory submission strategy for a De Novo AI device."""
    print("=" * 70)
    print("Example 06: Complete Multi-Component Regulatory Submission Strategy")
    print(f"Device: {DEVICE_NAME}")
    print(f"Indication: {INDICATION}")
    print("=" * 70)

    # Execute all steps
    documents = {}
    documents["classification"] = step_1_classification()
    documents["presub"] = step_2_presub()
    documents["pccp"] = step_3_pccp()
    documents["iec62304"] = step_4_iec62304()
    documents["clinical_evidence"] = step_5_evidence()
    documents["audit_trail"] = step_6_audit()

    # Summary
    print("\n" + "=" * 60)
    print("SUBMISSION PACKAGE SUMMARY")
    print("=" * 60)

    total_chars = 0
    for name, content in documents.items():
        chars = len(content)
        total_chars += chars
        print(f"  {name:25s}: {chars:>8,} characters")

    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':25s}: {total_chars:>8,} characters")
    print(f"\n  Documents generated: {len(documents)}")
    print("  Regulatory pathway: De Novo (Class II)")
    print(f"  Device: {DEVICE_NAME}")

    print("\nComplete regulatory submission strategy generation finished.")
    print("All documents are structured Markdown, ready for regulatory review.\n")


if __name__ == "__main__":
    main()
