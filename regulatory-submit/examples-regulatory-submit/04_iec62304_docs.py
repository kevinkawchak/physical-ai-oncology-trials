#!/usr/bin/env python3
"""Example 04: IEC 62304 Software Lifecycle Documentation.

Demonstrates generating IEC 62304-compliant software lifecycle documents
including Software Development Plan (SDP), Software Requirements
Specification (SRS), Software Architecture Document (SAD), and risk
analysis matrix for an AI oncology device.

Key Concepts:
    - Software safety classification
    - Software item architecture definition
    - Requirements specification with safety tracing
    - Risk analysis with ISO 14971 integration
    - Complete document set generation

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

from iec62304_generator import (
    IEC62304Generator,
    RiskEntry,
    RiskProbability,
    RiskSeverity,
    SafetyClass,
    SoftwareItem,
    SoftwareRequirement,
    RequirementType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture Definition
# ---------------------------------------------------------------------------

SOFTWARE_ITEMS = [
    SoftwareItem(
        item_id="SWI-001",
        name="AI Inference Engine",
        description="Core AI/ML model inference pipeline for tumor segmentation and path planning",
        safety_class=SafetyClass.CLASS_C,
        dependencies=["Input Validator", "Model Registry"],
        interfaces=["DICOM Input", "Clinical Display Output"],
        soup_components=["PyTorch", "ONNX Runtime", "NumPy"],
    ),
    SoftwareItem(
        item_id="SWI-002",
        name="Input Validator",
        description="Validates incoming DICOM data for format compliance, completeness, and quality",
        safety_class=SafetyClass.CLASS_B,
        interfaces=["DICOM Input", "Validation Report"],
        soup_components=["pydicom", "NumPy"],
    ),
    SoftwareItem(
        item_id="SWI-003",
        name="Digital Twin Engine",
        description="Patient-specific simulation engine for surgical outcome prediction",
        safety_class=SafetyClass.CLASS_C,
        dependencies=["AI Inference Engine"],
        interfaces=["Simulation Input", "Outcome Predictions"],
        soup_components=["SciPy", "NumPy", "VTK"],
    ),
    SoftwareItem(
        item_id="SWI-004",
        name="Clinical Display",
        description="User interface for displaying AI results, surgical plans, and clinician interaction",
        safety_class=SafetyClass.CLASS_B,
        dependencies=["AI Inference Engine", "Digital Twin Engine"],
        interfaces=["Clinician UI", "Override Controls"],
        soup_components=["VTK", "Qt"],
    ),
    SoftwareItem(
        item_id="SWI-005",
        name="Audit Logger",
        description="21 CFR Part 11 compliant audit trail for all predictions and user actions",
        safety_class=SafetyClass.CLASS_B,
        interfaces=["Audit Database", "Compliance Reports"],
        soup_components=["SQLite", "hashlib"],
    ),
    SoftwareItem(
        item_id="SWI-006",
        name="Model Registry",
        description="Version-controlled model storage with provenance tracking",
        safety_class=SafetyClass.CLASS_B,
        interfaces=["Model Storage", "Version API"],
        soup_components=["MLflow"],
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate IEC 62304 document set for an AI oncology device."""
    print("=" * 70)
    print("Example 04: IEC 62304 Software Lifecycle Documentation")
    print("=" * 70)

    generator = IEC62304Generator(sponsor="Physical AI Oncology Consortium")

    # Create document set
    doc_set = generator.create_document_set(
        project_name="OncoTwin-SW",
        device_name="OncoTwin AI Surgical Planning System",
        safety_class=SafetyClass.CLASS_C,
        version="1.0.0",
    )

    # Add software items
    for item in SOFTWARE_ITEMS:
        generator.add_software_item(doc_set, item)

    # Add default requirements
    generator.populate_default_requirements(doc_set)

    # Add project-specific requirements
    custom_requirements = [
        SoftwareRequirement(
            req_id="SRS-011",
            requirement_type=RequirementType.FUNCTIONAL,
            title="Digital Twin Simulation",
            description="The software shall support real-time digital twin simulation of surgical outcomes based on patient-specific tumor and anatomy models.",
            acceptance_criteria="Simulation completes within 30 seconds for standard complexity cases.",
            safety_relevant=True,
            source="Clinical Requirements Document",
        ),
        SoftwareRequirement(
            req_id="SRS-012",
            requirement_type=RequirementType.PERFORMANCE,
            title="Multi-Path Ranking",
            description="The system shall generate and rank at least 3 alternative surgical approaches for clinician review.",
            acceptance_criteria="3 ranked paths with confidence scores presented within 60 seconds.",
            safety_relevant=False,
            source="User Needs Document",
        ),
    ]
    for req in custom_requirements:
        generator.add_requirement(doc_set, req)

    # Add sample risks
    generator.populate_sample_risks(doc_set)

    # Add a project-specific risk
    generator.add_risk(
        doc_set,
        RiskEntry(
            risk_id="RSK-006",
            hazard="Digital twin simulation divergence",
            hazardous_situation="Simulation predicts favorable outcome but actual patient physiology differs significantly",
            harm="Surgeon proceeds with suboptimal plan based on inaccurate simulation",
            severity=RiskSeverity.CRITICAL,
            probability=RiskProbability.REMOTE,
            software_cause="Insufficient model personalization or stale patient data",
            mitigation="Mandatory clinician review, confidence intervals on predictions, data freshness checks",
            verification_method="Clinical validation comparing predicted vs. actual outcomes",
        ),
    )

    # Generate all documents
    docs = generator.generate_all(doc_set)

    # Output summary
    print(f"\nProject: {doc_set.project_name}")
    print(f"Device: {doc_set.device_name}")
    print(f"Safety Class: IEC 62304 Class {doc_set.safety_class.value}")
    print(f"Software Items: {len(doc_set.software_items)}")
    print(f"Requirements: {len(doc_set.requirements)}")
    print(f"Risks: {len(doc_set.risks)}")
    print("\nGenerated Documents:")
    for name, content in docs.items():
        print(f"  {name.upper()}: {len(content):,} characters")

    # Print SOUP summary
    all_soup = sorted({s for item in doc_set.software_items for s in item.soup_components})
    print(f"\nSOUP Components ({len(all_soup)}): {', '.join(all_soup)}")

    # Print risk summary
    risk_levels = {}
    for risk in doc_set.risks:
        risk_levels[risk.risk_level] = risk_levels.get(risk.risk_level, 0) + 1
    print("\nRisk Summary:")
    for level, count in sorted(risk_levels.items()):
        print(f"  {level}: {count}")

    print("\nIEC 62304 document generation complete.")


if __name__ == "__main__":
    main()
