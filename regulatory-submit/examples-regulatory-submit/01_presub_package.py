#!/usr/bin/env python3
"""Example 01: FDA Pre-Submission Package Generation.

Demonstrates generating a complete FDA Pre-Sub meeting request package
for an AI-enabled surgical planning system, including device description,
AI model documentation, proposed testing protocols, and questions for FDA.

Key Concepts:
    - Pre-Sub package creation and population
    - AI model description with performance metrics
    - Testing protocol specification
    - Auto-generated FDA questions
    - Markdown document output

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
# Device Configuration
# ---------------------------------------------------------------------------

DEVICE_CONFIG = {
    "name": "OncoTwin AI Surgical Planning System",
    "category": DeviceCategory.SURGICAL_PLANNING,
    "intended_use": (
        "AI-assisted surgical planning for solid tumor resection in adult "
        "patients with non-small cell lung cancer (NSCLC), using patient-specific "
        "digital twin models derived from pre-operative CT and MRI imaging"
    ),
    "pathway": RegulatoryPathway.DE_NOVO,
    "description": (
        "The OncoTwin AI Surgical Planning System is a software-only medical device "
        "that integrates deep learning-based tumor segmentation with patient-specific "
        "digital twin simulation to generate optimized surgical resection plans."
    ),
    "indication": "Non-small cell lung cancer (NSCLC), stages I-IIIA",
}


# ---------------------------------------------------------------------------
# AI Models
# ---------------------------------------------------------------------------

AI_MODELS = [
    AIModelDescription(
        model_name="TumorSeg-3D",
        model_type="segmentation",
        architecture="3D U-Net with attention gates",
        input_modalities=["CT volumetric", "MRI T1-weighted", "MRI T2-weighted"],
        output_type="3D tumor boundary mask with confidence map",
        training_data_description="Multi-institutional dataset from 5 NCI-designated cancer centers",
        training_data_size=12500,
        validation_approach="5-fold cross-validation with external test set",
        performance_metrics={
            "dice_coefficient": 0.912,
            "sensitivity": 0.945,
            "specificity": 0.987,
            "hausdorff_95": 3.21,
        },
        locked_or_adaptive="locked",
        clinical_role="Automated tumor boundary delineation for surgical planning",
    ),
    AIModelDescription(
        model_name="PathOptimizer-RL",
        model_type="reinforcement_learning",
        architecture="Proximal Policy Optimization (PPO) with anatomical constraints",
        input_modalities=["3D tumor segmentation", "vascular anatomy", "tissue density map"],
        output_type="Ranked surgical approach paths with predicted outcomes",
        training_data_description="Simulated surgical scenarios from digital twin environments",
        training_data_size=50000,
        validation_approach="Clinical expert review of top-3 recommended paths",
        performance_metrics={
            "path_safety_score": 0.934,
            "margin_adequacy": 0.891,
            "expert_agreement": 0.856,
        },
        locked_or_adaptive="adaptive",
        clinical_role="Surgical path optimization with outcome prediction",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate a Pre-Sub package for an AI surgical planning system."""
    print("=" * 70)
    print("Example 01: FDA Pre-Submission Package Generation")
    print("=" * 70)

    # Initialize generator
    generator = PreSubGenerator(
        sponsor="Physical AI Oncology Consortium",
        contact_name="Regulatory Affairs Office",
        contact_email="regulatory@example-consortium.org",
    )

    # Create package
    package = generator.create_package(
        device_name=DEVICE_CONFIG["name"],
        device_category=DEVICE_CONFIG["category"],
        intended_use=DEVICE_CONFIG["intended_use"],
        proposed_pathway=DEVICE_CONFIG["pathway"],
        device_description=DEVICE_CONFIG["description"],
        oncology_indication=DEVICE_CONFIG["indication"],
    )

    # Add AI models
    for model in AI_MODELS:
        generator.add_ai_model(package, model)

    # Add testing protocol
    generator.add_testing_protocol(
        package,
        TestingProtocol(
            protocol_name="Retrospective Clinical Validation Study",
            objective="Validate AI surgical planning accuracy against expert surgeon consensus",
            study_design="Multi-reader multi-case retrospective study",
            sample_size_rationale="200 cases for 80% power to detect 10% improvement (alpha=0.05)",
            endpoints=[
                "Primary: Surgical margin adequacy (R0 resection rate)",
                "Secondary: Expert agreement with AI-recommended approach",
                "Secondary: Time to complete surgical planning",
            ],
            comparator="Expert surgeon consensus panel (3 thoracic surgeons)",
            subgroup_analyses=["Tumor size", "Tumor location", "Disease stage"],
            statistical_methods="Non-inferiority with 5% margin, Bonferroni correction",
        ),
    )

    # Populate risks and auto-generate questions
    generator.populate_risk_considerations(package)
    generator.auto_generate_questions(package)

    # Add a custom question
    generator.add_question(
        package,
        topic="Digital Twin Clinical Evidence",
        question_text=(
            "Does FDA agree that digital twin simulation fidelity data can "
            "serve as supportive evidence for surgical planning accuracy claims, "
            "in addition to retrospective clinical validation?"
        ),
        context="Novel use of digital twins as part of clinical evidence strategy",
        proposed_approach=(
            "Digital twin validation as supplementary evidence alongside primary retrospective clinical study"
        ),
    )

    # Generate Markdown document
    markdown = generator.generate_markdown(package)

    # Output summary
    print(f"\nPackage ID: {package.package_id}")
    print(f"Device: {package.device_name}")
    print(f"Pathway: {package.proposed_pathway.value}")
    print(f"AI Models: {len(package.ai_models)}")
    print(f"Testing Protocols: {len(package.testing_protocols)}")
    print(f"Questions for FDA: {len(package.questions)}")
    print(f"Risk Considerations: {len(package.risk_considerations)}")
    print(f"Document length: {len(markdown):,} characters")

    # Print first section of generated document
    print("\n--- Generated Document (first 800 chars) ---")
    print(markdown[:800])
    print("...\n")

    print("Pre-Sub package generation complete.")


if __name__ == "__main__":
    main()
