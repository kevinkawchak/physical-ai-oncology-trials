#!/usr/bin/env python3
"""Example 03: 510(k) / De Novo Pathway Decision Support.

Demonstrates the classification advisor for determining the appropriate
FDA regulatory pathway for two different AI oncology devices: a novel
surgical planning system (De Novo) and a CADe device with a predicate
(510(k)).

Key Concepts:
    - Device characteristics specification
    - Predicate device analysis
    - Automated pathway recommendation
    - Decision factor documentation
    - Markdown recommendation output

DISCLAIMER: RESEARCH USE ONLY. Not approved for regulatory decision-making.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classification_advisor import (
    AIDeviceType,
    ClassificationAdvisor,
    DeviceCharacteristics,
    PredicateAnalysis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device Profiles
# ---------------------------------------------------------------------------

DEVICES = [
    {
        "name": "OncoTwin AI Surgical Planning System",
        "type": AIDeviceType.TREATMENT_PLANNING,
        "intended_use": "AI-assisted surgical planning for NSCLC tumor resection",
        "indication": "Non-small cell lung cancer (NSCLC)",
        "software_only": True,
        "clinician_in_loop": True,
        "novel_algorithm": True,
        "training_data_size": 12500,
        "life_threatening": True,
        "predicates": [],
    },
    {
        "name": "OncoDetect CT Lung Nodule Detector",
        "type": AIDeviceType.CADe,
        "intended_use": "Computer-aided detection of pulmonary nodules on CT imaging",
        "indication": "Lung cancer screening",
        "software_only": True,
        "clinician_in_loop": True,
        "novel_algorithm": False,
        "training_data_size": 25000,
        "life_threatening": False,
        "predicates": [
            PredicateAnalysis(
                predicate_name="Riverain ClearRead CT",
                predicate_510k_number="K201501",
                same_intended_use=True,
                same_technology=True,
                predicate_has_ai=True,
                performance_comparison_feasible=True,
            ),
            PredicateAnalysis(
                predicate_name="Optellum Virtual Nodule Clinic",
                predicate_510k_number="K212639",
                same_intended_use=True,
                same_technology=True,
                different_questions_of_safety_efficacy=True,
                predicate_has_ai=True,
                notes="Different questions of S/E due to risk stratification feature",
            ),
        ],
    },
    {
        "name": "RoboSurgeon AI Guidance Module",
        "type": AIDeviceType.ROBOTIC_CONTROL,
        "intended_use": "AI-guided robotic surgical assistance for tumor resection",
        "indication": "Solid tumor resection (multiple sites)",
        "software_only": False,
        "patient_contact": True,
        "autonomous_operation": True,
        "clinician_in_loop": True,
        "novel_algorithm": True,
        "training_data_size": 5000,
        "life_threatening": True,
        "predicates": [],
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Analyze three devices and produce pathway recommendations."""
    print("=" * 70)
    print("Example 03: 510(k) / De Novo Pathway Decision Support")
    print("=" * 70)

    advisor = ClassificationAdvisor()

    for i, device in enumerate(DEVICES, 1):
        print(f"\n{'─' * 60}")
        print(f"Device {i}: {device['name']}")
        print(f"{'─' * 60}")

        characteristics = DeviceCharacteristics(
            device_name=device["name"],
            device_type=device["type"],
            intended_use=device["intended_use"],
            oncology_indication=device["indication"],
            software_only=device.get("software_only", True),
            patient_contact=device.get("patient_contact", False),
            autonomous_operation=device.get("autonomous_operation", False),
            clinician_in_loop=device.get("clinician_in_loop", True),
            locked_algorithm=not device.get("novel_algorithm", False),
            novel_algorithm=device.get("novel_algorithm", False),
            training_data_size=device.get("training_data_size", 0),
            life_threatening_condition=device.get("life_threatening", False),
        )

        recommendation = advisor.analyze(
            characteristics,
            predicate_analyses=device.get("predicates", []),
        )

        markdown = advisor.generate_markdown(recommendation)

        print(f"  Recommended Pathway: {recommendation.recommended_pathway.value.upper()}")
        print(f"  Device Class: {recommendation.recommended_risk_class.value}")
        print(f"  SW Safety Class: IEC 62304 Class {recommendation.software_risk_level.value}")
        print(f"  Confidence: {recommendation.confidence}")
        print(f"  Decision Factors: {len(recommendation.decision_factors)}")

        if recommendation.special_considerations:
            print("  Special Considerations:")
            for sc in recommendation.special_considerations:
                print(f"    - {sc[:70]}")

        print(f"  Next Steps: {len(recommendation.recommended_next_steps)}")
        print(f"  Document length: {len(markdown):,} characters")

    print(f"\n{'=' * 70}")
    print("Classification analysis complete for all devices.")


if __name__ == "__main__":
    main()
