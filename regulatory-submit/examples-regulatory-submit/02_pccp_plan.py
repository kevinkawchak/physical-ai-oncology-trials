#!/usr/bin/env python3
"""Example 02: Predetermined Change Control Plan (PCCP) Creation.

Demonstrates generating a PCCP document for an adaptive AI oncology device,
defining pre-authorized modification boundaries, validation protocols,
and transparency requirements per FDA's August 2025 PCCP guidance.

Key Concepts:
    - PCCP document creation with default boundaries
    - Modification boundary customization
    - Validation protocol specification
    - Transparency and communication planning
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

from pccp_engine import (
    ApprovalCategory,
    ModificationBoundary,
    ModificationType,
    PCCPEngine,
    RiskLevel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate a PCCP for an adaptive AI oncology device."""
    print("=" * 70)
    print("Example 02: Predetermined Change Control Plan (PCCP)")
    print("=" * 70)

    engine = PCCPEngine(
        sponsor="Physical AI Oncology Consortium",
        regulatory_contact="Regulatory Affairs Office",
    )

    # Create PCCP document
    document = engine.create_pccp(
        device_name="OncoTwin AI Surgical Planning System",
        submission_reference="DN-2026-0042",
        ai_model_names=["TumorSeg-3D", "PathOptimizer-RL"],
        lifecycle_duration="3 years from initial De Novo authorization",
    )

    # Populate with default boundaries
    engine.populate_default_boundaries(document)

    # Add a custom boundary for indication expansion
    engine.add_custom_boundary(
        document,
        ModificationBoundary(
            modification_type=ModificationType.INDICATION_EXPANSION,
            description="Expansion to additional NSCLC sub-indications within the authorized scope",
            permitted_scope=(
                "Extension of the authorized intended use from NSCLC stages I-IIIA to include "
                "stage IIIB peripheral tumors, provided the tumor characteristics fall within "
                "the validated input parameter ranges."
            ),
            risk_level=RiskLevel.MODERATE,
            approval_category=ApprovalCategory.REQUIRES_NOTIFICATION,
            performance_thresholds={
                "dice_coefficient_minimum": 0.85,
                "expert_agreement_minimum": 0.80,
            },
            validation_requirements=[
                "Validation study on stage IIIB cohort (N >= 100)",
                "Non-inferiority comparison to existing authorized performance",
                "Subgroup analysis by tumor size and proximity to major vessels",
            ],
            monitoring_requirements=[
                "Dedicated performance monitoring for newly added sub-indication",
                "Monthly reporting for first 6 months post-expansion",
            ],
            exclusions=[
                "Expansion to cancer types other than NSCLC",
                "Expansion to metastatic disease (stage IV)",
            ],
        ),
    )

    # Populate validation protocols
    engine.populate_default_validation(document)

    # Set transparency plan
    engine.set_transparency_plan(document)

    # Generate Markdown
    markdown = engine.generate_markdown(document)

    # Output summary
    print(f"\nPCCP Document: {document.document_id}")
    print(f"Device: {document.device_name}")
    print(f"Models: {', '.join(document.ai_model_names)}")
    print(f"Lifecycle: {document.lifecycle_duration}")
    print(f"Modification Boundaries: {len(document.modification_boundaries)}")
    print(f"Validation Protocols: {len(document.validation_protocols)}")
    print(f"Document length: {len(markdown):,} characters")

    # Show boundary summary
    print("\nModification Boundaries:")
    for b in document.modification_boundaries:
        print(f"  [{b.risk_level.value:8s}] [{b.approval_category.value:30s}] {b.description[:50]}")

    print("\nPCCP generation complete.")


if __name__ == "__main__":
    main()
