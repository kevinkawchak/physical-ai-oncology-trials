#!/usr/bin/env python3
"""Example 01: Basic Two-Site Federation.

Demonstrates the minimal federated learning setup with two simulated
clinical sites performing FedAvg aggregation on a shared tumor
classification model.

Key Concepts:
    - Site registration and model initialization
    - FedAvg aggregation across 2 sites
    - Round-by-round training with convergence tracking
    - Federation summary and audit log

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.federated_coordinator import (
    AggregationStrategy,
    FederatedCoordinator,
    FederationConfig,
    SiteConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

TUMOR_CLASSIFIER_LAYERS = {
    "conv1_weights": (16, 1, 3, 3),
    "conv1_bias": (16,),
    "conv2_weights": (32, 16, 3, 3),
    "conv2_bias": (32,),
    "fc1_weights": (128, 64),
    "fc1_bias": (64,),
    "fc2_weights": (64, 32),
    "fc2_bias": (32,),
    "output_weights": (32, 3),
    "output_bias": (3,),
}


# ---------------------------------------------------------------------------
# Site Setup
# ---------------------------------------------------------------------------


def create_sites() -> list:
    """Create two simulated clinical sites.

    Returns:
        List of SiteConfig for each site.
    """
    return [
        SiteConfig(
            site_id="site_alpha",
            site_name="Memorial Oncology Center",
            institution="Memorial Health System",
            patient_count=150,
            data_quality_score=0.95,
            compute_capability=1.0,
        ),
        SiteConfig(
            site_id="site_beta",
            site_name="University Cancer Institute",
            institution="State University Hospital",
            patient_count=120,
            data_quality_score=0.92,
            compute_capability=0.8,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a basic two-site federation example."""
    print("=" * 70)
    print("Example 01: Basic Two-Site Federation")
    print("=" * 70)

    # Configure federation
    config = FederationConfig(
        trial_name="BREAST-FED-2026",
        strategy=AggregationStrategy.FEDAVG,
        num_rounds=8,
        min_sites_per_round=2,
        convergence_threshold=1e-4,
    )
    coordinator = FederatedCoordinator(config)

    # Register sites
    sites = create_sites()
    for site in sites:
        coordinator.register_site(site)
    print(f"\nRegistered {len(coordinator.sites)} sites")

    # Initialize global model
    global_model = coordinator.initialize_global_model(TUMOR_CLASSIFIER_LAYERS)
    total_params = sum(np.prod(s) for s in TUMOR_CLASSIFIER_LAYERS.values())
    print(f"Global model initialized: {len(TUMOR_CLASSIFIER_LAYERS)} layers, {total_params:,} parameters")

    # Run federation
    print("\n--- Training Rounds ---")
    rounds = coordinator.run_federation()

    # Display results
    print("\n--- Round Summary ---")
    for r in rounds:
        status = r.status.value
        loss = r.metrics.get("avg_loss", float("nan"))
        acc = r.metrics.get("avg_accuracy", float("nan"))
        sites_in_round = len(r.participating_sites)
        print(f"  Round {r.round_number}: status={status}, sites={sites_in_round}, loss={loss:.4f}, acc={acc:.4f}")

    # Federation summary
    summary = coordinator.get_federation_summary()
    print("\n--- Federation Summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Audit log
    print(f"\nAudit log: {len(coordinator.audit_log)} entries")
    for entry in coordinator.audit_log[:5]:
        print(f"  [{entry['event_type']}] {entry['details']}")

    # Verify global model updated
    final_model = coordinator.get_global_model()
    if final_model is not None:
        print(f"\nFinal model has {len(final_model.layer_weights)} layers")
        for name, weights in final_model.layer_weights.items():
            print(f"  {name}: shape={weights.shape}, mean={weights.mean():.6f}, std={weights.std():.6f}")

    print("\n" + "=" * 70)
    print("Basic two-site federation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
