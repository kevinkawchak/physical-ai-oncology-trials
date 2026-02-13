#!/usr/bin/env python3
"""Example 02: Differential Privacy Budget Demonstration.

Demonstrates configurable epsilon/delta privacy budgets with Gaussian
and Laplacian noise mechanisms applied to gradient updates and
summary statistics in a federated oncology trial.

Key Concepts:
    - Privacy budget management (epsilon accounting)
    - Gaussian vs. Laplacian noise mechanisms
    - Gradient clipping and privatization
    - Summary statistic privatization
    - Histogram noise injection
    - Privacy budget exhaustion handling

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

from federation.differential_privacy import (
    DifferentialPrivacyEngine,
    GradientClipper,
    NoiseMechanism,
    PrivacyBudget,
    PrivacyConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demonstration Functions
# ---------------------------------------------------------------------------


def demo_gradient_privatization() -> None:
    """Demonstrate gradient privatization with different epsilon values."""
    print("\n--- Gradient Privatization ---")

    layer_shapes = {"dense_1": (64, 32), "dense_2": (32, 16), "output": (16, 3)}

    for epsilon in [0.1, 0.5, 1.0, 5.0]:
        config = PrivacyConfig(
            mechanism=NoiseMechanism.GAUSSIAN,
            epsilon_per_round=epsilon,
            delta=1e-5,
            clip_norm=1.0,
        )
        budget = PrivacyBudget(epsilon=epsilon * 10)
        engine = DifferentialPrivacyEngine(config, budget)

        raw_gradients = {name: np.random.randn(*shape) * 0.1 for name, shape in layer_shapes.items()}
        private_gradients = engine.privatize_gradients(raw_gradients, site_id="demo_site")

        if private_gradients is not None:
            total_noise = sum(np.linalg.norm(raw_gradients[name] - private_gradients[name]) for name in raw_gradients)
            print(f"  epsilon={epsilon:.1f}: total noise magnitude = {total_noise:.4f}")


def demo_noise_mechanisms() -> None:
    """Compare Gaussian vs Laplacian noise mechanisms."""
    print("\n--- Gaussian vs Laplacian Noise Comparison ---")

    true_value = 62.5
    n_trials = 100

    for mechanism in [NoiseMechanism.GAUSSIAN, NoiseMechanism.LAPLACIAN]:
        config = PrivacyConfig(
            mechanism=mechanism,
            epsilon_per_round=1.0,
            delta=1e-5,
            sensitivity=1.0,
        )
        budget = PrivacyBudget(epsilon=n_trials * 1.0)
        engine = DifferentialPrivacyEngine(config, budget)

        noised_values = []
        for _ in range(n_trials):
            result = engine.privatize_statistic(true_value, sensitivity=1.0, query_name="mean_age")
            if result is not None:
                noised_values.append(result)

        noised_arr = np.array(noised_values)
        print(
            f"  {mechanism.value:>10s}: mean={noised_arr.mean():.2f}, std={noised_arr.std():.4f}, "
            f"MSE={np.mean((noised_arr - true_value) ** 2):.4f}"
        )


def demo_gradient_clipping() -> None:
    """Demonstrate gradient clipping at different norms."""
    print("\n--- Gradient Clipping ---")

    gradients = {"layer_1": np.random.randn(32, 16) * 5.0, "layer_2": np.random.randn(16, 8) * 3.0}
    original_norm = np.linalg.norm(np.concatenate([g.ravel() for g in gradients.values()]))
    print(f"  Original gradient norm: {original_norm:.4f}")

    for clip_norm in [0.5, 1.0, 5.0, 50.0]:
        clipper = GradientClipper(clip_norm)
        clipped = clipper.clip(gradients)
        clipped_norm = np.linalg.norm(np.concatenate([g.ravel() for g in clipped.values()]))
        print(f"  clip_norm={clip_norm:5.1f}: clipped gradient norm = {clipped_norm:.4f}")


def demo_histogram_privatization() -> None:
    """Demonstrate privatized histogram queries."""
    print("\n--- Histogram Privatization ---")

    enrollment_by_stage = np.array([45, 82, 63, 31])
    stage_labels = ["Stage I", "Stage II", "Stage III", "Stage IV"]

    print("  True enrollment by stage:")
    for label, count in zip(stage_labels, enrollment_by_stage):
        print(f"    {label}: {count}")

    for epsilon in [0.5, 1.0, 5.0]:
        config = PrivacyConfig(mechanism=NoiseMechanism.LAPLACIAN, epsilon_per_round=epsilon)
        budget = PrivacyBudget(epsilon=epsilon * 5)
        engine = DifferentialPrivacyEngine(config, budget)

        private_hist = engine.privatize_histogram(enrollment_by_stage, sensitivity=1.0)
        if private_hist is not None:
            error = np.abs(private_hist - enrollment_by_stage).mean()
            print(f"\n  Privatized (eps={epsilon}): {private_hist.astype(int).tolist()}, mean_error={error:.1f}")


def demo_budget_exhaustion() -> None:
    """Demonstrate privacy budget exhaustion."""
    print("\n--- Budget Exhaustion ---")

    config = PrivacyConfig(mechanism=NoiseMechanism.GAUSSIAN, epsilon_per_round=1.0)
    budget = PrivacyBudget(epsilon=3.0)
    engine = DifferentialPrivacyEngine(config, budget)

    for i in range(5):
        result = engine.privatize_statistic(50.0 + i, sensitivity=1.0, query_name=f"query_{i}")
        status = engine.get_budget_status()
        print(
            f"  Query {i + 1}: result={'%.2f' % result if result is not None else 'DENIED'}, "
            f"remaining={status['epsilon_remaining']:.2f}, status={status['status']}"
        )


def demo_privacy_report() -> None:
    """Generate a comprehensive privacy report."""
    print("\n--- Privacy Report ---")

    config = PrivacyConfig(mechanism=NoiseMechanism.GAUSSIAN, epsilon_per_round=0.5, clip_norm=1.0)
    budget = PrivacyBudget(epsilon=5.0)
    engine = DifferentialPrivacyEngine(config, budget)

    for i in range(3):
        grads = {"layer": np.random.randn(16, 8)}
        engine.privatize_gradients(grads, site_id=f"site_{i}")

    for name in ["mean_age", "tumor_size", "survival_months"]:
        engine.privatize_statistic(np.random.uniform(20, 80), sensitivity=1.0, query_name=name)

    report = engine.generate_report()
    print(f"  Report ID: {report.report_id}")
    print(f"  Total epsilon: {report.total_epsilon}")
    print(f"  Epsilon spent: {report.epsilon_spent:.2f}")
    print(f"  Epsilon remaining: {report.epsilon_remaining:.2f}")
    print(f"  Queries made: {report.queries}")
    print(f"  Mechanism: {report.mechanism}")
    print(f"  Status: {report.status}")
    print(f"  Per-query costs: {report.per_query_costs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all differential privacy demonstrations."""
    print("=" * 70)
    print("Example 02: Differential Privacy Budget Demonstration")
    print("=" * 70)

    demo_gradient_privatization()
    demo_noise_mechanisms()
    demo_gradient_clipping()
    demo_histogram_privatization()
    demo_budget_exhaustion()
    demo_privacy_report()

    print("\n" + "=" * 70)
    print("Differential privacy demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
