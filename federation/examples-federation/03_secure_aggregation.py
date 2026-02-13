#!/usr/bin/env python3
"""Example 03: Secure Weight Aggregation.

Demonstrates simulated secure multi-party computation for model weight
aggregation, preventing any single site from reconstructing another
site's contributions.

Key Concepts:
    - Additive secret sharing
    - Pairwise masking protocol
    - Masked input submission and aggregation
    - Integrity verification (commitment scheme)
    - Dropout tolerance

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

from federation.secure_aggregation import (
    AdditiveSecretSharing,
    AggregationVerifier,
    PairwiseMaskGenerator,
    SecureAggConfig,
    SecureAggregationProtocol,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


def demo_secret_sharing() -> None:
    """Demonstrate additive secret sharing."""
    print("\n--- Additive Secret Sharing ---")

    secret = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"  Original secret: {secret}")

    for num_shares in [2, 3, 5]:
        sharer = AdditiveSecretSharing(num_shares)
        shares = sharer.split(secret)

        print(f"\n  Splitting into {num_shares} shares:")
        for i, share in enumerate(shares):
            print(f"    Share {i + 1}: [{', '.join(f'{v:.4f}' for v in share)}]")

        reconstructed = sharer.reconstruct(shares)
        error = np.linalg.norm(secret - reconstructed)
        print(f"    Reconstructed: {reconstructed}")
        print(f"    Reconstruction error: {error:.2e}")


def demo_pairwise_masking() -> None:
    """Demonstrate pairwise masking cancellation."""
    print("\n--- Pairwise Masking ---")

    config = SecureAggConfig(min_participants=3)
    mask_gen = PairwiseMaskGenerator(config)

    sites = ["site_a", "site_b", "site_c"]
    seeds = mask_gen.setup_all_pairs(sites)
    print(f"  Shared seeds: {seeds}")

    shape = (4,)
    raw_inputs = {site: np.random.randn(*shape) for site in sites}

    print("\n  Raw inputs:")
    for site, raw in raw_inputs.items():
        print(f"    {site}: {raw}")

    masked_inputs = {}
    for site in sites:
        masked = raw_inputs[site].copy()
        for other in sites:
            if other != site:
                mask = mask_gen.generate_mask(site, other, shape)
                masked += mask
        masked_inputs[site] = masked

    print("\n  Masked inputs (individual values are meaningless):")
    for site, masked in masked_inputs.items():
        print(f"    {site}: {masked}")

    raw_sum = sum(raw_inputs.values())
    masked_sum = sum(masked_inputs.values())
    cancellation_error = np.linalg.norm(raw_sum - masked_sum)
    print(f"\n  Sum of raw inputs:    {raw_sum}")
    print(f"  Sum of masked inputs: {masked_sum}")
    print(f"  Cancellation error:   {cancellation_error:.2e}")


def demo_full_protocol() -> None:
    """Demonstrate the full secure aggregation protocol."""
    print("\n--- Full Secure Aggregation Protocol ---")

    config = SecureAggConfig(
        min_participants=3,
        dropout_tolerance=0.3,
        verification_enabled=True,
    )
    protocol = SecureAggregationProtocol(config)

    sites = ["hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"]
    agg_round = protocol.begin_round(sites)
    print(f"  Round {agg_round.round_id} started with {len(sites)} sites")

    model_size = 20
    raw_inputs = {}
    for site_id in sites:
        raw_input = np.random.randn(model_size) * 0.01
        raw_inputs[site_id] = raw_input
        protocol.submit_masked_input(agg_round, site_id, raw_input)
        print(f"  {site_id}: submitted masked input (norm={np.linalg.norm(raw_input):.4f})")

    result = protocol.aggregate(agg_round)
    if result is not None:
        print(f"\n  Aggregated result norm: {np.linalg.norm(result):.4f}")

        verification = protocol.verify_round(agg_round, raw_inputs)
        print(f"  Verification: {'PASSED' if verification['verified'] else 'FAILED'}")
        print(f"  Difference norm: {verification['difference_norm']:.2e}")

    summary = protocol.get_protocol_summary()
    print(f"\n  Protocol summary: {summary}")


def demo_dropout_tolerance() -> None:
    """Demonstrate protocol behavior with site dropouts."""
    print("\n--- Dropout Tolerance ---")

    config = SecureAggConfig(
        min_participants=2,
        dropout_tolerance=0.5,
        verification_enabled=False,
    )
    protocol = SecureAggregationProtocol(config)

    sites = ["site_1", "site_2", "site_3", "site_4"]
    agg_round = protocol.begin_round(sites)

    model_size = 10
    active_sites = ["site_1", "site_2", "site_3"]
    for site_id in active_sites:
        raw_input = np.random.randn(model_size)
        protocol.submit_masked_input(agg_round, site_id, raw_input)

    print(f"  Submitted: {len(active_sites)}/{len(sites)} sites (site_4 dropped)")

    result = protocol.aggregate(agg_round)
    if result is not None:
        print(f"  Aggregation succeeded with {len(active_sites)} sites")
        print(f"  Result norm: {np.linalg.norm(result):.4f}")
    else:
        print("  Aggregation failed: insufficient submissions")


def demo_commitment_verification() -> None:
    """Demonstrate commitment-based integrity verification."""
    print("\n--- Commitment Verification ---")

    verifier = AggregationVerifier(salt="demo_v1")

    value = np.array([1.0, 2.0, 3.0])
    commitment = verifier.create_commitment("site_a", value)
    print(f"  Commitment for site_a: {commitment[:16]}...")

    valid = verifier.verify_commitment("site_a", value)
    print(f"  Verify with original value: {valid}")

    tampered = np.array([1.0, 2.0, 3.1])
    invalid = verifier.verify_commitment("site_a", tampered)
    print(f"  Verify with tampered value: {invalid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all secure aggregation demonstrations."""
    print("=" * 70)
    print("Example 03: Secure Weight Aggregation")
    print("=" * 70)

    demo_secret_sharing()
    demo_pairwise_masking()
    demo_full_protocol()
    demo_dropout_tolerance()
    demo_commitment_verification()

    print("\n" + "=" * 70)
    print("Secure aggregation demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
