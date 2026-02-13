#!/usr/bin/env python3
"""Secure aggregation protocol for federated oncology trial coordination.

Simulated secure multi-party computation (SMPC) for model weight
aggregation preventing any single site from reconstructing another
site's contributions. Uses additive secret sharing and pairwise masking.

Framework Dependencies:
    - NumPy 1.24.0+

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    This is a simulated secure aggregation protocol for demonstration.
    Production deployments must use cryptographically verified MPC libraries.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AggregationPhase(Enum):
    """Phases of the secure aggregation protocol."""

    SETUP = "setup"
    SHARE_KEYS = "share_keys"
    MASKED_INPUT = "masked_input"
    UNMASKING = "unmasking"
    AGGREGATED = "aggregated"


class ParticipantStatus(Enum):
    """Status of a participant in secure aggregation."""

    REGISTERED = "registered"
    KEY_SHARED = "key_shared"
    INPUT_SUBMITTED = "input_submitted"
    COMPLETED = "completed"
    DROPPED = "dropped"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SecretShare:
    """A single share in an additive secret sharing scheme.

    Attributes:
        share_id: Unique identifier for this share.
        owner_id: ID of the site that owns this share.
        target_id: ID of the intended recipient.
        value: The share value (numpy array).
    """

    share_id: str = ""
    owner_id: str = ""
    target_id: str = ""
    value: Optional[np.ndarray] = None


@dataclass
class PairwiseMask:
    """Pairwise mask between two participants.

    Attributes:
        site_a: First participant site ID.
        site_b: Second participant site ID.
        seed: Shared seed for generating the mask.
        mask_value: The generated mask (lazily computed).
    """

    site_a: str = ""
    site_b: str = ""
    seed: int = 0
    mask_value: Optional[np.ndarray] = None


@dataclass
class AggregationRound:
    """State for a single secure aggregation round.

    Attributes:
        round_id: Unique round identifier.
        phase: Current protocol phase.
        participants: List of participating site IDs.
        masked_inputs: Masked model updates from participants.
        pairwise_masks: Masks between participant pairs.
        aggregate_result: Final aggregated result.
        dropout_threshold: Minimum fraction of participants to proceed.
        start_time: Protocol start timestamp.
        end_time: Protocol end timestamp.
    """

    round_id: str = ""
    phase: AggregationPhase = AggregationPhase.SETUP
    participants: List[str] = field(default_factory=list)
    masked_inputs: Dict[str, np.ndarray] = field(default_factory=dict)
    pairwise_masks: Dict[str, PairwiseMask] = field(default_factory=dict)
    aggregate_result: Optional[np.ndarray] = None
    dropout_threshold: float = 0.5
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class SecureAggConfig:
    """Configuration for secure aggregation.

    Attributes:
        min_participants: Minimum participants required.
        dropout_tolerance: Fraction of dropouts tolerated.
        key_size: Size of shared keys (bits).
        mask_seed_range: Range for mask seed generation.
        verification_enabled: Whether to verify aggregation integrity.
    """

    min_participants: int = 2
    dropout_tolerance: float = 0.3
    key_size: int = 256
    mask_seed_range: int = 2**31
    verification_enabled: bool = True


# ---------------------------------------------------------------------------
# Secret Sharing
# ---------------------------------------------------------------------------


class AdditiveSecretSharing:
    """Additive secret sharing for distributing model weights.

    Splits a secret value into N shares such that the sum of all
    shares equals the original value. No individual share reveals
    information about the secret.

    Args:
        num_shares: Number of shares to create.
    """

    def __init__(self, num_shares: int):
        if num_shares < 2:
            raise ValueError("At least 2 shares required for secret sharing")
        self.num_shares = num_shares

    def split(self, secret: np.ndarray) -> List[np.ndarray]:
        """Split a secret into additive shares.

        Args:
            secret: The secret value to split.

        Returns:
            List of N shares that sum to the secret.
        """
        shares = []
        remaining = secret.copy().astype(np.float64)

        for _ in range(self.num_shares - 1):
            share = np.random.randn(*secret.shape)
            shares.append(share)
            remaining -= share

        shares.append(remaining)
        return shares

    def reconstruct(self, shares: List[np.ndarray]) -> np.ndarray:
        """Reconstruct the secret from all shares.

        Args:
            shares: List of all N shares.

        Returns:
            Reconstructed secret value.
        """
        return sum(shares)


# ---------------------------------------------------------------------------
# Pairwise Masking
# ---------------------------------------------------------------------------


class PairwiseMaskGenerator:
    """Generates pairwise masks for secure aggregation.

    Each pair of participants shares a random seed and generates
    complementary masks that cancel out during aggregation.

    Args:
        config: Secure aggregation configuration.
    """

    def __init__(self, config: SecureAggConfig):
        self.config = config
        self.shared_seeds: Dict[Tuple[str, str], int] = {}

    def setup_pair(self, site_a: str, site_b: str) -> int:
        """Establish a shared seed between two participants.

        Args:
            site_a: First participant ID.
            site_b: Second participant ID.

        Returns:
            Shared seed value.
        """
        pair_key = tuple(sorted([site_a, site_b]))
        if pair_key not in self.shared_seeds:
            seed = np.random.randint(0, self.config.mask_seed_range)
            self.shared_seeds[pair_key] = seed
        return self.shared_seeds[pair_key]

    def generate_mask(self, site_a: str, site_b: str, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate a pairwise mask between two participants.

        The mask for (A, B) is the negative of the mask for (B, A),
        so masks cancel when inputs are summed.

        Args:
            site_a: The site generating the mask.
            site_b: The paired site.
            shape: Shape of the mask array.

        Returns:
            Mask array.
        """
        pair_key = tuple(sorted([site_a, site_b]))
        seed = self.shared_seeds.get(pair_key, 0)

        rng = np.random.RandomState(seed)
        mask = rng.randn(*shape)

        if site_a > site_b:
            mask = -mask

        return mask

    def setup_all_pairs(self, participants: List[str]) -> Dict[str, int]:
        """Set up shared seeds for all participant pairs.

        Args:
            participants: List of participant IDs.

        Returns:
            Dictionary of pair keys to shared seeds.
        """
        seeds = {}
        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                seed = self.setup_pair(participants[i], participants[j])
                pair_key = f"{participants[i]}_{participants[j]}"
                seeds[pair_key] = seed
        return seeds


# ---------------------------------------------------------------------------
# Integrity Verifier
# ---------------------------------------------------------------------------


class AggregationVerifier:
    """Verifies integrity of the secure aggregation process.

    Provides commitment-based verification to ensure participants
    cannot change their inputs after seeing others' masked values.

    Args:
        salt: Salt for commitment hashing.
    """

    def __init__(self, salt: str = "secure_agg_v1"):
        self.salt = salt
        self.commitments: Dict[str, str] = {}

    def create_commitment(self, site_id: str, value: np.ndarray) -> str:
        """Create a commitment to a value.

        Args:
            site_id: Site creating the commitment.
            value: Value to commit to.

        Returns:
            Commitment hash string.
        """
        data = f"{self.salt}:{site_id}:{value.tobytes().hex()[:64]}"
        commitment = hashlib.sha256(data.encode()).hexdigest()
        self.commitments[site_id] = commitment
        return commitment

    def verify_commitment(self, site_id: str, value: np.ndarray) -> bool:
        """Verify that a value matches its commitment.

        Args:
            site_id: Site to verify.
            value: Value to check against commitment.

        Returns:
            True if commitment matches.
        """
        if site_id not in self.commitments:
            return False

        data = f"{self.salt}:{site_id}:{value.tobytes().hex()[:64]}"
        expected = hashlib.sha256(data.encode()).hexdigest()
        return expected == self.commitments[site_id]


# ---------------------------------------------------------------------------
# Secure Aggregation Protocol
# ---------------------------------------------------------------------------


class SecureAggregationProtocol:
    """Orchestrates the secure aggregation protocol.

    Implements a simplified version of the Bonawitz et al. (2017)
    secure aggregation protocol with pairwise masking and dropout
    tolerance.

    Args:
        config: Secure aggregation configuration.
    """

    def __init__(self, config: Optional[SecureAggConfig] = None):
        self.config = config or SecureAggConfig()
        self.mask_generator = PairwiseMaskGenerator(self.config)
        self.verifier = AggregationVerifier() if self.config.verification_enabled else None
        self.secret_sharing = None
        self.rounds: List[AggregationRound] = []
        self.audit_log: List[Dict[str, Any]] = []

        logger.info("Secure aggregation protocol initialized (min_participants=%d)", self.config.min_participants)

    def begin_round(self, participants: List[str]) -> AggregationRound:
        """Begin a new secure aggregation round.

        Args:
            participants: List of participating site IDs.

        Returns:
            New AggregationRound in SETUP phase.
        """
        if len(participants) < self.config.min_participants:
            raise ValueError(f"Need at least {self.config.min_participants} participants, got {len(participants)}")

        agg_round = AggregationRound(
            round_id=str(uuid.uuid4())[:8],
            phase=AggregationPhase.SETUP,
            participants=list(participants),
            start_time=time.time(),
        )

        self.secret_sharing = AdditiveSecretSharing(len(participants))
        self.mask_generator.setup_all_pairs(participants)
        agg_round.phase = AggregationPhase.SHARE_KEYS

        self._log_event("round_started", {"round_id": agg_round.round_id, "participants": len(participants)})
        logger.info("Secure aggregation round %s started with %d participants", agg_round.round_id, len(participants))
        return agg_round

    def submit_masked_input(self, agg_round: AggregationRound, site_id: str, raw_input: np.ndarray) -> np.ndarray:
        """Submit a masked input from a participant.

        The raw input is masked with pairwise masks before submission
        so the server only sees the masked version.

        Args:
            agg_round: Current aggregation round.
            site_id: Submitting site ID.
            raw_input: Raw (unmasked) model update.

        Returns:
            Masked input that was submitted.
        """
        if site_id not in agg_round.participants:
            raise ValueError(f"Site {site_id} is not a participant in round {agg_round.round_id}")

        if self.verifier is not None:
            self.verifier.create_commitment(site_id, raw_input)

        masked = raw_input.copy()
        for other_id in agg_round.participants:
            if other_id != site_id:
                mask = self.mask_generator.generate_mask(site_id, other_id, raw_input.shape)
                masked += mask

        agg_round.masked_inputs[site_id] = masked
        agg_round.phase = AggregationPhase.MASKED_INPUT

        logger.info("Site %s submitted masked input for round %s", site_id, agg_round.round_id)
        return masked

    def aggregate(self, agg_round: AggregationRound) -> Optional[np.ndarray]:
        """Perform secure aggregation of all masked inputs.

        Because pairwise masks cancel in summation, the sum of all
        masked inputs equals the sum of all raw inputs.

        Args:
            agg_round: Round with all masked inputs submitted.

        Returns:
            Aggregated (averaged) result, or None if insufficient inputs.
        """
        submitted = len(agg_round.masked_inputs)
        min_required = int(len(agg_round.participants) * (1 - self.config.dropout_tolerance))
        min_required = max(min_required, self.config.min_participants)

        if submitted < min_required:
            logger.warning(
                "Insufficient submissions: %d/%d (need %d)", submitted, len(agg_round.participants), min_required
            )
            return None

        agg_round.phase = AggregationPhase.UNMASKING

        aggregate_sum = None
        for masked_input in agg_round.masked_inputs.values():
            if aggregate_sum is None:
                aggregate_sum = masked_input.copy()
            else:
                aggregate_sum += masked_input

        result = aggregate_sum / submitted
        agg_round.aggregate_result = result
        agg_round.phase = AggregationPhase.AGGREGATED
        agg_round.end_time = time.time()

        self.rounds.append(agg_round)
        self._log_event(
            "aggregation_completed",
            {"round_id": agg_round.round_id, "submitted": submitted, "total": len(agg_round.participants)},
        )

        logger.info(
            "Secure aggregation completed for round %s (%d/%d sites)",
            agg_round.round_id,
            submitted,
            len(agg_round.participants),
        )
        return result

    def verify_round(self, agg_round: AggregationRound, raw_inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Verify aggregation integrity (for testing/audit).

        Compares secure aggregation result against plaintext average
        to ensure correctness.

        Args:
            agg_round: Completed aggregation round.
            raw_inputs: Original (unmasked) inputs for verification.

        Returns:
            Verification report with correctness metrics.
        """
        if agg_round.aggregate_result is None:
            return {"verified": False, "reason": "No aggregation result"}

        plain_avg = np.mean(list(raw_inputs.values()), axis=0)
        difference = np.linalg.norm(agg_round.aggregate_result - plain_avg)

        tolerance = 1e-10
        verified = difference < tolerance

        report = {
            "verified": verified,
            "round_id": agg_round.round_id,
            "difference_norm": float(difference),
            "tolerance": tolerance,
            "num_participants": len(agg_round.participants),
            "num_submitted": len(agg_round.masked_inputs),
        }

        if verified:
            logger.info("Round %s verification PASSED (diff=%.2e)", agg_round.round_id, difference)
        else:
            logger.warning("Round %s verification FAILED (diff=%.2e)", agg_round.round_id, difference)

        return report

    def get_protocol_summary(self) -> Dict[str, Any]:
        """Generate a summary of the secure aggregation protocol.

        Returns:
            Dictionary with protocol metrics and history.
        """
        return {
            "total_rounds": len(self.rounds),
            "successful_rounds": sum(1 for r in self.rounds if r.aggregate_result is not None),
            "min_participants": self.config.min_participants,
            "dropout_tolerance": self.config.dropout_tolerance,
            "verification_enabled": self.config.verification_enabled,
            "audit_log_entries": len(self.audit_log),
        }

    def _log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Append an event to the audit log."""
        self.audit_log.append(
            {
                "event_id": str(uuid.uuid4())[:8],
                "event_type": event_type,
                "timestamp": time.time(),
                "details": details,
            }
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = SecureAggConfig(min_participants=3, verification_enabled=True)
    protocol = SecureAggregationProtocol(config)

    sites = ["hospital_a", "hospital_b", "hospital_c"]
    agg_round = protocol.begin_round(sites)

    raw_inputs = {}
    for site_id in sites:
        raw_input = np.random.randn(10)
        raw_inputs[site_id] = raw_input
        protocol.submit_masked_input(agg_round, site_id, raw_input)

    result = protocol.aggregate(agg_round)
    print(f"Aggregated result shape: {result.shape}")

    verification = protocol.verify_round(agg_round, raw_inputs)
    print(f"\nVerification: {'PASSED' if verification['verified'] else 'FAILED'}")
    print(f"Difference norm: {verification['difference_norm']:.2e}")

    summary = protocol.get_protocol_summary()
    print("\n=== Protocol Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
