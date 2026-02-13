"""Tests for federation.secure_aggregation module.

Covers AdditiveSecretSharing, PairwiseMaskGenerator,
AggregationVerifier, and SecureAggregationProtocol.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.secure_aggregation import (
    AdditiveSecretSharing,
    AggregationPhase,
    AggregationVerifier,
    PairwiseMaskGenerator,
    SecureAggConfig,
    SecureAggregationProtocol,
)


# ---------------------------------------------------------------------------
# AdditiveSecretSharing tests
# ---------------------------------------------------------------------------


class TestAdditiveSecretSharing:
    def test_minimum_shares(self):
        with pytest.raises(ValueError, match="(?i)at least 2"):
            AdditiveSecretSharing(1)

    def test_split_produces_correct_count(self):
        sharer = AdditiveSecretSharing(3)
        secret = np.array([1.0, 2.0, 3.0])
        shares = sharer.split(secret)
        assert len(shares) == 3

    def test_reconstruct_matches_original(self):
        sharer = AdditiveSecretSharing(4)
        secret = np.array([10.0, 20.0, 30.0, 40.0])
        shares = sharer.split(secret)
        reconstructed = sharer.reconstruct(shares)
        np.testing.assert_allclose(reconstructed, secret, atol=1e-10)

    def test_individual_shares_differ_from_secret(self):
        sharer = AdditiveSecretSharing(3)
        secret = np.array([1.0, 2.0, 3.0])
        shares = sharer.split(secret)
        for share in shares:
            assert not np.allclose(share, secret)


# ---------------------------------------------------------------------------
# PairwiseMaskGenerator tests
# ---------------------------------------------------------------------------


class TestPairwiseMaskGenerator:
    def test_setup_pair(self):
        config = SecureAggConfig()
        gen = PairwiseMaskGenerator(config)
        seed = gen.setup_pair("a", "b")
        assert isinstance(seed, (int, np.integer))

    def test_same_pair_same_seed(self):
        config = SecureAggConfig()
        gen = PairwiseMaskGenerator(config)
        s1 = gen.setup_pair("a", "b")
        s2 = gen.setup_pair("b", "a")
        assert s1 == s2

    def test_masks_cancel_in_sum(self):
        config = SecureAggConfig()
        gen = PairwiseMaskGenerator(config)
        sites = ["s1", "s2", "s3"]
        gen.setup_all_pairs(sites)

        shape = (5,)
        mask_sum = np.zeros(shape)
        for site in sites:
            for other in sites:
                if other != site:
                    mask = gen.generate_mask(site, other, shape)
                    mask_sum += mask
        np.testing.assert_allclose(mask_sum, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# AggregationVerifier tests
# ---------------------------------------------------------------------------


class TestAggregationVerifier:
    def test_create_and_verify(self):
        verifier = AggregationVerifier()
        value = np.array([1.0, 2.0, 3.0])
        verifier.create_commitment("site_a", value)
        assert verifier.verify_commitment("site_a", value)

    def test_tampered_value_fails(self):
        verifier = AggregationVerifier()
        value = np.array([1.0, 2.0, 3.0])
        verifier.create_commitment("site_a", value)
        tampered = np.array([1.0, 2.0, 3.1])
        assert not verifier.verify_commitment("site_a", tampered)

    def test_unknown_site_fails(self):
        verifier = AggregationVerifier()
        assert not verifier.verify_commitment("unknown", np.array([1.0]))


# ---------------------------------------------------------------------------
# SecureAggregationProtocol tests
# ---------------------------------------------------------------------------


class TestSecureAggregationProtocol:
    def test_begin_round_minimum_participants(self):
        config = SecureAggConfig(min_participants=3)
        protocol = SecureAggregationProtocol(config)
        with pytest.raises(ValueError, match="at least 3"):
            protocol.begin_round(["s1", "s2"])

    def test_full_protocol(self):
        config = SecureAggConfig(min_participants=2, verification_enabled=True)
        protocol = SecureAggregationProtocol(config)
        sites = ["s1", "s2", "s3"]
        agg_round = protocol.begin_round(sites)

        raw_inputs = {}
        for site in sites:
            raw_input = np.random.randn(8)
            raw_inputs[site] = raw_input
            protocol.submit_masked_input(agg_round, site, raw_input)

        result = protocol.aggregate(agg_round)
        assert result is not None
        assert result.shape == (8,)
        assert agg_round.phase == AggregationPhase.AGGREGATED

    def test_verification_passes(self):
        config = SecureAggConfig(min_participants=2)
        protocol = SecureAggregationProtocol(config)
        sites = ["a", "b"]
        agg_round = protocol.begin_round(sites)

        raw_inputs = {}
        for site in sites:
            raw_input = np.random.randn(5)
            raw_inputs[site] = raw_input
            protocol.submit_masked_input(agg_round, site, raw_input)

        protocol.aggregate(agg_round)
        report = protocol.verify_round(agg_round, raw_inputs)
        assert report["verified"]

    def test_invalid_participant_rejected(self):
        protocol = SecureAggregationProtocol()
        agg_round = protocol.begin_round(["s1", "s2"])
        with pytest.raises(ValueError, match="not a participant"):
            protocol.submit_masked_input(agg_round, "intruder", np.array([1.0]))

    def test_insufficient_submissions(self):
        config = SecureAggConfig(min_participants=3, dropout_tolerance=0.0)
        protocol = SecureAggregationProtocol(config)
        sites = ["s1", "s2", "s3"]
        agg_round = protocol.begin_round(sites)
        protocol.submit_masked_input(agg_round, "s1", np.array([1.0]))
        result = protocol.aggregate(agg_round)
        assert result is None

    def test_protocol_summary(self):
        protocol = SecureAggregationProtocol()
        summary = protocol.get_protocol_summary()
        assert summary["total_rounds"] == 0
        assert summary["successful_rounds"] == 0
