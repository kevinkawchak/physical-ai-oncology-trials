"""Tests for federation.federated_coordinator module.

Covers FederatedCoordinator, aggregation strategies (FedAvg, FedProx,
SCAFFOLD), ModelWeights, SiteConfig, round execution, convergence,
and audit logging.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.federated_coordinator import (
    AggregationStrategy,
    FederatedCoordinator,
    FederationConfig,
    ModelWeights,
    RoundStatus,
    SimulatedLocalTrainer,
    SiteConfig,
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_scaffold,
)


# ---------------------------------------------------------------------------
# ModelWeights tests
# ---------------------------------------------------------------------------


class TestModelWeights:
    def test_empty_flatten(self):
        mw = ModelWeights()
        flat = mw.flatten()
        assert flat.shape == (0,)

    def test_flatten_unflatten_roundtrip(self):
        mw = ModelWeights(layer_weights={"a": np.ones((2, 3)), "b": np.ones((4,))})
        flat = mw.flatten()
        assert flat.shape == (10,)

        template = {"a": (2, 3), "b": (4,)}
        mw2 = ModelWeights()
        mw2.unflatten(flat, template)
        np.testing.assert_array_equal(mw2.layer_weights["a"], np.ones((2, 3)))
        np.testing.assert_array_equal(mw2.layer_weights["b"], np.ones((4,)))


# ---------------------------------------------------------------------------
# SiteConfig tests
# ---------------------------------------------------------------------------


class TestSiteConfig:
    def test_defaults(self):
        sc = SiteConfig()
        assert sc.patient_count == 0
        assert sc.data_quality_score == 1.0
        assert sc.compute_capability == 1.0


# ---------------------------------------------------------------------------
# Aggregation strategy tests
# ---------------------------------------------------------------------------


class TestAggregateFedAvg:
    def test_equal_weight_average(self):
        updates = {
            "s1": ModelWeights(layer_weights={"l": np.array([1.0, 2.0])}),
            "s2": ModelWeights(layer_weights={"l": np.array([3.0, 4.0])}),
        }
        configs = {
            "s1": SiteConfig(patient_count=100),
            "s2": SiteConfig(patient_count=100),
        }
        result = aggregate_fedavg(updates, configs)
        np.testing.assert_allclose(result.layer_weights["l"], [2.0, 3.0])

    def test_weighted_average(self):
        updates = {
            "s1": ModelWeights(layer_weights={"l": np.array([0.0])}),
            "s2": ModelWeights(layer_weights={"l": np.array([10.0])}),
        }
        configs = {
            "s1": SiteConfig(patient_count=100),
            "s2": SiteConfig(patient_count=300),
        }
        result = aggregate_fedavg(updates, configs)
        np.testing.assert_allclose(result.layer_weights["l"], [7.5])

    def test_metadata_set(self):
        updates = {"s1": ModelWeights(layer_weights={"l": np.array([1.0])})}
        configs = {"s1": SiteConfig(patient_count=50)}
        result = aggregate_fedavg(updates, configs)
        assert result.metadata["strategy"] == "fedavg"
        assert result.metadata["num_sites"] == 1


class TestAggregateFedProx:
    def test_proximal_adjustment(self):
        global_w = ModelWeights(layer_weights={"l": np.array([0.0])})
        updates = {
            "s1": ModelWeights(layer_weights={"l": np.array([1.0])}),
            "s2": ModelWeights(layer_weights={"l": np.array([1.0])}),
        }
        configs = {
            "s1": SiteConfig(patient_count=100),
            "s2": SiteConfig(patient_count=100),
        }
        result = aggregate_fedprox(updates, configs, global_w, mu=0.1)
        assert result.layer_weights["l"][0] < 1.0
        assert result.metadata["strategy"] == "fedprox"


class TestAggregateScaffold:
    def test_returns_updated_control(self):
        updates = {
            "s1": ModelWeights(layer_weights={"l": np.array([1.0])}),
            "s2": ModelWeights(layer_weights={"l": np.array([2.0])}),
        }
        configs = {
            "s1": SiteConfig(patient_count=100),
            "s2": SiteConfig(patient_count=100),
        }
        control_variates = {
            "s1": ModelWeights(layer_weights={"l": np.array([0.1])}),
            "s2": ModelWeights(layer_weights={"l": np.array([0.2])}),
        }
        global_control = ModelWeights(layer_weights={"l": np.array([0.0])})
        result, new_control = aggregate_scaffold(updates, configs, control_variates, global_control)
        assert "l" in result.layer_weights
        assert "l" in new_control.layer_weights
        assert result.metadata["strategy"] == "scaffold"


# ---------------------------------------------------------------------------
# SimulatedLocalTrainer tests
# ---------------------------------------------------------------------------


class TestSimulatedLocalTrainer:
    def test_train_returns_updated_weights(self):
        sc = SiteConfig(site_id="test", patient_count=100)
        trainer = SimulatedLocalTrainer(sc)
        global_w = ModelWeights(layer_weights={"l": np.zeros((4, 3))})
        updated, metrics = trainer.train(global_w, round_number=0)
        assert "l" in updated.layer_weights
        assert updated.layer_weights["l"].shape == (4, 3)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_accuracy_bounded(self):
        sc = SiteConfig(site_id="test", patient_count=100)
        trainer = SimulatedLocalTrainer(sc)
        global_w = ModelWeights(layer_weights={"l": np.zeros((2, 2))})
        _, metrics = trainer.train(global_w, round_number=100)
        assert metrics["accuracy"] <= 0.98


# ---------------------------------------------------------------------------
# FederatedCoordinator tests
# ---------------------------------------------------------------------------


class TestFederatedCoordinator:
    def test_register_site(self):
        config = FederationConfig(trial_name="TEST", num_rounds=1)
        coord = FederatedCoordinator(config)
        sid = coord.register_site(SiteConfig(site_name="Hospital A", patient_count=50))
        assert sid in coord.sites
        assert len(coord.audit_log) == 1

    def test_auto_assigns_site_id(self):
        config = FederationConfig(trial_name="TEST", num_rounds=1)
        coord = FederatedCoordinator(config)
        sid = coord.register_site(SiteConfig(site_name="Test"))
        assert sid.startswith("site_")

    def test_initialize_global_model(self):
        config = FederationConfig(trial_name="TEST")
        coord = FederatedCoordinator(config)
        shapes = {"a": (4, 3), "b": (3,)}
        model = coord.initialize_global_model(shapes)
        assert model.layer_weights["a"].shape == (4, 3)
        assert model.layer_weights["b"].shape == (3,)

    def test_execute_round_requires_model(self):
        config = FederationConfig(trial_name="TEST")
        coord = FederatedCoordinator(config)
        with pytest.raises(ValueError, match="not initialized"):
            coord.execute_round(0)

    def test_execute_round_minimum_sites(self):
        config = FederationConfig(trial_name="TEST", num_rounds=1, min_sites_per_round=3)
        coord = FederatedCoordinator(config)
        coord.register_site(SiteConfig(patient_count=50))
        coord.register_site(SiteConfig(patient_count=50))
        coord.initialize_global_model({"l": (2, 2)})
        result = coord.execute_round(0)
        assert result.status == RoundStatus.FAILED

    def test_run_federation_completes(self):
        config = FederationConfig(trial_name="TEST", num_rounds=3, min_sites_per_round=2)
        coord = FederatedCoordinator(config)
        coord.register_site(SiteConfig(patient_count=100))
        coord.register_site(SiteConfig(patient_count=100))
        coord.initialize_global_model({"l": (4, 2)})
        rounds = coord.run_federation()
        assert len(rounds) >= 1
        completed = [r for r in rounds if r.status == RoundStatus.COMPLETED]
        assert len(completed) >= 1

    def test_get_federation_summary(self):
        config = FederationConfig(trial_name="SUMMARY-TEST", num_rounds=2, min_sites_per_round=2)
        coord = FederatedCoordinator(config)
        coord.register_site(SiteConfig(patient_count=80))
        coord.register_site(SiteConfig(patient_count=80))
        coord.initialize_global_model({"l": (3, 2)})
        coord.run_federation()
        summary = coord.get_federation_summary()
        assert summary["trial_name"] == "SUMMARY-TEST"
        assert summary["total_sites"] == 2
        assert summary["completed_rounds"] >= 1

    def test_get_global_model_is_copy(self):
        config = FederationConfig(trial_name="TEST", num_rounds=1, min_sites_per_round=2)
        coord = FederatedCoordinator(config)
        coord.register_site(SiteConfig(patient_count=50))
        coord.register_site(SiteConfig(patient_count=50))
        coord.initialize_global_model({"l": (2,)})
        coord.run_federation()
        m1 = coord.get_global_model()
        m2 = coord.get_global_model()
        assert m1 is not m2

    def test_select_sites_for_round(self):
        config = FederationConfig(trial_name="TEST", site_fraction=0.5, min_sites_per_round=2)
        coord = FederatedCoordinator(config)
        for i in range(6):
            coord.register_site(SiteConfig(patient_count=50 + i * 10, data_quality_score=0.9))
        selected = coord.select_sites_for_round()
        assert len(selected) >= 2
        assert len(selected) <= 6

    def test_fedprox_strategy(self):
        config = FederationConfig(
            trial_name="FEDPROX-TEST",
            strategy=AggregationStrategy.FEDPROX,
            num_rounds=2,
            min_sites_per_round=2,
            proximal_mu=0.05,
        )
        coord = FederatedCoordinator(config)
        coord.register_site(SiteConfig(patient_count=100))
        coord.register_site(SiteConfig(patient_count=100))
        coord.initialize_global_model({"l": (4, 2)})
        rounds = coord.run_federation()
        assert len(rounds) >= 1

    def test_scaffold_strategy(self):
        config = FederationConfig(
            trial_name="SCAFFOLD-TEST",
            strategy=AggregationStrategy.SCAFFOLD,
            num_rounds=2,
            min_sites_per_round=2,
        )
        coord = FederatedCoordinator(config)
        coord.register_site(SiteConfig(patient_count=100))
        coord.register_site(SiteConfig(patient_count=100))
        coord.initialize_global_model({"l": (4, 2)})
        rounds = coord.run_federation()
        assert len(rounds) >= 1
