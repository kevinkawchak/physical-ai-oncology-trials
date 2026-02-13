"""Tests for federation.differential_privacy module.

Covers PrivacyBudget, GaussianMechanism, LaplacianMechanism,
GradientClipper, and DifferentialPrivacyEngine.

License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from federation.differential_privacy import (
    BudgetStatus,
    DifferentialPrivacyEngine,
    GaussianMechanism,
    GradientClipper,
    LaplacianMechanism,
    NoiseMechanism,
    PrivacyBudget,
    PrivacyConfig,
)


# ---------------------------------------------------------------------------
# PrivacyBudget tests
# ---------------------------------------------------------------------------


class TestPrivacyBudget:
    def test_initial_remaining(self):
        budget = PrivacyBudget(epsilon=5.0)
        assert budget.epsilon_remaining == 5.0
        assert budget.status == BudgetStatus.AVAILABLE

    def test_consume_reduces_budget(self):
        budget = PrivacyBudget(epsilon=5.0)
        assert budget.consume(2.0)
        assert budget.epsilon_remaining == 3.0
        assert budget.queries_made == 1

    def test_consume_denied_when_exhausted(self):
        budget = PrivacyBudget(epsilon=1.0)
        assert budget.consume(0.5)
        assert budget.consume(0.5)
        assert not budget.consume(0.1)
        assert budget.status == BudgetStatus.EXHAUSTED

    def test_warning_status(self):
        budget = PrivacyBudget(epsilon=10.0)
        budget.consume(9.5)
        assert budget.status == BudgetStatus.WARNING

    def test_auto_id(self):
        budget = PrivacyBudget(epsilon=1.0)
        assert len(budget.budget_id) > 0


# ---------------------------------------------------------------------------
# GaussianMechanism tests
# ---------------------------------------------------------------------------


class TestGaussianMechanism:
    def test_noise_has_correct_shape(self):
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        value = np.zeros((4, 3))
        noised = mech.add_noise(value)
        assert noised.shape == (4, 3)

    def test_noise_is_nonzero(self):
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        value = np.zeros(10)
        noised = mech.add_noise(value)
        assert not np.allclose(noised, value)

    def test_scalar_noise(self):
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        result = mech.add_noise_scalar(0.0)
        assert isinstance(result, float)

    def test_higher_epsilon_less_noise(self):
        vals = []
        for eps in [0.1, 10.0]:
            mech = GaussianMechanism(sensitivity=1.0, epsilon=eps, delta=1e-5)
            diffs = []
            for _ in range(200):
                noised = mech.add_noise_scalar(0.0)
                diffs.append(abs(noised))
            vals.append(np.mean(diffs))
        assert vals[0] > vals[1]


# ---------------------------------------------------------------------------
# LaplacianMechanism tests
# ---------------------------------------------------------------------------


class TestLaplacianMechanism:
    def test_noise_has_correct_shape(self):
        mech = LaplacianMechanism(sensitivity=1.0, epsilon=1.0)
        value = np.zeros((3, 2))
        noised = mech.add_noise(value)
        assert noised.shape == (3, 2)

    def test_scalar_noise(self):
        mech = LaplacianMechanism(sensitivity=1.0, epsilon=1.0)
        result = mech.add_noise_scalar(5.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# GradientClipper tests
# ---------------------------------------------------------------------------


class TestGradientClipper:
    def test_no_clipping_below_norm(self):
        clipper = GradientClipper(clip_norm=100.0)
        grads = {"l": np.ones(5)}
        clipped = clipper.clip(grads)
        np.testing.assert_array_almost_equal(clipped["l"], grads["l"])

    def test_clipping_above_norm(self):
        clipper = GradientClipper(clip_norm=1.0)
        grads = {"l": np.ones(10) * 10.0}
        clipped = clipper.clip(grads)
        clipped_norm = np.linalg.norm(clipped["l"])
        assert clipped_norm <= 1.0 + 1e-6

    def test_per_layer_clip(self):
        clipper = GradientClipper(clip_norm=1.0)
        grads = {"a": np.ones(5) * 5.0, "b": np.ones(3) * 0.1}
        clipped = clipper.per_layer_clip(grads)
        assert np.linalg.norm(clipped["a"]) <= 1.0 + 1e-6
        np.testing.assert_array_almost_equal(clipped["b"], grads["b"])


# ---------------------------------------------------------------------------
# DifferentialPrivacyEngine tests
# ---------------------------------------------------------------------------


class TestDifferentialPrivacyEngine:
    def test_privatize_gradients(self):
        config = PrivacyConfig(mechanism=NoiseMechanism.GAUSSIAN, epsilon_per_round=1.0)
        budget = PrivacyBudget(epsilon=10.0)
        engine = DifferentialPrivacyEngine(config, budget)
        grads = {"l": np.ones((4, 3))}
        result = engine.privatize_gradients(grads, site_id="s1")
        assert result is not None
        assert result["l"].shape == (4, 3)

    def test_privatize_gradients_exhausted(self):
        config = PrivacyConfig(mechanism=NoiseMechanism.GAUSSIAN, epsilon_per_round=5.0)
        budget = PrivacyBudget(epsilon=3.0)
        engine = DifferentialPrivacyEngine(config, budget)
        grads = {"l": np.ones(5)}
        result = engine.privatize_gradients(grads)
        assert result is None

    def test_privatize_statistic(self):
        config = PrivacyConfig(mechanism=NoiseMechanism.LAPLACIAN, epsilon_per_round=1.0)
        budget = PrivacyBudget(epsilon=10.0)
        engine = DifferentialPrivacyEngine(config, budget)
        result = engine.privatize_statistic(62.5, sensitivity=1.0, query_name="mean_age")
        assert result is not None
        assert isinstance(result, float)

    def test_privatize_histogram(self):
        config = PrivacyConfig(mechanism=NoiseMechanism.LAPLACIAN, epsilon_per_round=1.0)
        budget = PrivacyBudget(epsilon=10.0)
        engine = DifferentialPrivacyEngine(config, budget)
        counts = np.array([10, 20, 30])
        result = engine.privatize_histogram(counts)
        assert result is not None
        assert result.shape == (3,)
        assert all(result >= 0)

    def test_budget_status(self):
        config = PrivacyConfig(epsilon_per_round=1.0)
        budget = PrivacyBudget(epsilon=5.0)
        engine = DifferentialPrivacyEngine(config, budget)
        status = engine.get_budget_status()
        assert status["total_epsilon"] == 5.0
        assert status["status"] == "available"

    def test_generate_report(self):
        config = PrivacyConfig(epsilon_per_round=0.5)
        budget = PrivacyBudget(epsilon=5.0)
        engine = DifferentialPrivacyEngine(config, budget)
        engine.privatize_statistic(10.0, 1.0)
        report = engine.generate_report()
        assert report.queries == 1
        assert report.epsilon_spent == 0.5
        assert len(report.per_query_costs) == 1

    def test_laplacian_engine(self):
        config = PrivacyConfig(mechanism=NoiseMechanism.LAPLACIAN, epsilon_per_round=1.0)
        budget = PrivacyBudget(epsilon=10.0)
        engine = DifferentialPrivacyEngine(config, budget)
        grads = {"l": np.ones(5)}
        result = engine.privatize_gradients(grads)
        assert result is not None
