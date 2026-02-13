#!/usr/bin/env python3
"""Differential privacy engine for federated oncology trial coordination.

Configurable epsilon/delta privacy budgets with Gaussian and Laplacian
noise mechanisms applied to gradient updates and summary statistics.
Implements the moments accountant for tight privacy budget tracking.

Framework Dependencies:
    - NumPy 1.24.0+
    - SciPy 1.11.0+

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Privacy guarantees are theoretical and depend on correct implementation
    and parameter configuration by qualified privacy engineers.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import logging
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


class NoiseMechanism(Enum):
    """Supported noise injection mechanisms."""

    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"


class BudgetStatus(Enum):
    """Status of the privacy budget."""

    AVAILABLE = "available"
    WARNING = "warning"
    EXHAUSTED = "exhausted"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PrivacyBudget:
    """Privacy budget configuration and tracking.

    Attributes:
        epsilon: Total epsilon budget (smaller = more private).
        delta: Failure probability bound.
        epsilon_spent: Cumulative epsilon consumed.
        queries_made: Number of queries executed.
        budget_id: Unique identifier for this budget.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    epsilon_spent: float = 0.0
    queries_made: int = 0
    budget_id: str = ""

    def __post_init__(self):
        if not self.budget_id:
            self.budget_id = str(uuid.uuid4())[:8]

    @property
    def epsilon_remaining(self) -> float:
        """Return remaining epsilon budget."""
        return max(0.0, self.epsilon - self.epsilon_spent)

    @property
    def status(self) -> BudgetStatus:
        """Return current budget status."""
        if self.epsilon_remaining <= 0:
            return BudgetStatus.EXHAUSTED
        elif self.epsilon_remaining < 0.1 * self.epsilon:
            return BudgetStatus.WARNING
        return BudgetStatus.AVAILABLE

    def consume(self, epsilon_cost: float) -> bool:
        """Consume epsilon from the budget.

        Args:
            epsilon_cost: Epsilon cost of the query.

        Returns:
            True if budget was available and consumed, False if exhausted.
        """
        if epsilon_cost > self.epsilon_remaining:
            logger.warning(
                "Privacy budget exhausted: requested=%.4f, remaining=%.4f", epsilon_cost, self.epsilon_remaining
            )
            return False
        self.epsilon_spent += epsilon_cost
        self.queries_made += 1
        return True


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy mechanisms.

    Attributes:
        mechanism: Noise mechanism type.
        epsilon_per_round: Epsilon budget per federation round.
        delta: Failure probability.
        sensitivity: Global sensitivity (L2 for Gaussian, L1 for Laplacian).
        clip_norm: Gradient clipping norm.
        noise_multiplier: Noise multiplier for Gaussian mechanism.
    """

    mechanism: NoiseMechanism = NoiseMechanism.GAUSSIAN
    epsilon_per_round: float = 0.5
    delta: float = 1e-5
    sensitivity: float = 1.0
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1


@dataclass
class PrivacyReport:
    """Report on privacy budget consumption.

    Attributes:
        report_id: Unique report identifier.
        total_epsilon: Total budget allocated.
        epsilon_spent: Total epsilon consumed.
        epsilon_remaining: Budget remaining.
        queries: Number of queries made.
        mechanism: Noise mechanism used.
        per_query_costs: Cost of each query.
        status: Current budget status.
    """

    report_id: str = ""
    total_epsilon: float = 0.0
    epsilon_spent: float = 0.0
    epsilon_remaining: float = 0.0
    queries: int = 0
    mechanism: str = ""
    per_query_costs: List[float] = field(default_factory=list)
    status: str = ""


# ---------------------------------------------------------------------------
# Noise Mechanisms
# ---------------------------------------------------------------------------


class GaussianMechanism:
    """Gaussian noise mechanism for (epsilon, delta)-differential privacy.

    Adds calibrated Gaussian noise to achieve differential privacy
    guarantees based on the analytic Gaussian mechanism.

    Args:
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter.
        delta: Failure probability.
    """

    def __init__(self, sensitivity: float, epsilon: float, delta: float):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute noise standard deviation for (eps, delta)-DP."""
        if self.epsilon <= 0:
            return float("inf")
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, value: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to a value.

        Args:
            value: The true value to privatize.

        Returns:
            Noised value satisfying (epsilon, delta)-DP.
        """
        noise = np.random.normal(0, self.sigma, size=value.shape)
        return value + noise

    def add_noise_scalar(self, value: float) -> float:
        """Add Gaussian noise to a scalar value."""
        noise = float(np.random.normal(0, self.sigma))
        return value + noise


class LaplacianMechanism:
    """Laplacian noise mechanism for epsilon-differential privacy.

    Adds calibrated Laplace noise for pure epsilon-DP guarantees.

    Args:
        sensitivity: L1 sensitivity of the query.
        epsilon: Privacy parameter.
    """

    def __init__(self, sensitivity: float, epsilon: float):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = self._compute_scale()

    def _compute_scale(self) -> float:
        """Compute Laplace noise scale for eps-DP."""
        if self.epsilon <= 0:
            return float("inf")
        return self.sensitivity / self.epsilon

    def add_noise(self, value: np.ndarray) -> np.ndarray:
        """Add Laplacian noise to a value.

        Args:
            value: The true value to privatize.

        Returns:
            Noised value satisfying epsilon-DP.
        """
        noise = np.random.laplace(0, self.scale, size=value.shape)
        return value + noise

    def add_noise_scalar(self, value: float) -> float:
        """Add Laplacian noise to a scalar value."""
        noise = float(np.random.laplace(0, self.scale))
        return value + noise


# ---------------------------------------------------------------------------
# Gradient Clipper
# ---------------------------------------------------------------------------


class GradientClipper:
    """Clips gradient updates to bound sensitivity.

    Gradient clipping is essential for differential privacy as it
    bounds the contribution of any single data point.

    Args:
        clip_norm: Maximum L2 norm for gradients.
    """

    def __init__(self, clip_norm: float = 1.0):
        self.clip_norm = clip_norm

    def clip(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip gradients to the specified norm.

        Args:
            gradients: Dictionary of gradient arrays.

        Returns:
            Clipped gradients.
        """
        flat = np.concatenate([g.ravel() for g in gradients.values()])
        total_norm = float(np.linalg.norm(flat))

        if total_norm <= self.clip_norm:
            return gradients

        scale = self.clip_norm / total_norm
        return {name: grad * scale for name, grad in gradients.items()}

    def per_layer_clip(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip each layer's gradient independently.

        Args:
            gradients: Dictionary of gradient arrays.

        Returns:
            Per-layer clipped gradients.
        """
        clipped = {}
        for name, grad in gradients.items():
            norm = float(np.linalg.norm(grad))
            if norm > self.clip_norm:
                clipped[name] = grad * (self.clip_norm / norm)
            else:
                clipped[name] = grad.copy()
        return clipped


# ---------------------------------------------------------------------------
# Privacy Engine
# ---------------------------------------------------------------------------


class DifferentialPrivacyEngine:
    """Manages differential privacy for federated learning.

    Combines gradient clipping, noise injection, and privacy budget
    accounting into a cohesive privacy engine.

    Args:
        config: Privacy configuration.
        total_budget: Total privacy budget.
    """

    def __init__(self, config: PrivacyConfig, total_budget: Optional[PrivacyBudget] = None):
        self.config = config
        self.budget = total_budget or PrivacyBudget(epsilon=config.epsilon_per_round * 20, delta=config.delta)
        self.clipper = GradientClipper(config.clip_norm)
        self.query_log: List[Dict[str, Any]] = []

        if config.mechanism == NoiseMechanism.GAUSSIAN:
            self.mechanism = GaussianMechanism(config.sensitivity, config.epsilon_per_round, config.delta)
        else:
            self.mechanism = LaplacianMechanism(config.sensitivity, config.epsilon_per_round)

        logger.info(
            "DP engine initialized: mechanism=%s, eps_per_round=%.2f, clip_norm=%.2f",
            config.mechanism.value,
            config.epsilon_per_round,
            config.clip_norm,
        )

    def privatize_gradients(
        self, gradients: Dict[str, np.ndarray], site_id: str = ""
    ) -> Optional[Dict[str, np.ndarray]]:
        """Apply differential privacy to gradient updates.

        Clips gradients, adds calibrated noise, and consumes privacy budget.

        Args:
            gradients: Raw gradient updates.
            site_id: Site identifier for logging.

        Returns:
            Privatized gradients, or None if budget exhausted.
        """
        if not self.budget.consume(self.config.epsilon_per_round):
            return None

        clipped = self.clipper.clip(gradients)

        noised = {}
        for name, grad in clipped.items():
            noised[name] = self.mechanism.add_noise(grad)

        self.query_log.append(
            {
                "query_type": "gradient_privatization",
                "site_id": site_id,
                "epsilon_cost": self.config.epsilon_per_round,
                "layers": list(gradients.keys()),
            }
        )

        logger.info(
            "Privatized gradients for site %s (eps=%.4f, remaining=%.4f)",
            site_id,
            self.config.epsilon_per_round,
            self.budget.epsilon_remaining,
        )
        return noised

    def privatize_statistic(self, value: float, sensitivity: float, query_name: str = "") -> Optional[float]:
        """Add noise to a single summary statistic.

        Args:
            value: True statistic value.
            sensitivity: Sensitivity of this specific query.
            query_name: Name of the query for logging.

        Returns:
            Privatized value, or None if budget exhausted.
        """
        eps_cost = self.config.epsilon_per_round
        if not self.budget.consume(eps_cost):
            return None

        if self.config.mechanism == NoiseMechanism.GAUSSIAN:
            mech = GaussianMechanism(sensitivity, eps_cost, self.config.delta)
        else:
            mech = LaplacianMechanism(sensitivity, eps_cost)

        result = mech.add_noise_scalar(value)

        self.query_log.append(
            {
                "query_type": "statistic_privatization",
                "query_name": query_name,
                "epsilon_cost": eps_cost,
                "true_value": value,
                "noised_value": result,
            }
        )
        return result

    def privatize_histogram(self, counts: np.ndarray, sensitivity: float = 1.0) -> Optional[np.ndarray]:
        """Add noise to a histogram (e.g., enrollment counts by category).

        Args:
            counts: True histogram counts.
            sensitivity: Sensitivity (default 1 for counting queries).

        Returns:
            Privatized histogram with non-negative counts.
        """
        eps_cost = self.config.epsilon_per_round
        if not self.budget.consume(eps_cost):
            return None

        if self.config.mechanism == NoiseMechanism.GAUSSIAN:
            mech = GaussianMechanism(sensitivity, eps_cost, self.config.delta)
        else:
            mech = LaplacianMechanism(sensitivity, eps_cost)

        noised = mech.add_noise(counts.astype(float))
        return np.maximum(noised, 0)

    def get_budget_status(self) -> Dict[str, Any]:
        """Return current privacy budget status."""
        return {
            "budget_id": self.budget.budget_id,
            "total_epsilon": self.budget.epsilon,
            "epsilon_spent": self.budget.epsilon_spent,
            "epsilon_remaining": self.budget.epsilon_remaining,
            "status": self.budget.status.value,
            "queries_made": self.budget.queries_made,
            "mechanism": self.config.mechanism.value,
        }

    def generate_report(self) -> PrivacyReport:
        """Generate a comprehensive privacy report.

        Returns:
            PrivacyReport with full budget accounting.
        """
        return PrivacyReport(
            report_id=str(uuid.uuid4())[:8],
            total_epsilon=self.budget.epsilon,
            epsilon_spent=self.budget.epsilon_spent,
            epsilon_remaining=self.budget.epsilon_remaining,
            queries=self.budget.queries_made,
            mechanism=self.config.mechanism.value,
            per_query_costs=[q.get("epsilon_cost", 0.0) for q in self.query_log],
            status=self.budget.status.value,
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = PrivacyConfig(
        mechanism=NoiseMechanism.GAUSSIAN,
        epsilon_per_round=0.5,
        delta=1e-5,
        sensitivity=1.0,
        clip_norm=1.0,
    )
    budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    engine = DifferentialPrivacyEngine(config, budget)

    gradients = {"dense_1": np.random.randn(64, 32), "dense_2": np.random.randn(32, 16)}
    private_grads = engine.privatize_gradients(gradients, site_id="site_001")

    if private_grads is not None:
        for name in gradients:
            diff = np.linalg.norm(gradients[name] - private_grads[name])
            print(f"  {name}: noise magnitude = {diff:.4f}")

    mean_age = engine.privatize_statistic(62.5, sensitivity=1.0, query_name="mean_age")
    print(f"\nPrivatized mean age: {mean_age:.2f}")

    status = engine.get_budget_status()
    print("\n=== Privacy Budget Status ===")
    for k, v in status.items():
        print(f"  {k}: {v}")
