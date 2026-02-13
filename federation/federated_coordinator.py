#!/usr/bin/env python3
"""Federated learning coordinator for multi-site oncology trial coordination.

Production-ready federated learning orchestration engine supporting
FedAvg, FedProx, and SCAFFOLD aggregation strategies across N simulated
clinical sites without sharing raw patient data.

Framework Dependencies:
    - NumPy 1.24.0+
    - SciPy 1.11.0+

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Federated learning results must be validated by qualified biostatisticians
    before any clinical interpretation.

Version: 1.0.0
Last Updated: February 2026
License: MIT
"""

from __future__ import annotations

import copy
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


class AggregationStrategy(Enum):
    """Supported federated aggregation strategies."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"


class SiteStatus(Enum):
    """Status of a participating site."""

    REGISTERED = "registered"
    ACTIVE = "active"
    TRAINING = "training"
    COMPLETED = "completed"
    DROPPED = "dropped"


class RoundStatus(Enum):
    """Status of a federation round."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ModelWeights:
    """Container for neural network model weights.

    Attributes:
        layer_weights: Dictionary mapping layer names to weight arrays.
        metadata: Additional metadata (e.g., training loss, epochs).
    """

    layer_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> np.ndarray:
        """Flatten all layer weights into a single 1-D vector."""
        if not self.layer_weights:
            return np.array([])
        return np.concatenate([w.ravel() for w in self.layer_weights.values()])

    def unflatten(self, flat: np.ndarray, template: Dict[str, Tuple[int, ...]]) -> None:
        """Restore layer weights from a flat vector using shape template."""
        offset = 0
        for name, shape in template.items():
            size = int(np.prod(shape))
            self.layer_weights[name] = flat[offset : offset + size].reshape(shape)
            offset += size


@dataclass
class SiteConfig:
    """Configuration for a participating clinical site.

    Attributes:
        site_id: Unique site identifier.
        site_name: Human-readable site name.
        institution: Affiliated institution.
        patient_count: Number of patients at this site.
        data_quality_score: Quality metric (0.0–1.0).
        compute_capability: Relative compute capability score.
    """

    site_id: str = ""
    site_name: str = ""
    institution: str = ""
    patient_count: int = 0
    data_quality_score: float = 1.0
    compute_capability: float = 1.0


@dataclass
class FederationRound:
    """State for a single federation training round.

    Attributes:
        round_id: Unique round identifier.
        round_number: Sequential round number.
        status: Current round status.
        participating_sites: List of site IDs in this round.
        site_updates: Model updates received from sites.
        aggregated_weights: Result of aggregation.
        metrics: Round-level metrics (loss, accuracy).
        start_time: Round start timestamp.
        end_time: Round end timestamp.
    """

    round_id: str = ""
    round_number: int = 0
    status: RoundStatus = RoundStatus.PENDING
    participating_sites: List[str] = field(default_factory=list)
    site_updates: Dict[str, ModelWeights] = field(default_factory=dict)
    aggregated_weights: Optional[ModelWeights] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class FederationConfig:
    """Configuration for the federated learning coordinator.

    Attributes:
        federation_id: Unique federation identifier.
        trial_name: Name of the clinical trial.
        strategy: Aggregation strategy to use.
        num_rounds: Total number of training rounds.
        min_sites_per_round: Minimum sites required per round.
        site_fraction: Fraction of sites selected per round.
        proximal_mu: FedProx proximal term coefficient.
        learning_rate: Global learning rate.
        convergence_threshold: Early stopping threshold.
    """

    federation_id: str = ""
    trial_name: str = ""
    strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    num_rounds: int = 10
    min_sites_per_round: int = 2
    site_fraction: float = 1.0
    proximal_mu: float = 0.01
    learning_rate: float = 0.01
    convergence_threshold: float = 1e-4


# ---------------------------------------------------------------------------
# Simulated Local Trainer
# ---------------------------------------------------------------------------


class SimulatedLocalTrainer:
    """Simulates local model training at a clinical site.

    This trainer generates synthetic gradient updates that mimic
    real local SGD training without requiring actual patient data.

    Args:
        site_config: Configuration for the site.
        noise_scale: Scale of simulated gradient noise.
    """

    def __init__(self, site_config: SiteConfig, noise_scale: float = 0.1):
        self.site_config = site_config
        self.noise_scale = noise_scale
        self.local_epochs = 5
        self.training_history: List[Dict[str, float]] = []

    def train(self, global_weights: ModelWeights, round_number: int) -> Tuple[ModelWeights, Dict[str, float]]:
        """Simulate local training and return updated weights.

        Args:
            global_weights: Current global model weights.
            round_number: Current federation round number.

        Returns:
            Tuple of (updated_weights, training_metrics).
        """
        updated = ModelWeights(
            layer_weights={},
            metadata={"site_id": self.site_config.site_id, "round": round_number},
        )

        total_loss = 0.0
        for name, weights in global_weights.layer_weights.items():
            gradient = np.random.randn(*weights.shape) * self.noise_scale
            scale = self.site_config.patient_count / max(self.site_config.patient_count, 1)
            updated.layer_weights[name] = weights - 0.01 * gradient * scale
            total_loss += float(np.mean(gradient**2))

        loss = total_loss / max(len(global_weights.layer_weights), 1)
        accuracy = min(0.5 + 0.05 * round_number + np.random.uniform(-0.02, 0.02), 0.98)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "samples_used": self.site_config.patient_count,
            "local_epochs": self.local_epochs,
        }
        self.training_history.append(metrics)

        logger.info(
            "Site %s completed local training: loss=%.4f, accuracy=%.4f",
            self.site_config.site_id,
            loss,
            accuracy,
        )
        return updated, metrics


# ---------------------------------------------------------------------------
# Aggregation Functions
# ---------------------------------------------------------------------------


def aggregate_fedavg(
    site_updates: Dict[str, ModelWeights],
    site_configs: Dict[str, SiteConfig],
) -> ModelWeights:
    """Federated Averaging (McMahan et al., 2017).

    Weighted average of model updates proportional to each site's
    patient count.

    Args:
        site_updates: Per-site model weight updates.
        site_configs: Per-site configuration (for weighting).

    Returns:
        Aggregated global model weights.
    """
    total_samples = sum(site_configs[sid].patient_count for sid in site_updates)
    if total_samples == 0:
        total_samples = len(site_updates)

    layer_names = list(next(iter(site_updates.values())).layer_weights.keys())
    aggregated = ModelWeights()

    for layer in layer_names:
        weighted_sum = np.zeros_like(next(iter(site_updates.values())).layer_weights[layer])
        for site_id, update in site_updates.items():
            weight = (
                site_configs[site_id].patient_count / total_samples if total_samples > 0 else 1.0 / len(site_updates)
            )
            weighted_sum += weight * update.layer_weights[layer]
        aggregated.layer_weights[layer] = weighted_sum

    aggregated.metadata = {"strategy": "fedavg", "num_sites": len(site_updates), "total_samples": total_samples}
    return aggregated


def aggregate_fedprox(
    site_updates: Dict[str, ModelWeights],
    site_configs: Dict[str, SiteConfig],
    global_weights: ModelWeights,
    mu: float = 0.01,
) -> ModelWeights:
    """FedProx aggregation (Li et al., 2020).

    Same weighted average as FedAvg but accounts for the proximal
    regularization term applied during local training.

    Args:
        site_updates: Per-site model weight updates.
        site_configs: Per-site configuration.
        global_weights: Previous global weights (for proximal term).
        mu: Proximal regularization coefficient.

    Returns:
        Aggregated global model weights with proximal adjustment.
    """
    avg = aggregate_fedavg(site_updates, site_configs)

    for layer in avg.layer_weights:
        if layer in global_weights.layer_weights:
            proximal_correction = mu * (avg.layer_weights[layer] - global_weights.layer_weights[layer])
            avg.layer_weights[layer] = avg.layer_weights[layer] - proximal_correction

    avg.metadata["strategy"] = "fedprox"
    avg.metadata["mu"] = mu
    return avg


def aggregate_scaffold(
    site_updates: Dict[str, ModelWeights],
    site_configs: Dict[str, SiteConfig],
    control_variates: Dict[str, ModelWeights],
    global_control: ModelWeights,
) -> Tuple[ModelWeights, ModelWeights]:
    """SCAFFOLD aggregation (Karimireddy et al., 2020).

    Variance-reduced federated optimization using control variates
    to correct for client drift.

    Args:
        site_updates: Per-site model weight updates.
        site_configs: Per-site configuration.
        control_variates: Per-site control variates.
        global_control: Global control variate.

    Returns:
        Tuple of (aggregated_weights, updated_global_control).
    """
    avg = aggregate_fedavg(site_updates, site_configs)

    new_global_control = ModelWeights()
    for layer in avg.layer_weights:
        variate_sum = np.zeros_like(avg.layer_weights[layer])
        count = 0
        for site_id in site_updates:
            if site_id in control_variates and layer in control_variates[site_id].layer_weights:
                variate_sum += control_variates[site_id].layer_weights[layer]
                count += 1
        if count > 0:
            new_global_control.layer_weights[layer] = variate_sum / count
        else:
            new_global_control.layer_weights[layer] = np.zeros_like(avg.layer_weights[layer])

    for layer in avg.layer_weights:
        if layer in global_control.layer_weights and layer in new_global_control.layer_weights:
            correction = new_global_control.layer_weights[layer] - global_control.layer_weights[layer]
            avg.layer_weights[layer] = avg.layer_weights[layer] + correction

    avg.metadata["strategy"] = "scaffold"
    return avg, new_global_control


# ---------------------------------------------------------------------------
# Federated Coordinator
# ---------------------------------------------------------------------------


class FederatedCoordinator:
    """Orchestrates federated learning across multiple clinical sites.

    Manages site registration, round execution, model aggregation,
    and convergence tracking for multi-site oncology trials.

    Args:
        config: Federation configuration.
    """

    def __init__(self, config: FederationConfig):
        self.config = config
        if not config.federation_id:
            self.config.federation_id = str(uuid.uuid4())[:8]
        self.sites: Dict[str, SiteConfig] = {}
        self.trainers: Dict[str, SimulatedLocalTrainer] = {}
        self.global_weights: Optional[ModelWeights] = None
        self.rounds: List[FederationRound] = []
        self.control_variates: Dict[str, ModelWeights] = {}
        self.global_control: ModelWeights = ModelWeights()
        self.audit_log: List[Dict[str, Any]] = []

        logger.info(
            "Federation %s initialized: strategy=%s, rounds=%d",
            self.config.federation_id,
            self.config.strategy.value,
            self.config.num_rounds,
        )

    def register_site(self, site_config: SiteConfig) -> str:
        """Register a clinical site for federation.

        Args:
            site_config: Site configuration.

        Returns:
            The site ID assigned.
        """
        if not site_config.site_id:
            site_config.site_id = f"site_{len(self.sites):03d}"

        self.sites[site_config.site_id] = site_config
        self.trainers[site_config.site_id] = SimulatedLocalTrainer(site_config)

        self._log_event("site_registered", {"site_id": site_config.site_id, "institution": site_config.institution})

        logger.info("Registered site: %s (%s)", site_config.site_id, site_config.institution)
        return site_config.site_id

    def initialize_global_model(self, layer_shapes: Dict[str, Tuple[int, ...]]) -> ModelWeights:
        """Initialize the global model with random weights.

        Args:
            layer_shapes: Dictionary mapping layer names to shapes.

        Returns:
            Initialized global model weights.
        """
        self.global_weights = ModelWeights()
        for name, shape in layer_shapes.items():
            self.global_weights.layer_weights[name] = np.random.randn(*shape) * 0.01

        for layer in self.global_weights.layer_weights:
            self.global_control.layer_weights[layer] = np.zeros_like(self.global_weights.layer_weights[layer])

        self._log_event("model_initialized", {"layers": list(layer_shapes.keys())})
        logger.info("Global model initialized with %d layers", len(layer_shapes))
        return self.global_weights

    def select_sites_for_round(self) -> List[str]:
        """Select participating sites for the current round.

        Returns:
            List of selected site IDs.
        """
        active_sites = [sid for sid, sc in self.sites.items() if sc.data_quality_score > 0.0]
        num_selected = max(self.config.min_sites_per_round, int(len(active_sites) * self.config.site_fraction))
        num_selected = min(num_selected, len(active_sites))

        if len(active_sites) <= num_selected:
            return active_sites

        quality_scores = np.array([self.sites[sid].data_quality_score for sid in active_sites])
        probs = quality_scores / quality_scores.sum()
        selected_indices = np.random.choice(len(active_sites), size=num_selected, replace=False, p=probs)
        return [active_sites[i] for i in selected_indices]

    def execute_round(self, round_number: int) -> FederationRound:
        """Execute a single federation training round.

        Args:
            round_number: Sequential round number.

        Returns:
            Completed FederationRound with aggregated results.
        """
        if self.global_weights is None:
            raise ValueError("Global model not initialized. Call initialize_global_model() first.")

        federation_round = FederationRound(
            round_id=str(uuid.uuid4())[:8],
            round_number=round_number,
            status=RoundStatus.IN_PROGRESS,
            start_time=time.time(),
        )

        selected_sites = self.select_sites_for_round()
        if len(selected_sites) < self.config.min_sites_per_round:
            federation_round.status = RoundStatus.FAILED
            federation_round.end_time = time.time()
            logger.warning("Round %d failed: insufficient sites (%d)", round_number, len(selected_sites))
            self.rounds.append(federation_round)
            return federation_round

        federation_round.participating_sites = selected_sites
        logger.info("Round %d: %d sites selected", round_number, len(selected_sites))

        for site_id in selected_sites:
            updated_weights, metrics = self.trainers[site_id].train(self.global_weights, round_number)
            federation_round.site_updates[site_id] = updated_weights
            federation_round.metrics[f"{site_id}_loss"] = metrics["loss"]
            federation_round.metrics[f"{site_id}_accuracy"] = metrics["accuracy"]

        federation_round.status = RoundStatus.AGGREGATING
        aggregated = self._aggregate(federation_round.site_updates)
        federation_round.aggregated_weights = aggregated
        self.global_weights = aggregated

        avg_loss = np.mean([v for k, v in federation_round.metrics.items() if k.endswith("_loss")])
        avg_acc = np.mean([v for k, v in federation_round.metrics.items() if k.endswith("_accuracy")])
        federation_round.metrics["avg_loss"] = float(avg_loss)
        federation_round.metrics["avg_accuracy"] = float(avg_acc)

        federation_round.status = RoundStatus.COMPLETED
        federation_round.end_time = time.time()
        self.rounds.append(federation_round)

        self._log_event(
            "round_completed",
            {
                "round": round_number,
                "sites": len(selected_sites),
                "avg_loss": float(avg_loss),
                "avg_accuracy": float(avg_acc),
            },
        )

        logger.info("Round %d completed: avg_loss=%.4f, avg_accuracy=%.4f", round_number, avg_loss, avg_acc)
        return federation_round

    def run_federation(self) -> List[FederationRound]:
        """Run the full federated learning process.

        Returns:
            List of all completed FederationRounds.
        """
        logger.info("Starting federation: %s (%d rounds)", self.config.trial_name, self.config.num_rounds)

        for r in range(self.config.num_rounds):
            federation_round = self.execute_round(r)

            if federation_round.status == RoundStatus.COMPLETED and r > 0:
                prev_loss = self.rounds[-2].metrics.get("avg_loss", float("inf"))
                curr_loss = federation_round.metrics.get("avg_loss", float("inf"))
                if abs(prev_loss - curr_loss) < self.config.convergence_threshold:
                    logger.info("Convergence reached at round %d", r)
                    break

        logger.info("Federation completed: %d rounds executed", len(self.rounds))
        return self.rounds

    def get_global_model(self) -> Optional[ModelWeights]:
        """Return the current global model weights."""
        return copy.deepcopy(self.global_weights)

    def get_federation_summary(self) -> Dict[str, Any]:
        """Generate a summary of the federation process.

        Returns:
            Dictionary with federation metrics and status.
        """
        completed_rounds = [r for r in self.rounds if r.status == RoundStatus.COMPLETED]
        return {
            "federation_id": self.config.federation_id,
            "trial_name": self.config.trial_name,
            "strategy": self.config.strategy.value,
            "total_sites": len(self.sites),
            "completed_rounds": len(completed_rounds),
            "total_rounds": self.config.num_rounds,
            "final_avg_loss": completed_rounds[-1].metrics.get("avg_loss", None) if completed_rounds else None,
            "final_avg_accuracy": completed_rounds[-1].metrics.get("avg_accuracy", None) if completed_rounds else None,
            "audit_log_entries": len(self.audit_log),
        }

    def _aggregate(self, site_updates: Dict[str, ModelWeights]) -> ModelWeights:
        """Dispatch to the configured aggregation strategy."""
        if self.config.strategy == AggregationStrategy.FEDAVG:
            return aggregate_fedavg(site_updates, self.sites)
        elif self.config.strategy == AggregationStrategy.FEDPROX:
            return aggregate_fedprox(site_updates, self.sites, self.global_weights, self.config.proximal_mu)
        elif self.config.strategy == AggregationStrategy.SCAFFOLD:
            aggregated, self.global_control = aggregate_scaffold(
                site_updates, self.sites, self.control_variates, self.global_control
            )
            return aggregated
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.config.strategy}")

    def _log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Append an event to the audit log."""
        self.audit_log.append(
            {
                "event_id": str(uuid.uuid4())[:8],
                "event_type": event_type,
                "timestamp": time.time(),
                "federation_id": self.config.federation_id,
                "details": details,
            }
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = FederationConfig(
        trial_name="ONCO-FED-001",
        strategy=AggregationStrategy.FEDAVG,
        num_rounds=5,
        min_sites_per_round=2,
    )
    coordinator = FederatedCoordinator(config)

    for i in range(3):
        coordinator.register_site(
            SiteConfig(
                site_name=f"Hospital {chr(65 + i)}",
                institution=f"Medical Center {chr(65 + i)}",
                patient_count=100 + i * 50,
            )
        )

    coordinator.initialize_global_model({"dense_1": (64, 32), "dense_2": (32, 16), "output": (16, 1)})

    rounds = coordinator.run_federation()
    summary = coordinator.get_federation_summary()
    print("\n=== Federation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
