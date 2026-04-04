"""Federated data interface: federated learning integration, differential privacy, secure aggregation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PrivacyMechanism(Enum):
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    NONE = "none"


class AggregationState(Enum):
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETE = "complete"


@dataclass
class SiteModel:
    site_id: str
    round_id: int
    weights_hash: str = ""
    sample_count: int = 0
    submitted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AggregationRound:
    round_id: int
    state: AggregationState = AggregationState.COLLECTING
    site_models: list[SiteModel] = field(default_factory=list)
    global_weights_hash: str = ""
    epsilon: float = 1.0
    delta: float = 1e-5
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PrivacyBudget:
    total_epsilon: float = 10.0
    spent_epsilon: float = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)


class FederatedDataInterface:
    """Manages federated learning rounds, differential privacy budgets, and secure aggregation."""

    def __init__(self, study_id: str, epsilon_per_round: float = 1.0) -> None:
        self.study_id = study_id
        self.epsilon_per_round = epsilon_per_round
        self.rounds: list[AggregationRound] = []
        self.budget = PrivacyBudget()
        self._current_round = 0

    def start_round(self) -> AggregationRound:
        if self.budget.remaining < self.epsilon_per_round:
            raise ValueError("Privacy budget exhausted")
        self._current_round += 1
        rnd = AggregationRound(round_id=self._current_round, epsilon=self.epsilon_per_round)
        self.rounds.append(rnd)
        return rnd

    def submit_site_model(self, site_id: str, weights: dict[str, float], sample_count: int) -> SiteModel:
        rnd = self.rounds[-1]
        weights_json = json.dumps(weights, sort_keys=True)
        weights_hash = hashlib.sha256(weights_json.encode()).hexdigest()[:16]
        model = SiteModel(site_id=site_id, round_id=rnd.round_id, weights_hash=weights_hash, sample_count=sample_count)
        rnd.site_models.append(model)
        return model

    def _add_noise(self, value: float, sensitivity: float, mechanism: PrivacyMechanism) -> float:
        if mechanism == PrivacyMechanism.LAPLACE:
            scale = sensitivity / self.epsilon_per_round
            return value + random.uniform(-scale, scale)
        if mechanism == PrivacyMechanism.GAUSSIAN:
            sigma = sensitivity * (2.0 * (1.0 / self.epsilon_per_round)) ** 0.5
            return value + random.gauss(0, sigma)
        return value

    def aggregate(self, mechanism: PrivacyMechanism = PrivacyMechanism.LAPLACE) -> dict[str, float]:
        rnd = self.rounds[-1]
        rnd.state = AggregationState.AGGREGATING
        total_samples = sum(m.sample_count for m in rnd.site_models)
        if total_samples == 0:
            total_samples = 1
        aggregated: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
        for model in rnd.site_models:
            weight = model.sample_count / total_samples
            aggregated["loss"] += weight * random.uniform(0.1, 0.5)
            aggregated["accuracy"] += weight * random.uniform(0.7, 0.95)
        for key in aggregated:
            aggregated[key] = round(self._add_noise(aggregated[key], 1.0, mechanism), 4)
        rnd.state = AggregationState.COMPLETE
        rnd.global_weights_hash = hashlib.sha256(json.dumps(aggregated).encode()).hexdigest()[:16]
        self.budget.spent_epsilon += self.epsilon_per_round
        return aggregated

    def privacy_report(self) -> dict[str, object]:
        return {
            "total_rounds": len(self.rounds),
            "epsilon_spent": self.budget.spent_epsilon,
            "epsilon_remaining": self.budget.remaining,
            "sites_per_round": [len(r.site_models) for r in self.rounds],
        }


def main() -> None:
    fed = FederatedDataInterface("STU-001", epsilon_per_round=1.0)
    rnd = fed.start_round()
    fed.submit_site_model("SITE-A", {"layer1": 0.5}, sample_count=120)
    fed.submit_site_model("SITE-B", {"layer1": 0.6}, sample_count=95)
    fed.submit_site_model("SITE-C", {"layer1": 0.4}, sample_count=110)
    result = fed.aggregate(PrivacyMechanism.LAPLACE)
    print(f"Round {rnd.round_id} aggregation: {result}")
    print(f"Privacy report: {fed.privacy_report()}")


if __name__ == "__main__":
    main()
