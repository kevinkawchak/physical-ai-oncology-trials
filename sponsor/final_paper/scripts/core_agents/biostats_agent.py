"""Biostats agent: SAP generation, Bayesian adaptive methods, futility rules."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum


class AdaptiveMethod(Enum):
    BAYESIAN_RESPONSE_ADAPTIVE = "bayesian_response_adaptive"
    GROUP_SEQUENTIAL = "group_sequential"
    SAMPLE_SIZE_RE_ESTIMATION = "sample_size_re_estimation"
    BIOMARKER_ADAPTIVE = "biomarker_adaptive"


class FutilityBoundary(Enum):
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    GAMMA_FAMILY = "gamma_family"


@dataclass
class InterimAnalysis:
    analysis_number: int
    information_fraction: float
    n_enrolled: int
    n_events: int
    efficacy_boundary: float
    futility_boundary: float
    observed_statistic: float = 0.0

    @property
    def cross_efficacy(self) -> bool:
        return self.observed_statistic >= self.efficacy_boundary

    @property
    def cross_futility(self) -> bool:
        return self.observed_statistic <= self.futility_boundary


@dataclass
class BayesianPosterior:
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    successes: int = 0
    trials: int = 0

    @property
    def alpha_post(self) -> float:
        return self.alpha_prior + self.successes

    @property
    def beta_post(self) -> float:
        return self.beta_prior + self.trials - self.successes

    @property
    def mean(self) -> float:
        return self.alpha_post / (self.alpha_post + self.beta_post)

    def prob_greater_than(self, threshold: float, n_samples: int = 10000) -> float:
        count = 0
        rng = random.Random(42)
        for _ in range(n_samples):
            sample = rng.betavariate(self.alpha_post, self.beta_post)
            if sample > threshold:
                count += 1
        return count / n_samples


@dataclass
class SAPSection:
    title: str
    content: str = ""


@dataclass
class StatisticalAnalysisPlan:
    study_id: str
    primary_endpoint: str
    sample_size: int
    alpha: float = 0.025
    power: float = 0.80
    adaptive_method: AdaptiveMethod = AdaptiveMethod.GROUP_SEQUENTIAL
    interims: list[InterimAnalysis] = field(default_factory=list)
    sections: list[SAPSection] = field(default_factory=list)


class BiostatsAgent:
    """Generates SAPs, computes Bayesian posteriors, and evaluates futility."""

    def __init__(self) -> None:
        self.plans: dict[str, StatisticalAnalysisPlan] = {}

    def create_sap(self, study_id: str, endpoint: str, n: int, method: AdaptiveMethod) -> StatisticalAnalysisPlan:
        sap = StatisticalAnalysisPlan(study_id=study_id, primary_endpoint=endpoint, sample_size=n, adaptive_method=method)
        sap.sections = [
            SAPSection("Introduction"),
            SAPSection("Study Objectives and Endpoints"),
            SAPSection("Analysis Populations"),
            SAPSection("Statistical Methods"),
            SAPSection("Interim Analyses"),
            SAPSection("Sample Size Justification"),
            SAPSection("Multiplicity Adjustments"),
        ]
        self.plans[study_id] = sap
        return sap

    def obrien_fleming_boundary(self, information_fraction: float, alpha: float = 0.025) -> float:
        z_alpha = 1.96
        return z_alpha / math.sqrt(information_fraction)

    def compute_futility_boundary(self, information_fraction: float, boundary_type: FutilityBoundary) -> float:
        if boundary_type == FutilityBoundary.OBRIEN_FLEMING:
            return -self.obrien_fleming_boundary(information_fraction) * 0.5
        if boundary_type == FutilityBoundary.POCOCK:
            return 0.0
        return -0.5 * math.sqrt(information_fraction)

    def sample_size_two_arm(self, p1: float, p2: float, alpha: float = 0.025, power: float = 0.80) -> int:
        z_a = 1.96
        z_b = 0.8416
        p_bar = (p1 + p2) / 2.0
        numerator = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p1 - p2) ** 2
        return math.ceil(numerator / denominator)

    def bayesian_go_no_go(self, successes: int, trials: int, go_threshold: float = 0.2, prob_cutoff: float = 0.9) -> str:
        posterior = BayesianPosterior(successes=successes, trials=trials)
        prob = posterior.prob_greater_than(go_threshold)
        if prob >= prob_cutoff:
            return "GO"
        if prob < 0.1:
            return "NO_GO"
        return "EVALUATE"


def main() -> None:
    agent = BiostatsAgent()
    sap = agent.create_sap("STUDY-001", "ORR", 200, AdaptiveMethod.GROUP_SEQUENTIAL)
    n_per_arm = agent.sample_size_two_arm(0.35, 0.20)
    print(f"SAP: {sap.study_id}, sections={len(sap.sections)}, n_per_arm={n_per_arm}")
    decision = agent.bayesian_go_no_go(successes=18, trials=50)
    print(f"Bayesian Go/No-Go: {decision}")
    boundary = agent.obrien_fleming_boundary(0.5)
    print(f"O'Brien-Fleming boundary at 50% info: {boundary:.3f}")


if __name__ == "__main__":
    main()
