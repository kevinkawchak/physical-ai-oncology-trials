"""Safety signal detector: disproportionality analysis - PRR, ROR, MGPS, BCPNN."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ContingencyTable:
    """2x2 table: drug-event vs all-other."""

    a: int = 0  # target drug + target event
    b: int = 0  # target drug + other events
    c: int = 0  # other drugs + target event
    d: int = 0  # other drugs + other events

    @property
    def total(self) -> int:
        return self.a + self.b + self.c + self.d

    @property
    def expected(self) -> float:
        n = self.total
        if n == 0:
            return 0.0
        return (self.a + self.b) * (self.a + self.c) / n


@dataclass
class SignalScore:
    event: str
    method: str
    score: float
    lower_ci: float = 0.0
    upper_ci: float = 0.0
    is_signal: bool = False


@dataclass
class DrugEventProfile:
    drug: str
    events: dict[str, int] = field(default_factory=dict)


class SafetySignalDetector:
    """Runs disproportionality analyses: PRR, ROR, MGPS, BCPNN."""

    def __init__(self) -> None:
        self.profiles: list[DrugEventProfile] = []
        self.signals: list[SignalScore] = []

    def add_profile(self, profile: DrugEventProfile) -> None:
        self.profiles.append(profile)

    def build_table(self, target_drug: str, target_event: str) -> ContingencyTable:
        a = b = c = d = 0
        for prof in self.profiles:
            is_target_drug = prof.drug == target_drug
            for event, count in prof.events.items():
                is_target_event = event == target_event
                if is_target_drug and is_target_event:
                    a += count
                elif is_target_drug and not is_target_event:
                    b += count
                elif not is_target_drug and is_target_event:
                    c += count
                else:
                    d += count
        return ContingencyTable(a, b, c, d)

    def prr(self, table: ContingencyTable) -> SignalScore:
        """Proportional Reporting Ratio."""
        denom_num = table.a / max(table.a + table.b, 1)
        denom_den = table.c / max(table.c + table.d, 1)
        ratio = denom_num / max(denom_den, 1e-9)
        se = math.sqrt(
            1 / max(table.a, 1) - 1 / max(table.a + table.b, 1) + 1 / max(table.c, 1) - 1 / max(table.c + table.d, 1)
        )
        lower = math.exp(math.log(max(ratio, 1e-9)) - 1.96 * se)
        upper = math.exp(math.log(max(ratio, 1e-9)) + 1.96 * se)
        return SignalScore("", "PRR", round(ratio, 3), round(lower, 3), round(upper, 3), lower > 1.0 and table.a >= 3)

    def ror(self, table: ContingencyTable) -> SignalScore:
        """Reporting Odds Ratio."""
        ratio = (table.a * table.d) / max(table.b * table.c, 1)
        se = math.sqrt(1 / max(table.a, 1) + 1 / max(table.b, 1) + 1 / max(table.c, 1) + 1 / max(table.d, 1))
        lower = math.exp(math.log(max(ratio, 1e-9)) - 1.96 * se)
        upper = math.exp(math.log(max(ratio, 1e-9)) + 1.96 * se)
        return SignalScore("", "ROR", round(ratio, 3), round(lower, 3), round(upper, 3), lower > 1.0)

    def mgps(self, table: ContingencyTable) -> SignalScore:
        """Multi-item Gamma Poisson Shrinker (EBGM approximation)."""
        observed = table.a
        expected = table.expected
        if expected == 0:
            return SignalScore("", "MGPS", 0.0)
        shrunk = (observed + 0.5) / (expected + 0.5)
        return SignalScore("", "MGPS", round(shrunk, 3), is_signal=shrunk >= 2.0)

    def bcpnn(self, table: ContingencyTable) -> SignalScore:
        """Bayesian Confidence Propagation Neural Network (IC approximation)."""
        n = table.total
        if n == 0 or table.a == 0:
            return SignalScore("", "BCPNN", 0.0)
        observed = table.a / n
        exp_drug = (table.a + table.b) / n
        exp_event = (table.a + table.c) / n
        expected = exp_drug * exp_event
        ic = math.log2(observed / max(expected, 1e-12))
        return SignalScore("", "BCPNN", round(ic, 3), is_signal=ic > 0)

    def analyze_event(self, drug: str, event: str) -> list[SignalScore]:
        table = self.build_table(drug, event)
        results: list[SignalScore] = []
        for method in [self.prr, self.ror, self.mgps, self.bcpnn]:
            score = method(table)
            score.event = event
            results.append(score)
            self.signals.append(score)
        return results

    def flagged_signals(self) -> list[SignalScore]:
        return [s for s in self.signals if s.is_signal]


def main() -> None:
    detector = SafetySignalDetector()
    detector.add_profile(DrugEventProfile("PAI-101", {"neutropenia": 15, "nausea": 30, "rash": 5}))
    detector.add_profile(DrugEventProfile("Comparator", {"neutropenia": 8, "nausea": 40, "rash": 12}))
    results = detector.analyze_event("PAI-101", "neutropenia")
    for r in results:
        print(f"{r.method}: score={r.score}, signal={r.is_signal}")
    print(f"Total flagged: {len(detector.flagged_signals())}")


if __name__ == "__main__":
    main()
