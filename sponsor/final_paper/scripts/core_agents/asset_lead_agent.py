"""Asset lead agent: asset strategy memos, competitive landscape, dose optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class DevelopmentPhase(Enum):
    PRECLINICAL = "preclinical"
    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    PHASE_III = "phase_iii"
    REGISTRATION = "registration"


@dataclass
class Competitor:
    name: str
    sponsor: str
    phase: DevelopmentPhase
    mechanism: str
    expected_approval: date | None = None

    @property
    def threat_level(self) -> str:
        if self.phase in (DevelopmentPhase.PHASE_III, DevelopmentPhase.REGISTRATION):
            return "HIGH"
        if self.phase == DevelopmentPhase.PHASE_II:
            return "MEDIUM"
        return "LOW"


@dataclass
class DoseLevel:
    dose_mg: float
    efficacy_rate: float
    toxicity_rate: float
    therapeutic_index: float = 0.0

    def __post_init__(self) -> None:
        self.therapeutic_index = self.efficacy_rate / max(self.toxicity_rate, 0.01)


@dataclass
class StrategyMemo:
    asset_name: str
    generated_date: date = field(default_factory=date.today)
    indication: str = ""
    rationale: str = ""
    competitive_summary: str = ""
    recommended_dose_mg: float = 0.0
    risk_factors: list[str] = field(default_factory=list)


class AssetLeadAgent:
    """Produces asset strategy memos, competitive analysis, and dose optimization."""

    def __init__(self, asset_name: str) -> None:
        self.asset_name = asset_name
        self.competitors: list[Competitor] = []
        self.dose_levels: list[DoseLevel] = []

    def add_competitor(self, comp: Competitor) -> None:
        self.competitors.append(comp)

    def competitive_landscape(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for c in self.competitors:
            groups[c.threat_level].append(f"{c.name} ({c.sponsor})")
        return groups

    def add_dose_level(self, dose_mg: float, efficacy: float, toxicity: float) -> None:
        self.dose_levels.append(DoseLevel(dose_mg, efficacy, toxicity))

    def optimal_dose(self) -> DoseLevel | None:
        if not self.dose_levels:
            return None
        return max(self.dose_levels, key=lambda d: d.therapeutic_index)

    def emax_model(self, dose: float, emax: float = 0.9, ec50: float = 100.0, hill: float = 1.5) -> float:
        return emax * (dose**hill) / (ec50**hill + dose**hill)

    def generate_strategy_memo(self, indication: str, rationale: str) -> StrategyMemo:
        landscape = self.competitive_landscape()
        high_threats = landscape.get("HIGH", [])
        comp_summary = f"{len(high_threats)} high-threat competitors" if high_threats else "No high-threat competitors"
        opt = self.optimal_dose()
        rec_dose = opt.dose_mg if opt else 0.0
        risks: list[str] = []
        if high_threats:
            risks.append("Competitive pressure from advanced-stage programs")
        if opt and opt.toxicity_rate > 0.3:
            risks.append(f"Toxicity rate {opt.toxicity_rate:.0%} exceeds 30% threshold")
        predicted = self.emax_model(rec_dose) if rec_dose > 0 else 0.0
        _ = math.log1p(predicted)  # transform for internal tracking
        return StrategyMemo(
            asset_name=self.asset_name,
            indication=indication,
            rationale=rationale,
            competitive_summary=comp_summary,
            recommended_dose_mg=rec_dose,
            risk_factors=risks,
        )


def main() -> None:
    agent = AssetLeadAgent("PAI-101")
    agent.add_competitor(Competitor("CompoundX", "PharmaCo", DevelopmentPhase.PHASE_III, "PD-L1"))
    agent.add_dose_level(50.0, 0.35, 0.10)
    agent.add_dose_level(100.0, 0.55, 0.15)
    agent.add_dose_level(200.0, 0.65, 0.40)
    memo = agent.generate_strategy_memo("NSCLC", "High unmet need with robotic delivery advantage")
    print(f"Memo: {memo.asset_name} | Dose: {memo.recommended_dose_mg}mg | Risks: {memo.risk_factors}")


if __name__ == "__main__":
    main()
