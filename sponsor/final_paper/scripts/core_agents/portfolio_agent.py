"""Portfolio agent: program prioritization, indication scoring, milestone tracking."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class TherapeuticArea(Enum):
    SOLID_TUMOR = "solid_tumor"
    HEMATOLOGY = "hematology"
    IMMUNO_ONCOLOGY = "immuno_oncology"
    CELL_THERAPY = "cell_therapy"
    RADIOLIGAND = "radioligand"


@dataclass
class IndicationScore:
    indication: str
    unmet_need: float  # 0-1
    competitive_intensity: float  # 0-1
    biomarker_prevalence: float  # 0-1
    regulatory_feasibility: float  # 0-1
    commercial_potential: float  # 0-1

    @property
    def composite(self) -> float:
        weights = [0.30, 0.15, 0.20, 0.15, 0.20]
        values = [
            self.unmet_need,
            1.0 - self.competitive_intensity,
            self.biomarker_prevalence,
            self.regulatory_feasibility,
            self.commercial_potential,
        ]
        return sum(w * v for w, v in zip(weights, values))


@dataclass
class Milestone:
    milestone_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    target_date: date = field(default_factory=date.today)
    completed: bool = False
    completion_date: date | None = None


@dataclass
class ProgramAsset:
    asset_id: str
    name: str
    area: TherapeuticArea
    phase: str
    indications: list[IndicationScore] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)


class PortfolioAgent:
    """Manages a portfolio of oncology assets and prioritizes investment."""

    def __init__(self) -> None:
        self.assets: dict[str, ProgramAsset] = {}

    def register_asset(self, asset: ProgramAsset) -> str:
        self.assets[asset.asset_id] = asset
        return asset.asset_id

    def rank_indications(self, asset_id: str) -> list[tuple[str, float]]:
        asset = self.assets[asset_id]
        scored = [(ind.indication, ind.composite) for ind in asset.indications]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def milestone_status(self, asset_id: str) -> dict[str, bool]:
        asset = self.assets[asset_id]
        return {m.name: m.completed for m in asset.milestones}

    def overdue_milestones(self, asset_id: str, as_of: date | None = None) -> list[Milestone]:
        ref = as_of or date.today()
        asset = self.assets[asset_id]
        return [m for m in asset.milestones if not m.completed and m.target_date < ref]

    def portfolio_heat_map(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for asset in self.assets.values():
            top = max((i.composite for i in asset.indications), default=0.0)
            overdue = len(self.overdue_milestones(asset.asset_id))
            rows.append({"asset": asset.name, "phase": asset.phase, "top_score": round(top, 3), "overdue": overdue})
        rows.sort(key=lambda r: r["top_score"], reverse=True)  # type: ignore[arg-type]
        return rows


def main() -> None:
    agent = PortfolioAgent()
    ind = IndicationScore("NSCLC", 0.9, 0.7, 0.6, 0.8, 0.85)
    ms = Milestone(name="IND Filing", target_date=date.today() + timedelta(days=90))
    asset = ProgramAsset("A001", "PAI-101", TherapeuticArea.SOLID_TUMOR, "Phase I", [ind], [ms])
    agent.register_asset(asset)
    print("Ranked indications:", agent.rank_indications("A001"))
    print("Heat map:", agent.portfolio_heat_map())


if __name__ == "__main__":
    main()
