"""Financial model: cost projections, ROI analysis, NPV, Monte Carlo simulation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum


class CostCategory(Enum):
    CLINICAL_OPERATIONS = "clinical_operations"
    REGULATORY = "regulatory"
    CMC = "cmc"
    BIOSTATISTICS = "biostatistics"
    MEDICAL_WRITING = "medical_writing"
    SAFETY = "safety"
    DATA_MANAGEMENT = "data_management"
    ROBOTIC_SYSTEMS = "robotic_systems"
    SITE_COSTS = "site_costs"
    OVERHEAD = "overhead"


@dataclass
class CostLineItem:
    category: CostCategory
    description: str
    amount_usd: float
    year: int = 1
    probability: float = 1.0

    @property
    def expected_value(self) -> float:
        return self.amount_usd * self.probability


@dataclass
class RevenueProjection:
    year: int
    revenue_usd: float
    probability_of_success: float = 1.0

    @property
    def expected_revenue(self) -> float:
        return self.revenue_usd * self.probability_of_success


@dataclass
class NPVResult:
    npv: float
    irr_approx: float
    payback_year: int
    total_cost: float
    total_revenue: float


class FinancialModel:
    """Financial modeling with NPV, ROI, and Monte Carlo simulation for trial costs."""

    def __init__(self, discount_rate: float = 0.10) -> None:
        self.discount_rate = discount_rate
        self.costs: list[CostLineItem] = []
        self.revenues: list[RevenueProjection] = []

    def add_cost(self, category: CostCategory, description: str, amount: float,
                 year: int = 1, probability: float = 1.0) -> None:
        self.costs.append(CostLineItem(category, description, amount, year, probability))

    def add_revenue(self, year: int, revenue: float, pos: float = 1.0) -> None:
        self.revenues.append(RevenueProjection(year, revenue, pos))

    def total_cost_by_year(self) -> dict[int, float]:
        by_year: dict[int, float] = {}
        for c in self.costs:
            by_year[c.year] = by_year.get(c.year, 0.0) + c.expected_value
        return by_year

    def cost_breakdown(self) -> dict[str, float]:
        by_cat: dict[str, float] = {}
        for c in self.costs:
            key = c.category.value
            by_cat[key] = by_cat.get(key, 0.0) + c.expected_value
        return by_cat

    def compute_npv(self) -> NPVResult:
        max_year = max(
            max((c.year for c in self.costs), default=0),
            max((r.year for r in self.revenues), default=0),
        )
        total_cost = sum(c.expected_value for c in self.costs)
        total_rev = sum(r.expected_revenue for r in self.revenues)
        npv = 0.0
        cumulative = 0.0
        payback = max_year + 1
        for yr in range(1, max_year + 1):
            cost_yr = sum(c.expected_value for c in self.costs if c.year == yr)
            rev_yr = sum(r.expected_revenue for r in self.revenues if r.year == yr)
            net = rev_yr - cost_yr
            npv += net / ((1 + self.discount_rate) ** yr)
            cumulative += net
            if cumulative >= 0 and payback > max_year:
                payback = yr
        irr_approx = (total_rev / max(total_cost, 1)) ** (1 / max(max_year, 1)) - 1
        return NPVResult(npv=round(npv, 2), irr_approx=round(irr_approx, 4),
                         payback_year=payback, total_cost=round(total_cost, 2),
                         total_revenue=round(total_rev, 2))

    def monte_carlo_npv(self, n_simulations: int = 5000, cost_cv: float = 0.15,
                        revenue_cv: float = 0.25, seed: int = 42) -> dict[str, float]:
        rng = random.Random(seed)
        npvs: list[float] = []
        for _ in range(n_simulations):
            sim_npv = 0.0
            for c in self.costs:
                sim_cost = c.expected_value * rng.gauss(1.0, cost_cv)
                sim_npv -= sim_cost / ((1 + self.discount_rate) ** c.year)
            for r in self.revenues:
                sim_rev = r.expected_revenue * rng.gauss(1.0, revenue_cv)
                sim_npv += sim_rev / ((1 + self.discount_rate) ** r.year)
            npvs.append(sim_npv)
        npvs.sort()
        mean_npv = sum(npvs) / len(npvs)
        std_npv = math.sqrt(sum((x - mean_npv) ** 2 for x in npvs) / len(npvs))
        p5 = npvs[int(0.05 * len(npvs))]
        p95 = npvs[int(0.95 * len(npvs))]
        prob_positive = sum(1 for x in npvs if x > 0) / len(npvs)
        return {"mean_npv": round(mean_npv, 2), "std_npv": round(std_npv, 2),
                "p5": round(p5, 2), "p95": round(p95, 2),
                "prob_positive": round(prob_positive, 4)}

    def roi(self) -> float:
        total_cost = sum(c.expected_value for c in self.costs)
        total_rev = sum(r.expected_revenue for r in self.revenues)
        if total_cost == 0:
            return 0.0
        return round((total_rev - total_cost) / total_cost * 100, 2)


def main() -> None:
    model = FinancialModel(discount_rate=0.10)
    model.add_cost(CostCategory.CLINICAL_OPERATIONS, "CRO fees", 8_000_000, year=1)
    model.add_cost(CostCategory.ROBOTIC_SYSTEMS, "Robot fleet", 3_000_000, year=1)
    model.add_cost(CostCategory.SITE_COSTS, "Site payments", 5_000_000, year=2)
    model.add_cost(CostCategory.REGULATORY, "IND/NDA fees", 2_000_000, year=3)
    model.add_revenue(4, 50_000_000, pos=0.35)
    model.add_revenue(5, 150_000_000, pos=0.35)
    npv_result = model.compute_npv()
    print(f"NPV: ${npv_result.npv:,.0f}, Payback: Year {npv_result.payback_year}, ROI: {model.roi()}%")
    mc = model.monte_carlo_npv()
    print(f"MC NPV: mean=${mc['mean_npv']:,.0f}, P(positive)={mc['prob_positive']:.1%}")


if __name__ == "__main__":
    main()
