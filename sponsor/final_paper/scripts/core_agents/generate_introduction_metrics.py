"""Generate introduction metrics: parse source files, extract regulatory dates, cost figures, staffing."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RegulatoryDate:
    regulation: str
    date_str: str
    description: str = ""


@dataclass
class CostFigure:
    category: str
    amount_usd: float
    source: str = ""
    year: int = 0


@dataclass
class StaffingMetric:
    role: str
    count: int
    source: str = ""


@dataclass
class IntroductionMetrics:
    regulatory_dates: list[RegulatoryDate] = field(default_factory=list)
    cost_figures: list[CostFigure] = field(default_factory=list)
    staffing: list[StaffingMetric] = field(default_factory=list)
    source_files_parsed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "regulatory_dates": [{"regulation": r.regulation, "date": r.date_str,
                                  "description": r.description} for r in self.regulatory_dates],
            "cost_figures": [{"category": c.category, "amount_usd": c.amount_usd,
                              "source": c.source} for c in self.cost_figures],
            "staffing": [{"role": s.role, "count": s.count, "source": s.source} for s in self.staffing],
            "source_files_parsed": self.source_files_parsed,
        }


KNOWN_REGULATIONS: list[dict[str, str]] = [
    {"regulation": "EU CTR 536/2014", "date": "2022-01-31", "description": "EU Clinical Trials Regulation effective"},
    {"regulation": "ICH E6(R3)", "date": "2023-05-19", "description": "ICH E6(R3) step 2 draft"},
    {"regulation": "ICH E8(R1)", "date": "2021-10-06", "description": "General considerations for clinical studies"},
    {"regulation": "FDA RMAT", "date": "2017-01-01", "description": "Regenerative medicine advanced therapy"},
    {"regulation": "21 CFR Part 11", "date": "1997-03-20", "description": "Electronic records and signatures"},
]

KNOWN_COSTS: list[dict[str, object]] = [
    {"category": "phase_3_oncology_trial", "amount": 150_000_000, "source": "Tufts CSDD 2022"},
    {"category": "average_daily_delay_cost", "amount": 150_000, "source": "Pharma industry benchmark"},
    {"category": "average_per_patient_cost", "amount": 47_000, "source": "CenterWatch 2023"},
    {"category": "screen_failure_cost", "amount": 5_000, "source": "Industry average"},
    {"category": "robot_platform_annual_lease", "amount": 1_500_000, "source": "Intuitive Surgical"},
]


class IntroductionMetricsGenerator:
    """Parses source files and known references to extract regulatory dates, costs, and staffing."""

    def __init__(self, source_dir: Path | None = None) -> None:
        self.source_dir = source_dir or Path(".")
        self.metrics = IntroductionMetrics()

    def load_known_references(self) -> None:
        for reg in KNOWN_REGULATIONS:
            self.metrics.regulatory_dates.append(
                RegulatoryDate(regulation=reg["regulation"], date_str=reg["date"],
                               description=reg["description"]))
        for cost in KNOWN_COSTS:
            self.metrics.cost_figures.append(
                CostFigure(category=str(cost["category"]), amount_usd=float(cost["amount"]),
                           source=str(cost["source"])))

    def parse_source_files(self, pattern: str = "*.py") -> int:
        count = 0
        if not self.source_dir.exists():
            return count
        for path in sorted(self.source_dir.glob(pattern)):
            content = path.read_text(errors="replace")
            self._extract_staffing(content, path.name)
            count += 1
        self.metrics.source_files_parsed = count
        return count

    def _extract_staffing(self, content: str, source: str) -> None:
        patterns = [
            (r"(?:staff|personnel|team)\D{0,20}(\d{1,4})", "general_staff"),
            (r"(?:investigator|PI)\D{0,20}(\d{1,3})", "investigator"),
            (r"(?:coordinator|CRC)\D{0,20}(\d{1,3})", "coordinator"),
        ]
        for pat, role in patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            for match in matches:
                val = int(match)
                if 1 <= val <= 5000:
                    self.metrics.staffing.append(StaffingMetric(role=role, count=val, source=source))

    def add_custom_cost(self, category: str, amount: float, source: str = "") -> CostFigure:
        fig = CostFigure(category=category, amount_usd=amount, source=source)
        self.metrics.cost_figures.append(fig)
        return fig

    def export_json(self, output_path: Path | None = None) -> str:
        data = self.metrics.to_dict()
        text = json.dumps(data, indent=2)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text)
        return text

    def summary(self) -> dict[str, int]:
        return {"regulatory_dates": len(self.metrics.regulatory_dates),
                "cost_figures": len(self.metrics.cost_figures),
                "staffing_entries": len(self.metrics.staffing),
                "files_parsed": self.metrics.source_files_parsed}


def main() -> None:
    gen = IntroductionMetricsGenerator()
    gen.load_known_references()
    gen.add_custom_cost("ai_platform_development", 5_000_000, "Internal estimate")
    gen.add_custom_cost("regulatory_submission_cost", 2_500_000, "FDA user fees 2026")
    print(f"Metrics loaded: {gen.summary()}")
    output = gen.export_json()
    print(output[:300])


if __name__ == "__main__":
    main()
