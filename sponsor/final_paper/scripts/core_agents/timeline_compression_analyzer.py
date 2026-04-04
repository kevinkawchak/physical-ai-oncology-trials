"""Timeline compression analyzer: timeline comparison, daily delay cost calculations, sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TrialPhase(Enum):
    STARTUP = "startup"
    ENROLLMENT = "enrollment"
    TREATMENT = "treatment"
    FOLLOW_UP = "follow_up"
    CLOSEOUT = "closeout"
    DATABASE_LOCK = "database_lock"
    CSR_SUBMISSION = "csr_submission"


@dataclass
class PhaseTimeline:
    phase: TrialPhase
    traditional_days: int = 0
    autonomous_days: int = 0

    @property
    def compression_days(self) -> int:
        return self.traditional_days - self.autonomous_days

    @property
    def compression_pct(self) -> float:
        return (self.compression_days / self.traditional_days * 100) if self.traditional_days else 0.0


@dataclass
class DelayCostAnalysis:
    daily_revenue_at_risk: float = 0.0
    daily_operational_cost: float = 0.0
    days_saved: int = 0
    revenue_recovered: float = 0.0
    cost_avoided: float = 0.0
    total_value: float = 0.0


@dataclass
class SensitivityResult:
    parameter: str = ""
    base_value: float = 0.0
    low_value: float = 0.0
    high_value: float = 0.0
    base_savings: float = 0.0
    low_savings: float = 0.0
    high_savings: float = 0.0


class TimelineCompressionAnalyzer:
    """Compares traditional vs autonomous trial timelines with delay cost and sensitivity analysis."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.phases: list[PhaseTimeline] = []
        self._init_default_timelines()

    def _init_default_timelines(self) -> None:
        defaults = [
            (TrialPhase.STARTUP, 180, 90),
            (TrialPhase.ENROLLMENT, 365, 219),
            (TrialPhase.TREATMENT, 540, 540),
            (TrialPhase.FOLLOW_UP, 180, 150),
            (TrialPhase.CLOSEOUT, 90, 45),
            (TrialPhase.DATABASE_LOCK, 60, 14),
            (TrialPhase.CSR_SUBMISSION, 120, 45),
        ]
        for phase, trad, auto in defaults:
            self.phases.append(PhaseTimeline(phase=phase, traditional_days=trad, autonomous_days=auto))

    def total_comparison(self) -> dict[str, object]:
        trad = sum(p.traditional_days for p in self.phases)
        auto = sum(p.autonomous_days for p in self.phases)
        return {
            "traditional_days": trad,
            "autonomous_days": auto,
            "days_saved": trad - auto,
            "compression_pct": round((1 - auto / trad) * 100, 1) if trad else 0.0,
        }

    def delay_cost_analysis(
        self, daily_revenue: float = 1_500_000, daily_ops_cost: float = 85_000
    ) -> DelayCostAnalysis:
        comp = self.total_comparison()
        days = comp["days_saved"]
        rev = daily_revenue * days
        ops = daily_ops_cost * days
        return DelayCostAnalysis(
            daily_revenue_at_risk=daily_revenue,
            daily_operational_cost=daily_ops_cost,
            days_saved=days,
            revenue_recovered=rev,
            cost_avoided=ops,
            total_value=rev + ops,
        )

    def sensitivity_analysis(self, base_daily_revenue: float = 1_500_000) -> list[SensitivityResult]:
        comp = self.total_comparison()
        days = comp["days_saved"]
        parameters = [
            ("daily_revenue", base_daily_revenue, 0.75, 1.25),
            ("compression_factor", 1.0, 0.8, 1.2),
            ("discount_rate", 0.10, 0.05, 0.15),
        ]
        results: list[SensitivityResult] = []
        for name, base, low_mult, high_mult in parameters:
            base_sav = base * days
            low_sav = base * low_mult * days
            high_sav = base * high_mult * days
            results.append(
                SensitivityResult(
                    parameter=name,
                    base_value=base,
                    low_value=base * low_mult,
                    high_value=base * high_mult,
                    base_savings=round(base_sav, 2),
                    low_savings=round(low_sav, 2),
                    high_savings=round(high_sav, 2),
                )
            )
        return results

    def phase_detail(self) -> list[dict[str, object]]:
        return [
            {
                "phase": p.phase.value,
                "traditional": p.traditional_days,
                "autonomous": p.autonomous_days,
                "saved": p.compression_days,
                "compression_pct": round(p.compression_pct, 1),
            }
            for p in self.phases
        ]

    def summary(self) -> dict[str, object]:
        return {"study_id": self.study_id, "comparison": self.total_comparison(), "phases": len(self.phases)}


def main() -> None:
    analyzer = TimelineCompressionAnalyzer("STU-001")
    print(f"Comparison: {analyzer.total_comparison()}")
    delay = analyzer.delay_cost_analysis()
    print(f"Delay cost: days_saved={delay.days_saved}, value=${delay.total_value:,.0f}")
    for detail in analyzer.phase_detail():
        print(f"  {detail['phase']}: {detail['saved']} days saved ({detail['compression_pct']}%)")
    sensitivity = analyzer.sensitivity_analysis()
    for s in sensitivity:
        print(f"  Sensitivity {s.parameter}: ${s.low_savings:,.0f} - ${s.high_savings:,.0f}")
    print(f"Summary: {analyzer.summary()}")


if __name__ == "__main__":
    main()
