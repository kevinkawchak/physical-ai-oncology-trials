"""DSUR generator: Development Safety Update Report per ICH E2F with cumulative aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class ReportingPeriod(Enum):
    ANNUAL = "annual"
    INTERIM = "interim"


class ExposureUnit(Enum):
    PATIENT_YEARS = "patient_years"
    PATIENT_MONTHS = "patient_months"
    SUBJECTS = "subjects"


@dataclass
class CumulativeSafetyData:
    total_exposed: int = 0
    total_aes: int = 0
    total_saes: int = 0
    fatal_saes: int = 0
    related_saes: int = 0
    discontinued_for_ae: int = 0
    exposure_patient_years: float = 0.0

    @property
    def sae_rate(self) -> float:
        return self.total_saes / max(self.total_exposed, 1)

    @property
    def incidence_per_100py(self) -> float:
        return (self.total_saes / max(self.exposure_patient_years, 0.01)) * 100


@dataclass
class SocSummary:
    soc_term: str
    ae_count: int = 0
    sae_count: int = 0
    subjects_affected: int = 0


@dataclass
class DSURSection:
    section_number: str
    title: str
    content: str = ""
    completed: bool = False


@dataclass
class DSURReport:
    report_id: str
    product_name: str
    dibr_date: date
    period_start: date
    period_end: date
    period_type: ReportingPeriod = ReportingPeriod.ANNUAL
    sections: list[DSURSection] = field(default_factory=list)
    cumulative: CumulativeSafetyData = field(default_factory=CumulativeSafetyData)
    soc_summaries: list[SocSummary] = field(default_factory=list)

    @property
    def completion_pct(self) -> float:
        if not self.sections:
            return 0.0
        return sum(1 for s in self.sections if s.completed) / len(self.sections)


ICH_E2F_SECTIONS: list[tuple[str, str]] = [
    ("1", "Introduction"),
    ("2", "Worldwide Marketing Approval Status"),
    ("3", "Actions Taken for Safety Reasons"),
    ("4", "Changes to Reference Safety Information"),
    ("5", "Estimated Cumulative Exposure"),
    ("6", "Data in Line Listings and Summary Tabulations"),
    ("7", "Significant Findings from Clinical Trials"),
    ("8", "Safety Findings from Non-Interventional Studies"),
    ("9", "Other Safety Information"),
    ("10", "Non-Clinical Data"),
    ("11", "Literature"),
    ("12", "Lack of Efficacy"),
    ("13", "Late-Breaking Information"),
    ("14", "Overall Safety Assessment"),
    ("15", "Summary of Important Risks"),
    ("16", "Benefit-Risk Analysis Conclusion"),
]


class DSURGenerator:
    """Generates Development Safety Update Reports per ICH E2F guidelines."""

    def __init__(self, product_name: str, dibr_date: date) -> None:
        self.product_name = product_name
        self.dibr_date = dibr_date
        self.reports: list[DSURReport] = []

    def create_report(self, report_id: str, period_type: ReportingPeriod = ReportingPeriod.ANNUAL) -> DSURReport:
        period_end = self.dibr_date
        period_start = period_end - timedelta(days=365 if period_type == ReportingPeriod.ANNUAL else 182)
        report = DSURReport(
            report_id=report_id, product_name=self.product_name, dibr_date=self.dibr_date,
            period_start=period_start, period_end=period_end, period_type=period_type,
        )
        report.sections = [DSURSection(section_number=num, title=title) for num, title in ICH_E2F_SECTIONS]
        self.reports.append(report)
        return report

    def populate_exposure(self, report_id: str, data: CumulativeSafetyData) -> None:
        report = self._find_report(report_id)
        report.cumulative = data
        for sec in report.sections:
            if sec.section_number == "5":
                sec.content = (f"Cumulative exposure: {data.total_exposed} subjects, "
                               f"{data.exposure_patient_years:.1f} patient-years")
                sec.completed = True
                break

    def add_soc_summary(self, report_id: str, soc: SocSummary) -> None:
        report = self._find_report(report_id)
        report.soc_summaries.append(soc)

    def populate_safety_summary(self, report_id: str) -> None:
        report = self._find_report(report_id)
        cum = report.cumulative
        summary = (f"Total AEs: {cum.total_aes}, SAEs: {cum.total_saes} "
                   f"(rate: {cum.sae_rate:.2%}), Fatal: {cum.fatal_saes}, "
                   f"Incidence: {cum.incidence_per_100py:.1f}/100 PY")
        for sec in report.sections:
            if sec.section_number == "14":
                sec.content = summary
                sec.completed = True
            elif sec.section_number == "6":
                lines = [f"  {s.soc_term}: {s.ae_count} AEs, {s.sae_count} SAEs" for s in report.soc_summaries]
                sec.content = "SOC Summary:\n" + "\n".join(lines) if lines else "No SOC data"
                sec.completed = True

    def finalize_report(self, report_id: str) -> dict[str, object]:
        report = self._find_report(report_id)
        incomplete = [s.title for s in report.sections if not s.completed]
        return {"report_id": report.report_id, "product": report.product_name,
                "period": f"{report.period_start} to {report.period_end}",
                "completion_pct": round(report.completion_pct * 100, 1),
                "incomplete_sections": incomplete}

    def _find_report(self, report_id: str) -> DSURReport:
        for r in self.reports:
            if r.report_id == report_id:
                return r
        msg = f"Report {report_id} not found"
        raise KeyError(msg)


def main() -> None:
    gen = DSURGenerator("PAI-101", date(2026, 3, 15))
    report = gen.create_report("DSUR-2026-001")
    gen.populate_exposure("DSUR-2026-001", CumulativeSafetyData(
        total_exposed=245, total_aes=512, total_saes=38, fatal_saes=2,
        related_saes=12, discontinued_for_ae=18, exposure_patient_years=187.5))
    gen.add_soc_summary("DSUR-2026-001", SocSummary("Blood disorders", 85, 12, 45))
    gen.add_soc_summary("DSUR-2026-001", SocSummary("GI disorders", 120, 8, 78))
    gen.populate_safety_summary("DSUR-2026-001")
    result = gen.finalize_report(report.report_id)
    print(f"DSUR: {result}")


if __name__ == "__main__":
    main()
