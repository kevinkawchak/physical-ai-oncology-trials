"""ClinOps agent: site startup, monitoring plan, enrollment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class SiteStatus(Enum):
    IDENTIFIED = "identified"
    SELECTED = "selected"
    IRB_SUBMITTED = "irb_submitted"
    IRB_APPROVED = "irb_approved"
    CONTRACT_EXECUTED = "contract_executed"
    ACTIVATED = "activated"
    ENROLLING = "enrolling"
    CLOSED = "closed"


class MonitoringType(Enum):
    ON_SITE = "on_site"
    REMOTE = "remote"
    CENTRALIZED = "centralized"
    RISK_BASED = "risk_based"


@dataclass
class SiteRecord:
    site_id: str
    name: str
    pi_name: str
    country: str
    status: SiteStatus = SiteStatus.IDENTIFIED
    irb_submission_date: date | None = None
    activation_date: date | None = None
    enrolled: int = 0
    target_enrollment: int = 0
    screen_fail_rate: float = 0.0


@dataclass
class MonitoringVisit:
    visit_id: str
    site_id: str
    visit_type: MonitoringType
    scheduled_date: date
    completed: bool = False
    findings: list[str] = field(default_factory=list)


@dataclass
class MonitoringPlan:
    plan_id: str
    study_id: str
    risk_tier_high_frequency_weeks: int = 4
    risk_tier_medium_frequency_weeks: int = 8
    risk_tier_low_frequency_weeks: int = 12
    visits: list[MonitoringVisit] = field(default_factory=list)


class ClinOpsAgent:
    """Manages site startup, monitoring visits, and enrollment tracking."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.sites: dict[str, SiteRecord] = {}
        self.monitoring_plan: MonitoringPlan | None = None

    def add_site(self, site: SiteRecord) -> str:
        self.sites[site.site_id] = site
        return site.site_id

    def advance_site(self, site_id: str, new_status: SiteStatus) -> None:
        site = self.sites[site_id]
        site.status = new_status
        if new_status == SiteStatus.ACTIVATED:
            site.activation_date = date.today()

    def startup_metrics(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for site in self.sites.values():
            key = site.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def enrollment_summary(self) -> dict[str, object]:
        total_enrolled = sum(s.enrolled for s in self.sites.values())
        total_target = sum(s.target_enrollment for s in self.sites.values())
        active = [s for s in self.sites.values() if s.status == SiteStatus.ENROLLING]
        avg_sfr = sum(s.screen_fail_rate for s in active) / max(len(active), 1)
        return {
            "total_enrolled": total_enrolled,
            "total_target": total_target,
            "pct_enrolled": round(total_enrolled / max(total_target, 1) * 100, 1),
            "active_sites": len(active),
            "avg_screen_fail_rate": round(avg_sfr, 3),
        }

    def create_monitoring_plan(self) -> MonitoringPlan:
        plan = MonitoringPlan(plan_id=f"MP-{self.study_id}", study_id=self.study_id)
        visit_num = 0
        for site in self.sites.values():
            if site.status in (SiteStatus.ACTIVATED, SiteStatus.ENROLLING):
                freq_weeks = plan.risk_tier_medium_frequency_weeks
                base = site.activation_date or date.today()
                for i in range(4):
                    visit_num += 1
                    plan.visits.append(
                        MonitoringVisit(
                            visit_id=f"MV-{visit_num:04d}",
                            site_id=site.site_id,
                            visit_type=MonitoringType.RISK_BASED,
                            scheduled_date=base + timedelta(weeks=freq_weeks * (i + 1)),
                        )
                    )
        self.monitoring_plan = plan
        return plan

    def site_risk_score(self, site_id: str) -> float:
        site = self.sites[site_id]
        score = 0.0
        if site.screen_fail_rate > 0.40:
            score += 0.3
        if site.enrolled < site.target_enrollment * 0.25:
            score += 0.3
        if site.status == SiteStatus.ACTIVATED and not site.enrolled:
            score += 0.2
        return min(score, 1.0)


def main() -> None:
    agent = ClinOpsAgent("STU-001")
    site = SiteRecord("S01", "Memorial Cancer Center", "Dr. Lee", "US", target_enrollment=30)
    agent.add_site(site)
    agent.advance_site("S01", SiteStatus.ACTIVATED)
    agent.sites["S01"].enrolled = 8
    agent.sites["S01"].status = SiteStatus.ENROLLING
    agent.sites["S01"].screen_fail_rate = 0.25
    plan = agent.create_monitoring_plan()
    print(f"Sites: {agent.startup_metrics()}")
    print(f"Enrollment: {agent.enrollment_summary()}")
    print(f"Monitoring visits: {len(plan.visits)}, Risk: {agent.site_risk_score('S01'):.2f}")


if __name__ == "__main__":
    main()
