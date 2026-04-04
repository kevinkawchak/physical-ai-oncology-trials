"""RBQM engine: QTL definition, KRI tracking, centralized statistical monitoring."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class RiskCategory(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    OPERATIONAL = "operational"


class KRIStatus(Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class ActionType(Enum):
    SITE_CONTACT = "site_contact"
    TARGETED_SDV = "targeted_sdv"
    ON_SITE_VISIT = "on_site_visit"
    CAPA = "capa"


@dataclass
class QualityToleranceLimit:
    qtl_id: str = field(default_factory=lambda: f"QTL-{uuid.uuid4().hex[:6]}")
    parameter: str = ""
    risk_category: RiskCategory = RiskCategory.IMPORTANT
    threshold_amber: float = 0.0
    threshold_red: float = 0.0
    unit: str = "%"


@dataclass
class KRIObservation:
    site_id: str
    kri_parameter: str
    value: float
    observed_date: date = field(default_factory=date.today)
    status: KRIStatus = KRIStatus.GREEN


@dataclass
class RiskAction:
    action_id: str = field(default_factory=lambda: f"RA-{uuid.uuid4().hex[:6]}")
    site_id: str = ""
    action_type: ActionType = ActionType.SITE_CONTACT
    reason: str = ""
    created_date: date = field(default_factory=date.today)
    resolved: bool = False


class RBQMEngine:
    """Risk-Based Quality Management engine with QTL, KRI tracking, and centralized monitoring."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.qtls: dict[str, QualityToleranceLimit] = {}
        self.observations: list[KRIObservation] = []
        self.actions: list[RiskAction] = []

    def define_qtl(self, parameter: str, risk_category: RiskCategory,
                   amber: float, red: float, unit: str = "%") -> QualityToleranceLimit:
        qtl = QualityToleranceLimit(parameter=parameter, risk_category=risk_category,
                                    threshold_amber=amber, threshold_red=red, unit=unit)
        self.qtls[parameter] = qtl
        return qtl

    def evaluate_kri(self, site_id: str, parameter: str, value: float) -> KRIObservation:
        qtl = self.qtls.get(parameter)
        if qtl is None:
            status = KRIStatus.GREEN
        elif value >= qtl.threshold_red:
            status = KRIStatus.RED
        elif value >= qtl.threshold_amber:
            status = KRIStatus.AMBER
        else:
            status = KRIStatus.GREEN
        obs = KRIObservation(site_id=site_id, kri_parameter=parameter, value=value, status=status)
        self.observations.append(obs)
        if status == KRIStatus.RED:
            self._auto_action(site_id, parameter, value)
        return obs

    def _auto_action(self, site_id: str, parameter: str, value: float) -> RiskAction:
        action = RiskAction(site_id=site_id, action_type=ActionType.TARGETED_SDV,
                            reason=f"KRI breach: {parameter}={value}")
        self.actions.append(action)
        return action

    def centralized_statistical_monitor(self, parameter: str) -> dict[str, float]:
        values = [o.value for o in self.observations if o.kri_parameter == parameter]
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        std = math.sqrt(variance)
        return {"mean": round(mean, 4), "std": round(std, 4), "n": float(n)}

    def site_risk_profile(self, site_id: str) -> dict[str, object]:
        site_obs = [o for o in self.observations if o.site_id == site_id]
        reds = sum(1 for o in site_obs if o.status == KRIStatus.RED)
        ambers = sum(1 for o in site_obs if o.status == KRIStatus.AMBER)
        open_actions = sum(1 for a in self.actions if a.site_id == site_id and not a.resolved)
        return {"site_id": site_id, "total_kris": len(site_obs),
                "red": reds, "amber": ambers, "open_actions": open_actions}

    def overall_quality_score(self) -> float:
        if not self.observations:
            return 1.0
        total = len(self.observations)
        reds = sum(1 for o in self.observations if o.status == KRIStatus.RED)
        ambers = sum(1 for o in self.observations if o.status == KRIStatus.AMBER)
        return round(max(1.0 - (reds * 0.1 + ambers * 0.03) / max(total, 1), 0.0), 4)

    def dashboard(self) -> dict[str, object]:
        sites = {o.site_id for o in self.observations}
        return {
            "study": self.study_id, "qtls_defined": len(self.qtls),
            "sites_monitored": len(sites), "total_observations": len(self.observations),
            "open_actions": sum(1 for a in self.actions if not a.resolved),
            "quality_score": self.overall_quality_score(),
        }


def main() -> None:
    engine = RBQMEngine("STU-001")
    engine.define_qtl("protocol_deviation_rate", RiskCategory.CRITICAL, amber=5.0, red=10.0)
    engine.define_qtl("query_rate", RiskCategory.IMPORTANT, amber=15.0, red=25.0)
    engine.define_qtl("consent_error_rate", RiskCategory.CRITICAL, amber=2.0, red=5.0)
    for site, dev_rate, qr in [("SITE-01", 3.2, 12.0), ("SITE-02", 7.5, 28.0), ("SITE-03", 11.0, 18.0)]:
        engine.evaluate_kri(site, "protocol_deviation_rate", dev_rate)
        engine.evaluate_kri(site, "query_rate", qr)
    stats = engine.centralized_statistical_monitor("protocol_deviation_rate")
    print(f"Stats: {stats}")
    print(f"Site-02 profile: {engine.site_risk_profile('SITE-02')}")
    print(f"Dashboard: {engine.dashboard()}")


if __name__ == "__main__":
    main()
