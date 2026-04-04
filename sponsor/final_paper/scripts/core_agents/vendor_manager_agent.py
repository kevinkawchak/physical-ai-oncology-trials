"""Vendor manager agent: contracting, SLA monitoring, performance tracking."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class VendorCategory(Enum):
    CRO = "cro"
    CENTRAL_LAB = "central_lab"
    IRT = "irt"
    EDC = "edc"
    IMAGING = "imaging"
    LOGISTICS = "logistics"
    BIOANALYTICAL = "bioanalytical"
    ROBOTICS = "robotics"


class ContractStatus(Enum):
    DRAFT = "draft"
    NEGOTIATING = "negotiating"
    EXECUTED = "executed"
    AMENDED = "amended"
    TERMINATED = "terminated"


class SLAStatus(Enum):
    MET = "met"
    AT_RISK = "at_risk"
    BREACHED = "breached"


@dataclass
class SLAMetric:
    metric_id: str
    name: str
    target: float
    actual: float = 0.0
    unit: str = ""

    @property
    def status(self) -> SLAStatus:
        if self.actual >= self.target:
            return SLAStatus.MET
        if self.actual >= self.target * 0.9:
            return SLAStatus.AT_RISK
        return SLAStatus.BREACHED


@dataclass
class VendorContract:
    contract_id: str = field(default_factory=lambda: f"VC-{uuid.uuid4().hex[:6]}")
    vendor_name: str = ""
    category: VendorCategory = VendorCategory.CRO
    status: ContractStatus = ContractStatus.DRAFT
    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=lambda: date.today() + timedelta(days=730))
    value_usd: float = 0.0
    sla_metrics: list[SLAMetric] = field(default_factory=list)


@dataclass
class PerformanceReview:
    vendor_name: str
    review_period: str
    overall_score: float = 0.0
    sla_compliance: float = 0.0
    quality_score: float = 0.0
    timeliness_score: float = 0.0
    communication_score: float = 0.0


class VendorManagerAgent:
    """Manages vendor contracts, SLA monitoring, and performance tracking."""

    def __init__(self) -> None:
        self.contracts: dict[str, VendorContract] = {}
        self.reviews: list[PerformanceReview] = []

    def create_contract(self, vendor_name: str, category: VendorCategory,
                        value_usd: float) -> VendorContract:
        contract = VendorContract(vendor_name=vendor_name, category=category, value_usd=value_usd)
        self.contracts[contract.contract_id] = contract
        return contract

    def execute_contract(self, contract_id: str) -> None:
        self.contracts[contract_id].status = ContractStatus.EXECUTED

    def add_sla_metric(self, contract_id: str, name: str, target: float, unit: str = "") -> SLAMetric:
        metric = SLAMetric(metric_id=f"SLA-{len(self.contracts[contract_id].sla_metrics)+1:02d}",
                           name=name, target=target, unit=unit)
        self.contracts[contract_id].sla_metrics.append(metric)
        return metric

    def update_sla(self, contract_id: str, metric_id: str, actual: float) -> SLAStatus:
        for metric in self.contracts[contract_id].sla_metrics:
            if metric.metric_id == metric_id:
                metric.actual = actual
                return metric.status
        return SLAStatus.MET

    def vendor_sla_dashboard(self, contract_id: str) -> list[dict[str, object]]:
        contract = self.contracts[contract_id]
        return [{"metric": m.name, "target": m.target, "actual": m.actual,
                 "status": m.status.value} for m in contract.sla_metrics]

    def conduct_review(self, vendor_name: str, period: str, quality: float,
                       timeliness: float, communication: float) -> PerformanceReview:
        sla_scores: list[float] = []
        for contract in self.contracts.values():
            if contract.vendor_name == vendor_name:
                for m in contract.sla_metrics:
                    sla_scores.append(min(m.actual / max(m.target, 1e-9), 1.0))
        sla_compliance = sum(sla_scores) / max(len(sla_scores), 1)
        overall = (quality * 0.3 + timeliness * 0.3 + communication * 0.2 + sla_compliance * 0.2)
        review = PerformanceReview(vendor_name=vendor_name, review_period=period,
                                   overall_score=round(overall, 3), sla_compliance=round(sla_compliance, 3),
                                   quality_score=quality, timeliness_score=timeliness,
                                   communication_score=communication)
        self.reviews.append(review)
        return review

    def summary(self) -> dict[str, object]:
        active = sum(1 for c in self.contracts.values() if c.status == ContractStatus.EXECUTED)
        total_value = sum(c.value_usd for c in self.contracts.values() if c.status == ContractStatus.EXECUTED)
        breached = sum(1 for c in self.contracts.values()
                       for m in c.sla_metrics if m.status == SLAStatus.BREACHED)
        return {"total_contracts": len(self.contracts), "active": active,
                "total_value_usd": total_value, "sla_breaches": breached}


def main() -> None:
    mgr = VendorManagerAgent()
    cro = mgr.create_contract("GlobalCRO Inc.", VendorCategory.CRO, 5_000_000)
    mgr.execute_contract(cro.contract_id)
    sla1 = mgr.add_sla_metric(cro.contract_id, "Query Resolution Time", 5.0, "days")
    sla2 = mgr.add_sla_metric(cro.contract_id, "Monitoring Visit Compliance", 95.0, "%")
    mgr.update_sla(cro.contract_id, sla1.metric_id, 4.2)
    mgr.update_sla(cro.contract_id, sla2.metric_id, 88.0)
    review = mgr.conduct_review("GlobalCRO Inc.", "Q1 2026", 0.85, 0.90, 0.80)
    print(f"Dashboard: {mgr.vendor_sla_dashboard(cro.contract_id)}")
    print(f"Review: score={review.overall_score}")
    print(f"Summary: {mgr.summary()}")


if __name__ == "__main__":
    main()
