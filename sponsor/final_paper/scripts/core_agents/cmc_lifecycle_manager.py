"""CMC lifecycle manager: CMC document management, manufacturing changes, GMP compliance."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class CMCSectionType(Enum):
    DRUG_SUBSTANCE = "3.2.S"
    DRUG_PRODUCT = "3.2.P"
    APPENDICES = "3.2.A"
    REGIONAL = "3.2.R"
    NONCLINICAL = "nonclinical_support"


class ChangeCategory(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    PRIOR_APPROVAL = "prior_approval"


class GMPStatus(Enum):
    COMPLIANT = "compliant"
    OBSERVATION = "observation"
    NON_COMPLIANT = "non_compliant"
    PENDING_INSPECTION = "pending_inspection"


@dataclass
class CMCDocument:
    doc_id: str = field(default_factory=lambda: f"CMC-{uuid.uuid4().hex[:6]}")
    section: CMCSectionType = CMCSectionType.DRUG_PRODUCT
    title: str = ""
    version: str = "1.0"
    effective_date: date = field(default_factory=date.today)
    author: str = ""
    approved: bool = False


@dataclass
class ManufacturingChange:
    change_id: str = field(default_factory=lambda: f"MC-{uuid.uuid4().hex[:6]}")
    description: str = ""
    category: ChangeCategory = ChangeCategory.MINOR
    affected_documents: list[str] = field(default_factory=list)
    impact_assessment: str = ""
    regulatory_filing_needed: bool = False
    initiated_date: date = field(default_factory=date.today)
    completed: bool = False


@dataclass
class GMPAudit:
    audit_id: str = field(default_factory=lambda: f"GMP-{uuid.uuid4().hex[:6]}")
    facility: str = ""
    audit_date: date = field(default_factory=date.today)
    status: GMPStatus = GMPStatus.PENDING_INSPECTION
    observations: list[str] = field(default_factory=list)
    next_audit_due: date | None = None


class CMCLifecycleManager:
    """Manages CMC documents, manufacturing changes, and GMP compliance tracking."""

    def __init__(self, product_name: str) -> None:
        self.product_name = product_name
        self.documents: dict[str, CMCDocument] = {}
        self.changes: list[ManufacturingChange] = []
        self.audits: list[GMPAudit] = []

    def add_document(self, section: CMCSectionType, title: str, author: str) -> CMCDocument:
        doc = CMCDocument(section=section, title=title, author=author)
        self.documents[doc.doc_id] = doc
        return doc

    def approve_document(self, doc_id: str) -> CMCDocument:
        doc = self.documents[doc_id]
        doc.approved = True
        return doc

    def revise_document(self, doc_id: str, new_version: str, author: str) -> CMCDocument:
        original = self.documents[doc_id]
        revised = CMCDocument(section=original.section, title=original.title, version=new_version, author=author)
        self.documents[revised.doc_id] = revised
        return revised

    def initiate_change(
        self, description: str, category: ChangeCategory, affected_doc_ids: list[str]
    ) -> ManufacturingChange:
        filing_needed = category in (ChangeCategory.MAJOR, ChangeCategory.PRIOR_APPROVAL)
        impact = self._assess_change_impact(category, len(affected_doc_ids))
        change = ManufacturingChange(
            description=description,
            category=category,
            affected_documents=affected_doc_ids,
            impact_assessment=impact,
            regulatory_filing_needed=filing_needed,
        )
        self.changes.append(change)
        return change

    def _assess_change_impact(self, category: ChangeCategory, doc_count: int) -> str:
        risk = {"minor": "Low", "moderate": "Medium", "major": "High", "prior_approval": "Critical"}
        base = risk.get(category.value, "Unknown")
        if doc_count > 3:
            return f"{base} risk - affects {doc_count} documents, cross-functional review required"
        return f"{base} risk - affects {doc_count} document(s)"

    def record_gmp_audit(
        self, facility: str, status: GMPStatus, observations: list[str], reaudit_days: int = 730
    ) -> GMPAudit:
        audit = GMPAudit(
            facility=facility,
            status=status,
            observations=observations,
            next_audit_due=date.today() + timedelta(days=reaudit_days),
        )
        self.audits.append(audit)
        return audit

    def facilities_needing_audit(self, within_days: int = 90) -> list[GMPAudit]:
        cutoff = date.today() + timedelta(days=within_days)
        return [a for a in self.audits if a.next_audit_due and a.next_audit_due <= cutoff]

    def pending_changes(self) -> list[ManufacturingChange]:
        return [c for c in self.changes if not c.completed]

    def cmc_dashboard(self) -> dict[str, object]:
        approved = sum(1 for d in self.documents.values() if d.approved)
        return {
            "product": self.product_name,
            "total_documents": len(self.documents),
            "approved_documents": approved,
            "pending_changes": len(self.pending_changes()),
            "filing_required": sum(1 for c in self.pending_changes() if c.regulatory_filing_needed),
            "facilities_audited": len(self.audits),
            "gmp_compliant": sum(1 for a in self.audits if a.status == GMPStatus.COMPLIANT),
        }


def main() -> None:
    mgr = CMCLifecycleManager("PAI-101")
    ds = mgr.add_document(CMCSectionType.DRUG_SUBSTANCE, "Drug Substance Specification", "CMC Lead")
    dp = mgr.add_document(CMCSectionType.DRUG_PRODUCT, "Drug Product Manufacturing Process", "CMC Lead")
    mgr.approve_document(ds.doc_id)
    mgr.approve_document(dp.doc_id)
    change = mgr.initiate_change("Excipient supplier change", ChangeCategory.MODERATE, [dp.doc_id])
    mgr.record_gmp_audit("CMO-US-01", GMPStatus.COMPLIANT, [])
    mgr.record_gmp_audit("CMO-EU-01", GMPStatus.OBSERVATION, ["Minor documentation gap in batch records"])
    print(f"Change: {change.impact_assessment}, Filing: {change.regulatory_filing_needed}")
    print(f"Dashboard: {mgr.cmc_dashboard()}")


if __name__ == "__main__":
    main()
