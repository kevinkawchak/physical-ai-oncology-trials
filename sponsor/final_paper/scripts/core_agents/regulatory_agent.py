"""Regulatory agent: IND lifecycle, amendments (312.30), annual reports (312.33), authority correspondence."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class SubmissionType(Enum):
    ORIGINAL_IND = "original_ind"
    AMENDMENT_312_30 = "amendment_312_30"
    ANNUAL_REPORT_312_33 = "annual_report_312_33"
    SAFETY_REPORT = "safety_report"
    INFO_AMENDMENT = "information_amendment"
    RESPONSE = "response_to_authority"


class SubmissionStatus(Enum):
    DRAFT = "draft"
    REVIEW = "internal_review"
    READY = "ready_to_submit"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    REFUSED = "refused"


class AuthorityType(Enum):
    FDA = "FDA"
    EMA = "EMA"
    PMDA = "PMDA"
    NMPA = "NMPA"
    MHRA = "MHRA"


@dataclass
class Correspondence:
    corr_id: str = field(default_factory=lambda: f"CORR-{uuid.uuid4().hex[:6]}")
    authority: AuthorityType = AuthorityType.FDA
    subject: str = ""
    sent_date: date | None = None
    response_due: date | None = None
    resolved: bool = False


@dataclass
class Submission:
    sub_id: str = field(default_factory=lambda: f"SUB-{uuid.uuid4().hex[:8]}")
    sub_type: SubmissionType = SubmissionType.ORIGINAL_IND
    authority: AuthorityType = AuthorityType.FDA
    status: SubmissionStatus = SubmissionStatus.DRAFT
    created_date: date = field(default_factory=date.today)
    submitted_date: date | None = None
    sections: list[str] = field(default_factory=list)
    serial_number: int = 0


@dataclass
class INDLifecycle:
    ind_number: str
    sponsor: str
    product_name: str
    original_submission_date: date | None = None
    submissions: list[Submission] = field(default_factory=list)
    correspondence: list[Correspondence] = field(default_factory=list)
    next_serial: int = 0

    def next_serial_number(self) -> int:
        self.next_serial += 1
        return self.next_serial


class RegulatoryAgent:
    """Manages IND lifecycle: amendments, annual reports, and authority correspondence."""

    def __init__(self) -> None:
        self.inds: dict[str, INDLifecycle] = {}

    def create_ind(self, ind_number: str, sponsor: str, product: str) -> INDLifecycle:
        lifecycle = INDLifecycle(ind_number=ind_number, sponsor=sponsor, product_name=product)
        self.inds[ind_number] = lifecycle
        return lifecycle

    def file_submission(self, ind_number: str, sub_type: SubmissionType,
                        authority: AuthorityType = AuthorityType.FDA,
                        sections: list[str] | None = None) -> Submission:
        lifecycle = self.inds[ind_number]
        sub = Submission(sub_type=sub_type, authority=authority,
                         sections=sections or [], serial_number=lifecycle.next_serial_number())
        lifecycle.submissions.append(sub)
        return sub

    def submit(self, ind_number: str, sub_id: str) -> Submission:
        lifecycle = self.inds[ind_number]
        for sub in lifecycle.submissions:
            if sub.sub_id == sub_id:
                sub.status = SubmissionStatus.SUBMITTED
                sub.submitted_date = date.today()
                return sub
        msg = f"Submission {sub_id} not found"
        raise KeyError(msg)

    def create_amendment(self, ind_number: str, sections: list[str]) -> Submission:
        return self.file_submission(ind_number, SubmissionType.AMENDMENT_312_30, sections=sections)

    def create_annual_report(self, ind_number: str) -> Submission:
        sections = ["summary_of_studies", "safety_report_summary", "protocol_amendments",
                     "investigator_brochure_changes", "manufacturing_changes", "subject_disposition"]
        return self.file_submission(ind_number, SubmissionType.ANNUAL_REPORT_312_33, sections=sections)

    def log_correspondence(self, ind_number: str, authority: AuthorityType,
                           subject: str, response_days: int = 30) -> Correspondence:
        lifecycle = self.inds[ind_number]
        corr = Correspondence(authority=authority, subject=subject, sent_date=date.today(),
                               response_due=date.today() + timedelta(days=response_days))
        lifecycle.correspondence.append(corr)
        return corr

    def pending_actions(self, ind_number: str) -> dict[str, list[str]]:
        lifecycle = self.inds[ind_number]
        pending_subs = [f"{s.sub_type.value} (serial {s.serial_number})"
                        for s in lifecycle.submissions if s.status in (SubmissionStatus.DRAFT, SubmissionStatus.REVIEW)]
        open_corr = [f"{c.subject} (due {c.response_due})" for c in lifecycle.correspondence if not c.resolved]
        return {"pending_submissions": pending_subs, "open_correspondence": open_corr}

    def lifecycle_summary(self, ind_number: str) -> dict[str, object]:
        lc = self.inds[ind_number]
        return {"ind_number": lc.ind_number, "product": lc.product_name,
                "total_submissions": len(lc.submissions),
                "submitted": sum(1 for s in lc.submissions if s.status == SubmissionStatus.SUBMITTED),
                "open_correspondence": sum(1 for c in lc.correspondence if not c.resolved)}


def main() -> None:
    agent = RegulatoryAgent()
    agent.create_ind("IND-123456", "PhysicalAI Corp", "PAI-101")
    amend = agent.create_amendment("IND-123456", ["updated_protocol", "revised_ib"])
    agent.submit("IND-123456", amend.sub_id)
    agent.create_annual_report("IND-123456")
    agent.log_correspondence("IND-123456", AuthorityType.FDA, "Information Request - CMC")
    print(f"Summary: {agent.lifecycle_summary('IND-123456')}")
    print(f"Pending: {agent.pending_actions('IND-123456')}")


if __name__ == "__main__":
    main()
