"""IND filing engine: IND/CTA package assembly in eCTD v4.0 format."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class ECTDModule(Enum):
    M1_REGIONAL = "m1_regional"
    M2_SUMMARIES = "m2_summaries"
    M3_QUALITY = "m3_quality"
    M4_NONCLINICAL = "m4_nonclinical"
    M5_CLINICAL = "m5_clinical"


class DocumentStatus(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    SUBMITTED = "submitted"


@dataclass
class ECTDDocument:
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    title: str = ""
    module: ECTDModule = ECTDModule.M1_REGIONAL
    status: DocumentStatus = DocumentStatus.DRAFT
    version: str = "1.0"
    file_path: str = ""
    checksum: str = ""
    author: str = ""
    last_modified: datetime = field(default_factory=datetime.utcnow)

    def compute_checksum(self, content: str) -> str:
        self.checksum = hashlib.md5(content.encode()).hexdigest()
        return self.checksum


@dataclass
class ECTDSequence:
    sequence_number: int
    submission_type: str
    documents: list[ECTDDocument] = field(default_factory=list)
    created_at: date = field(default_factory=date.today)


@dataclass
class INDPackage:
    ind_number: str
    product_name: str
    sponsor: str
    sequences: list[ECTDSequence] = field(default_factory=list)
    target_region: str = "US"


REQUIRED_DOCUMENTS: dict[ECTDModule, list[str]] = {
    ECTDModule.M1_REGIONAL: ["cover_letter", "form_1571", "form_3674"],
    ECTDModule.M2_SUMMARIES: ["introduction", "quality_overall_summary", "nonclinical_overview", "clinical_overview"],
    ECTDModule.M3_QUALITY: ["drug_substance", "drug_product", "reference_standards", "container_closure"],
    ECTDModule.M4_NONCLINICAL: ["pharmacology", "pharmacokinetics", "toxicology"],
    ECTDModule.M5_CLINICAL: ["clinical_pharmacology", "clinical_efficacy", "clinical_safety", "protocol", "ib"],
}


class INDFilingEngine:
    """Assembles IND/CTA packages in eCTD v4.0 format."""

    def __init__(self) -> None:
        self.packages: dict[str, INDPackage] = {}

    def create_package(self, ind_number: str, product: str, sponsor: str, region: str = "US") -> INDPackage:
        pkg = INDPackage(ind_number=ind_number, product_name=product, sponsor=sponsor, target_region=region)
        self.packages[ind_number] = pkg
        return pkg

    def add_document(self, ind_number: str, sequence_num: int, doc: ECTDDocument) -> None:
        pkg = self.packages[ind_number]
        seq = None
        for s in pkg.sequences:
            if s.sequence_number == sequence_num:
                seq = s
                break
        if seq is None:
            seq = ECTDSequence(sequence_number=sequence_num, submission_type="original")
            pkg.sequences.append(seq)
        seq.documents.append(doc)

    def gap_analysis(self, ind_number: str) -> dict[str, list[str]]:
        pkg = self.packages[ind_number]
        all_docs = [d for s in pkg.sequences for d in s.documents]
        doc_titles = {d.title.lower() for d in all_docs}
        gaps: dict[str, list[str]] = {}
        for module, required in REQUIRED_DOCUMENTS.items():
            missing = [r for r in required if r not in doc_titles]
            if missing:
                gaps[module.value] = missing
        return gaps

    def validate_checksums(self, ind_number: str) -> list[str]:
        pkg = self.packages[ind_number]
        issues: list[str] = []
        for seq in pkg.sequences:
            for doc in seq.documents:
                if not doc.checksum:
                    issues.append(f"{doc.title}: missing checksum")
                if doc.status == DocumentStatus.DRAFT:
                    issues.append(f"{doc.title}: still in draft")
        return issues

    def submission_readiness(self, ind_number: str) -> dict[str, object]:
        gaps = self.gap_analysis(ind_number)
        issues = self.validate_checksums(ind_number)
        total_missing = sum(len(v) for v in gaps.values())
        return {
            "ind_number": ind_number,
            "ready": total_missing == 0 and len(issues) == 0,
            "missing_documents": total_missing,
            "validation_issues": len(issues),
            "gaps": gaps,
        }

    def generate_ectd_xml_stub(self, ind_number: str) -> str:
        pkg = self.packages[ind_number]
        header = '<ectd:ectd xmlns:ectd="urn:hl7-org:v3" version="4.0">'
        lines = [header, f'  <ind-number>{pkg.ind_number}</ind-number>']
        for seq in pkg.sequences:
            lines.append(f'  <sequence number="{seq.sequence_number}">')
            for doc in seq.documents:
                lines.append(f'    <document id="{doc.doc_id}" module="{doc.module.value}" title="{doc.title}"/>')
            lines.append("  </sequence>")
        lines.append("</ectd:ectd>")
        return "\n".join(lines)


def main() -> None:
    engine = INDFilingEngine()
    engine.create_package("IND-123456", "PAI-101", "PhysicalAI Oncology")
    for module, titles in REQUIRED_DOCUMENTS.items():
        for title in titles:
            doc = ECTDDocument(title=title, module=module, status=DocumentStatus.APPROVED)
            doc.compute_checksum(f"content of {title}")
            engine.add_document("IND-123456", 0, doc)
    readiness = engine.submission_readiness("IND-123456")
    print(f"Ready: {readiness['ready']}, Missing: {readiness['missing_documents']}")
    print(engine.generate_ectd_xml_stub("IND-123456")[:200])


if __name__ == "__main__":
    main()
