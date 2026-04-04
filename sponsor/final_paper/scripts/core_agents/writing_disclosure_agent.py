"""Writing and disclosure agent: Protocol, IB, CSR, DSUR, TLF automation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class DocumentType(Enum):
    PROTOCOL = "protocol"
    INVESTIGATOR_BROCHURE = "investigator_brochure"
    CSR = "clinical_study_report"
    DSUR = "dsur"
    TFL = "tables_figures_listings"
    SAP = "statistical_analysis_plan"
    ICF = "informed_consent_form"
    CSP = "clinical_summary_plan"


class DocumentStatus(Enum):
    OUTLINE = "outline"
    FIRST_DRAFT = "first_draft"
    INTERNAL_REVIEW = "internal_review"
    QC = "quality_check"
    FINAL = "final"
    APPROVED = "approved"


@dataclass
class DocumentSection:
    section_id: str
    title: str
    word_count: int = 0
    status: DocumentStatus = DocumentStatus.OUTLINE
    author: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WritingProject:
    project_id: str = field(default_factory=lambda: f"WP-{uuid.uuid4().hex[:6]}")
    document_type: DocumentType = DocumentType.PROTOCOL
    study_id: str = ""
    version: str = "1.0"
    sections: list[DocumentSection] = field(default_factory=list)
    target_date: date | None = None
    created_at: date = field(default_factory=date.today)

    @property
    def completion_pct(self) -> float:
        if not self.sections:
            return 0.0
        advanced = sum(1 for s in self.sections if s.status in (DocumentStatus.FINAL, DocumentStatus.APPROVED))
        return round(advanced / len(self.sections) * 100, 1)

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)


CSR_SECTIONS: list[tuple[str, str]] = [
    ("1", "Title Page"),
    ("2", "Synopsis"),
    ("3", "Table of Contents"),
    ("4", "List of Abbreviations"),
    ("5", "Ethics"),
    ("6", "Investigators"),
    ("7", "Introduction"),
    ("8", "Study Objectives"),
    ("9", "Study Design"),
    ("10", "Study Population"),
    ("11", "Study Treatment"),
    ("12", "Efficacy"),
    ("13", "Safety"),
    ("14", "Discussion"),
    ("15", "Conclusions"),
    ("16", "Appendices"),
]


class WritingDisclosureAgent:
    """Automates clinical document writing: Protocol, IB, CSR, DSUR, TLFs."""

    def __init__(self) -> None:
        self.projects: dict[str, WritingProject] = {}

    def create_project(self, doc_type: DocumentType, study_id: str, target_date: date | None = None) -> WritingProject:
        project = WritingProject(document_type=doc_type, study_id=study_id, target_date=target_date)
        if doc_type == DocumentType.CSR:
            project.sections = [DocumentSection(sid, title) for sid, title in CSR_SECTIONS]
        self.projects[project.project_id] = project
        return project

    def update_section(
        self, project_id: str, section_id: str, status: DocumentStatus, word_count: int = 0, author: str = ""
    ) -> None:
        project = self.projects[project_id]
        for section in project.sections:
            if section.section_id == section_id:
                section.status = status
                if word_count:
                    section.word_count = word_count
                if author:
                    section.author = author
                section.last_updated = datetime.utcnow()
                break

    def add_section(self, project_id: str, section_id: str, title: str) -> DocumentSection:
        section = DocumentSection(section_id=section_id, title=title)
        self.projects[project_id].sections.append(section)
        return section

    def generate_tfl_shell(
        self, project_id: str, n_tables: int = 20, n_figures: int = 10, n_listings: int = 15
    ) -> WritingProject:
        tfl_project = self.create_project(DocumentType.TFL, self.projects[project_id].study_id)
        for i in range(1, n_tables + 1):
            tfl_project.sections.append(DocumentSection(f"T-{i:02d}", f"Table {i}"))
        for i in range(1, n_figures + 1):
            tfl_project.sections.append(DocumentSection(f"F-{i:02d}", f"Figure {i}"))
        for i in range(1, n_listings + 1):
            tfl_project.sections.append(DocumentSection(f"L-{i:02d}", f"Listing {i}"))
        return tfl_project

    def pipeline_status(self) -> list[dict[str, object]]:
        return [
            {
                "id": p.project_id,
                "type": p.document_type.value,
                "study": p.study_id,
                "completion": p.completion_pct,
                "words": p.total_words,
            }
            for p in self.projects.values()
        ]

    def summary(self) -> dict[str, object]:
        return {
            "total_projects": len(self.projects),
            "by_type": {
                t.value: sum(1 for p in self.projects.values() if p.document_type == t)
                for t in DocumentType
                if any(p.document_type == t for p in self.projects.values())
            },
        }


def main() -> None:
    agent = WritingDisclosureAgent()
    csr = agent.create_project(DocumentType.CSR, "STU-001", date(2026, 12, 31))
    agent.update_section(csr.project_id, "2", DocumentStatus.FINAL, 1500, "MW-001")
    agent.update_section(csr.project_id, "12", DocumentStatus.FIRST_DRAFT, 5000, "MW-002")
    tfl = agent.generate_tfl_shell(csr.project_id)
    print(f"CSR: {csr.completion_pct}% complete, {csr.total_words} words")
    print(f"TFLs: {len(tfl.sections)} shells")
    print(f"Pipeline: {agent.pipeline_status()}")


if __name__ == "__main__":
    main()
