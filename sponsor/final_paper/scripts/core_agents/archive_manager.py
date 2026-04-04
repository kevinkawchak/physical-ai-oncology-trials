"""Archive manager: 25-year EU CTR archive, format migration, retention policies."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class ArchiveFormat(Enum):
    PDF_A = "pdf_a"
    XML = "xml"
    DICOM = "dicom"
    SAS_XPT = "sas_xpt"
    CSV = "csv"
    JSON = "json"


class RetentionPolicy(Enum):
    EU_CTR_25 = "eu_ctr_25_years"
    FDA_15 = "fda_15_years"
    ICH_15 = "ich_15_years"
    PERMANENT = "permanent"


class MigrationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class ArchivedDocument:
    doc_id: str
    title: str
    original_format: ArchiveFormat
    archive_format: ArchiveFormat = ArchiveFormat.PDF_A
    checksum: str = ""
    size_bytes: int = 0
    archived_date: date = field(default_factory=date.today)
    retention_policy: RetentionPolicy = RetentionPolicy.EU_CTR_25
    destruction_date: date | None = None

    def compute_destruction_date(self) -> date:
        years_map = {
            RetentionPolicy.EU_CTR_25: 25,
            RetentionPolicy.FDA_15: 15,
            RetentionPolicy.ICH_15: 15,
            RetentionPolicy.PERMANENT: 100,
        }
        years = years_map.get(self.retention_policy, 25)
        self.destruction_date = self.archived_date + timedelta(days=years * 365)
        return self.destruction_date


@dataclass
class MigrationTask:
    task_id: str
    doc_id: str
    source_format: ArchiveFormat
    target_format: ArchiveFormat
    status: MigrationStatus = MigrationStatus.PENDING
    source_checksum: str = ""
    target_checksum: str = ""

    @property
    def integrity_verified(self) -> bool:
        return self.source_checksum != "" and self.source_checksum == self.target_checksum


class ArchiveManager:
    """Manages 25-year EU CTR archives, format migration, and retention policies."""

    def __init__(self, study_id: str) -> None:
        self.study_id = study_id
        self.documents: dict[str, ArchivedDocument] = {}
        self.migrations: list[MigrationTask] = []

    def archive_document(self, doc_id: str, title: str, fmt: ArchiveFormat,
                         size_bytes: int, content_hash: str = "",
                         policy: RetentionPolicy = RetentionPolicy.EU_CTR_25) -> ArchivedDocument:
        if not content_hash:
            content_hash = hashlib.sha256(f"{doc_id}|{title}".encode()).hexdigest()[:16]
        doc = ArchivedDocument(doc_id=doc_id, title=title, original_format=fmt,
                               archive_format=fmt, checksum=content_hash, size_bytes=size_bytes,
                               retention_policy=policy)
        doc.compute_destruction_date()
        self.documents[doc_id] = doc
        return doc

    def schedule_migration(self, doc_id: str, target_format: ArchiveFormat) -> MigrationTask:
        doc = self.documents[doc_id]
        task = MigrationTask(task_id=f"MIG-{len(self.migrations)+1:04d}", doc_id=doc_id,
                             source_format=doc.archive_format, target_format=target_format,
                             source_checksum=doc.checksum)
        self.migrations.append(task)
        return task

    def complete_migration(self, task_id: str, target_checksum: str) -> bool:
        for task in self.migrations:
            if task.task_id == task_id:
                task.target_checksum = target_checksum
                task.status = MigrationStatus.VERIFIED if task.integrity_verified else MigrationStatus.FAILED
                if task.status == MigrationStatus.VERIFIED:
                    doc = self.documents[task.doc_id]
                    doc.archive_format = task.target_format
                    doc.checksum = target_checksum
                return task.status == MigrationStatus.VERIFIED
        return False

    def documents_due_destruction(self, within_days: int = 365) -> list[ArchivedDocument]:
        cutoff = date.today() + timedelta(days=within_days)
        return [d for d in self.documents.values()
                if d.destruction_date and d.destruction_date <= cutoff]

    def storage_summary(self) -> dict[str, object]:
        total_bytes = sum(d.size_bytes for d in self.documents.values())
        formats: dict[str, int] = {}
        for d in self.documents.values():
            key = d.archive_format.value
            formats[key] = formats.get(key, 0) + 1
        return {"study": self.study_id, "total_documents": len(self.documents),
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "formats": formats, "pending_migrations": sum(
                    1 for m in self.migrations if m.status == MigrationStatus.PENDING)}


def main() -> None:
    mgr = ArchiveManager("STU-001")
    mgr.archive_document("DOC-001", "Final Protocol", ArchiveFormat.PDF_A, 2_500_000)
    mgr.archive_document("DOC-002", "SDTM Datasets", ArchiveFormat.SAS_XPT, 15_000_000)
    mgr.archive_document("DOC-003", "Imaging Data", ArchiveFormat.DICOM, 500_000_000)
    task = mgr.schedule_migration("DOC-002", ArchiveFormat.CSV)
    mgr.complete_migration(task.task_id, mgr.documents["DOC-002"].checksum)
    print(f"Storage: {mgr.storage_summary()}")
    print(f"Due for destruction: {len(mgr.documents_due_destruction())}")


if __name__ == "__main__":
    main()
