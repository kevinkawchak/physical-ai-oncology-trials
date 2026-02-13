"""
=============================================================================
21 CFR Part 11 Audit Trail Generator for Physical AI Oncology Trials
=============================================================================

Generates 21 CFR Part 11-compliant audit trail records for AI model
training runs, validation experiments, and configuration changes in
physical AI oncology systems.

CLINICAL CONTEXT:
-----------------
21 CFR Part 11 establishes FDA requirements for electronic records and
electronic signatures, ensuring data integrity, traceability, and
non-repudiation for regulated activities. For AI/ML-enabled oncology
devices, audit trail requirements extend to:
  - Model training and retraining events
  - Validation and performance evaluation runs
  - Configuration and hyperparameter changes
  - Data pipeline modifications
  - Deployment and rollback events
  - User access and administrative actions

Key 21 CFR Part 11 audit trail requirements:
  - Computer-generated, time-stamped audit trails
  - Record of operator actions that create, modify, or delete electronic records
  - Records maintained for the required retention period
  - Available for FDA review and copying
  - Secure, immutable record chain

USE CASES COVERED:
    1. AI model training run audit records
    2. Validation experiment audit trails
    3. Configuration change documentation
    4. Deployment event tracking
    5. Comprehensive audit report generation

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - 21 CFR Part 11 — Electronic Records; Electronic Signatures
    - FDA Guidance: Part 11, Scope and Application (2003, updated 2024)
    - FDA Q&A on Part 11 (Oct 2024)
    - GAMP 5 Guide: Compliant GxP Computerized Systems (2022)
    - ICH E6(R3) — Data Governance provisions

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Audit trail implementation requires IT validation and compliance review
    per institutional 21 CFR Part 11 policies.

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================


class AuditEventType(Enum):
    """Types of auditable events in AI/ML workflows."""

    MODEL_TRAINING = "model_training"
    MODEL_RETRAINING = "model_retraining"
    VALIDATION_RUN = "validation_run"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    CONFIG_CHANGE = "configuration_change"
    HYPERPARAMETER_CHANGE = "hyperparameter_change"
    DATA_PIPELINE_CHANGE = "data_pipeline_change"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    THRESHOLD_CHANGE = "threshold_change"
    DATA_INGESTION = "data_ingestion"
    USER_ACCESS = "user_access"
    ADMINISTRATIVE = "administrative"


class AuditSeverity(Enum):
    """Severity classification for audit events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """Individual 21 CFR Part 11 compliant audit record."""

    record_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: str
    operator: str
    action: str
    description: str
    system_id: str = ""
    previous_value: str = ""
    new_value: str = ""
    reason_for_change: str = ""
    electronic_signature: str = ""
    record_hash: str = ""
    parent_record_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRunRecord:
    """Specialized audit record for AI model training runs."""

    run_id: str
    model_name: str
    model_version: str
    dataset_id: str
    dataset_size: int = 0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    training_duration_seconds: float = 0.0
    hardware_description: str = ""
    framework_version: str = ""
    random_seed: int = 42
    convergence_status: str = "completed"
    notes: str = ""


@dataclass
class ValidationRecord:
    """Specialized audit record for validation experiments."""

    validation_id: str
    model_name: str
    model_version: str
    test_dataset_id: str
    test_dataset_size: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    acceptance_criteria: dict[str, str] = field(default_factory=dict)
    pass_fail: str = "pass"
    reviewer: str = ""
    review_date: str = ""
    notes: str = ""


@dataclass
class AuditTrailReport:
    """Complete audit trail report."""

    report_id: str
    system_name: str
    reporting_period_start: str
    reporting_period_end: str
    records: list[AuditRecord] = field(default_factory=list)
    training_records: list[TrainingRunRecord] = field(default_factory=list)
    validation_records: list[ValidationRecord] = field(default_factory=list)
    generated_date: str = ""
    generated_markdown: str = ""


# =============================================================================
# SECTION 2: HASH AND SIGNATURE UTILITIES
# =============================================================================


def compute_record_hash(record: AuditRecord) -> str:
    """
    Compute SHA-256 hash for an audit record to ensure integrity.

    Args:
        record: Audit record to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    content = (
        f"{record.record_id}|{record.event_type.value}|{record.timestamp}|"
        f"{record.operator}|{record.action}|{record.description}|"
        f"{record.previous_value}|{record.new_value}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_chain_hash(records: list[AuditRecord]) -> str:
    """
    Compute chain hash linking all records for tamper detection.

    Args:
        records: Ordered list of audit records.

    Returns:
        Hex-encoded SHA-256 chain hash.
    """
    chain = ""
    for record in records:
        chain += record.record_hash
    return hashlib.sha256(chain.encode("utf-8")).hexdigest()


# =============================================================================
# SECTION 3: AUDIT TRAIL GENERATOR
# =============================================================================


class AuditTrailGenerator:
    """
    Generates 21 CFR Part 11-compliant audit trail records and reports
    for AI/ML model lifecycle events in oncology device development.

    Produces structured Markdown audit reports with tamper-evident
    record chains, training run documentation, and validation records.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Record individual audit events with hash chain integrity
    2. Document AI model training runs with full provenance
    3. Document validation experiments with acceptance criteria
    4. Generate comprehensive audit trail reports
    """

    def __init__(self, system_name: str, system_id: str = ""):
        """
        Initialize audit trail generator.

        Args:
            system_name: Name of the system being audited.
            system_id: Unique system identifier.
        """
        self.system_name = system_name
        self.system_id = system_id or f"SYS-{system_name.upper().replace(' ', '-')[:20]}"
        self._record_counter = 0
        self._records: list[AuditRecord] = []
        self._training_records: list[TrainingRunRecord] = []
        self._validation_records: list[ValidationRecord] = []

        logger.info(f"AuditTrailGenerator initialized: {system_name} ({self.system_id})")

    def record_event(
        self,
        event_type: AuditEventType,
        operator: str,
        action: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        previous_value: str = "",
        new_value: str = "",
        reason_for_change: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """
        Record an auditable event.

        Args:
            event_type: Type of audit event.
            operator: Person or system performing the action.
            action: Brief action description.
            description: Detailed description of the event.
            severity: Event severity level.
            previous_value: Previous value (for changes).
            new_value: New value (for changes).
            reason_for_change: Justification for the change.
            metadata: Additional metadata.

        Returns:
            AuditRecord with computed hash.
        """
        self._record_counter += 1
        record_id = f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._record_counter:06d}"

        # Chain to previous record
        parent_id = self._records[-1].record_id if self._records else ""

        record = AuditRecord(
            record_id=record_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            operator=operator,
            action=action,
            description=description,
            system_id=self.system_id,
            previous_value=previous_value,
            new_value=new_value,
            reason_for_change=reason_for_change,
            parent_record_id=parent_id,
            metadata=metadata or {},
        )

        # Compute integrity hash
        record.record_hash = compute_record_hash(record)

        self._records.append(record)

        logger.info(f"Audit record {record_id}: [{event_type.value}] {action} by {operator}")

        return record

    def record_training_run(
        self,
        operator: str,
        model_name: str,
        model_version: str,
        dataset_id: str,
        dataset_size: int = 0,
        hyperparameters: dict[str, Any] | None = None,
        final_metrics: dict[str, float] | None = None,
        training_duration_seconds: float = 0.0,
        hardware_description: str = "",
        framework_version: str = "",
        random_seed: int = 42,
        notes: str = "",
    ) -> TrainingRunRecord:
        """
        Record an AI model training run.

        Args:
            operator: Person or system initiating training.
            model_name: Name of the model.
            model_version: Version of the model.
            dataset_id: Identifier for the training dataset.
            dataset_size: Number of samples in training set.
            hyperparameters: Training hyperparameters.
            final_metrics: Final training metrics.
            training_duration_seconds: Training duration.
            hardware_description: Hardware used for training.
            framework_version: ML framework version.
            random_seed: Random seed for reproducibility.
            notes: Additional notes.

        Returns:
            TrainingRunRecord.
        """
        run_id = f"TRN-{datetime.now().strftime('%Y%m%d')}-{len(self._training_records) + 1:04d}"

        training_record = TrainingRunRecord(
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            dataset_id=dataset_id,
            dataset_size=dataset_size,
            hyperparameters=hyperparameters or {},
            final_metrics=final_metrics or {},
            training_duration_seconds=training_duration_seconds,
            hardware_description=hardware_description,
            framework_version=framework_version,
            random_seed=random_seed,
            notes=notes,
        )

        self._training_records.append(training_record)

        # Also create an audit record for the training event
        metrics_summary = ", ".join(f"{k}={v:.4f}" for k, v in (final_metrics or {}).items())
        self.record_event(
            event_type=AuditEventType.MODEL_TRAINING,
            operator=operator,
            action=f"Training run {run_id}",
            description=f"Model {model_name} v{model_version} trained on {dataset_id} (N={dataset_size}). Metrics: {metrics_summary}",
            metadata={"run_id": run_id, "model_name": model_name, "model_version": model_version},
        )

        logger.info(f"Training run recorded: {run_id} ({model_name} v{model_version})")
        return training_record

    def record_validation(
        self,
        operator: str,
        model_name: str,
        model_version: str,
        test_dataset_id: str,
        test_dataset_size: int = 0,
        metrics: dict[str, float] | None = None,
        acceptance_criteria: dict[str, str] | None = None,
        pass_fail: str = "pass",
        reviewer: str = "",
        notes: str = "",
    ) -> ValidationRecord:
        """
        Record a validation experiment.

        Args:
            operator: Person or system running validation.
            model_name: Name of the model.
            model_version: Version of the model.
            test_dataset_id: Identifier for the test dataset.
            test_dataset_size: Number of samples in test set.
            metrics: Validation metrics.
            acceptance_criteria: Acceptance criteria definitions.
            pass_fail: Overall pass/fail status.
            reviewer: Person reviewing validation results.
            notes: Additional notes.

        Returns:
            ValidationRecord.
        """
        val_id = f"VAL-{datetime.now().strftime('%Y%m%d')}-{len(self._validation_records) + 1:04d}"

        validation_record = ValidationRecord(
            validation_id=val_id,
            model_name=model_name,
            model_version=model_version,
            test_dataset_id=test_dataset_id,
            test_dataset_size=test_dataset_size,
            metrics=metrics or {},
            acceptance_criteria=acceptance_criteria or {},
            pass_fail=pass_fail,
            reviewer=reviewer,
            review_date=datetime.now().strftime("%Y-%m-%d") if reviewer else "",
            notes=notes,
        )

        self._validation_records.append(validation_record)

        # Also create an audit record
        severity = AuditSeverity.INFO if pass_fail == "pass" else AuditSeverity.WARNING
        metrics_summary = ", ".join(f"{k}={v:.4f}" for k, v in (metrics or {}).items())
        self.record_event(
            event_type=AuditEventType.VALIDATION_RUN,
            operator=operator,
            action=f"Validation {val_id} — {pass_fail.upper()}",
            description=f"Model {model_name} v{model_version} validated on {test_dataset_id}. Result: {pass_fail}. Metrics: {metrics_summary}",
            severity=severity,
            metadata={"validation_id": val_id, "pass_fail": pass_fail},
        )

        logger.info(f"Validation recorded: {val_id} ({model_name} v{model_version}) — {pass_fail}")
        return validation_record

    def generate_report(
        self,
        reporting_period_start: str = "",
        reporting_period_end: str = "",
    ) -> AuditTrailReport:
        """
        Generate a comprehensive audit trail report.

        Args:
            reporting_period_start: Start of reporting period (ISO date).
            reporting_period_end: End of reporting period (ISO date).

        Returns:
            AuditTrailReport with all records and generated Markdown.
        """
        report_id = f"ATR-{datetime.now().strftime('%Y%m%d')}-001"

        if not reporting_period_start:
            if self._records:
                reporting_period_start = self._records[0].timestamp[:10]
            else:
                reporting_period_start = datetime.now().strftime("%Y-%m-%d")
        if not reporting_period_end:
            reporting_period_end = datetime.now().strftime("%Y-%m-%d")

        report = AuditTrailReport(
            report_id=report_id,
            system_name=self.system_name,
            reporting_period_start=reporting_period_start,
            reporting_period_end=reporting_period_end,
            records=list(self._records),
            training_records=list(self._training_records),
            validation_records=list(self._validation_records),
            generated_date=datetime.now().isoformat(),
        )

        report.generated_markdown = self._render_markdown(report)

        logger.info(
            f"Audit trail report generated: {report_id}, "
            f"{len(self._records)} records, "
            f"{len(self._training_records)} training runs, "
            f"{len(self._validation_records)} validations"
        )

        return report

    def _render_markdown(self, report: AuditTrailReport) -> str:
        """Render audit trail report as Markdown."""
        sections: list[str] = []

        sections.append("# 21 CFR Part 11 Audit Trail Report\n")
        sections.append(f"**Report ID:** {report.report_id}  ")
        sections.append(f"**System:** {report.system_name}  ")
        sections.append(f"**Reporting Period:** {report.reporting_period_start} to {report.reporting_period_end}  ")
        sections.append(f"**Generated:** {report.generated_date[:10]}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** This audit trail is generated for planning and ")
        sections.append("> demonstration purposes. Production audit trails require validated systems ")
        sections.append("> per institutional 21 CFR Part 11 policies.\n")

        # Summary
        sections.append("## 1. Audit Summary\n")
        sections.append("| Metric | Count |")
        sections.append("|--------|-------|")
        sections.append(f"| Total Audit Records | {len(report.records)} |")
        sections.append(f"| Training Runs | {len(report.training_records)} |")
        sections.append(f"| Validation Runs | {len(report.validation_records)} |")
        critical = sum(1 for r in report.records if r.severity == AuditSeverity.CRITICAL)
        warnings = sum(1 for r in report.records if r.severity == AuditSeverity.WARNING)
        sections.append(f"| Critical Events | {critical} |")
        sections.append(f"| Warning Events | {warnings} |")

        # Chain integrity
        chain_hash = compute_chain_hash(report.records) if report.records else "N/A"
        sections.append(f"| Chain Hash | `{chain_hash[:16]}...` |")
        sections.append("")

        # Audit Event Timeline
        if report.records:
            sections.append("## 2. Audit Event Timeline\n")
            sections.append("| Record ID | Timestamp | Type | Operator | Action | Severity |")
            sections.append("|-----------|-----------|------|----------|--------|----------|")
            for r in report.records:
                ts = r.timestamp[:19]
                action_preview = r.action[:30] + "..." if len(r.action) > 30 else r.action
                sections.append(
                    f"| {r.record_id} | {ts} | {r.event_type.value} | "
                    f"{r.operator} | {action_preview} | {r.severity.value} |"
                )
            sections.append("")

        # Training Runs
        if report.training_records:
            sections.append("## 3. Model Training Runs\n")
            for tr in report.training_records:
                sections.append(f"### {tr.run_id}: {tr.model_name} v{tr.model_version}\n")
                sections.append("| Attribute | Value |")
                sections.append("|-----------|-------|")
                sections.append(f"| Dataset | {tr.dataset_id} (N={tr.dataset_size:,}) |")
                if tr.training_duration_seconds > 0:
                    sections.append(f"| Duration | {tr.training_duration_seconds:.1f}s |")
                if tr.hardware_description:
                    sections.append(f"| Hardware | {tr.hardware_description} |")
                if tr.framework_version:
                    sections.append(f"| Framework | {tr.framework_version} |")
                sections.append(f"| Random Seed | {tr.random_seed} |")
                sections.append(f"| Status | {tr.convergence_status} |")
                sections.append("")

                if tr.hyperparameters:
                    sections.append("**Hyperparameters:**\n")
                    sections.append("| Parameter | Value |")
                    sections.append("|-----------|-------|")
                    for k, v in tr.hyperparameters.items():
                        sections.append(f"| {k} | {v} |")
                    sections.append("")

                if tr.final_metrics:
                    sections.append("**Final Metrics:**\n")
                    sections.append("| Metric | Value |")
                    sections.append("|--------|-------|")
                    for k, v in tr.final_metrics.items():
                        sections.append(f"| {k} | {v:.4f} |")
                    sections.append("")

        # Validation Records
        if report.validation_records:
            sections.append("## 4. Validation Records\n")
            for vr in report.validation_records:
                status_icon = "PASS" if vr.pass_fail == "pass" else "FAIL"
                sections.append(f"### {vr.validation_id}: {vr.model_name} v{vr.model_version} — {status_icon}\n")
                sections.append("| Attribute | Value |")
                sections.append("|-----------|-------|")
                sections.append(f"| Test Dataset | {vr.test_dataset_id} (N={vr.test_dataset_size:,}) |")
                sections.append(f"| Result | {vr.pass_fail.upper()} |")
                if vr.reviewer:
                    sections.append(f"| Reviewer | {vr.reviewer} |")
                if vr.review_date:
                    sections.append(f"| Review Date | {vr.review_date} |")
                sections.append("")

                if vr.metrics:
                    sections.append("**Metrics:**\n")
                    sections.append("| Metric | Value | Acceptance |")
                    sections.append("|--------|-------|------------|")
                    for k, v in vr.metrics.items():
                        criteria = vr.acceptance_criteria.get(k, "N/A")
                        sections.append(f"| {k} | {v:.4f} | {criteria} |")
                    sections.append("")

        # Integrity Verification
        sections.append("## 5. Record Integrity Verification\n")
        sections.append(
            "All audit records are linked via SHA-256 hash chain. Each record's hash "
            "incorporates the record ID, timestamp, operator, action, and data values.\n"
        )
        if report.records:
            sections.append(f"**Chain Hash:** `{chain_hash}`  ")
            sections.append(f"**Total Records in Chain:** {len(report.records)}  ")
            sections.append(f"**First Record:** {report.records[0].record_id}  ")
            sections.append(f"**Last Record:** {report.records[-1].record_id}  ")
        sections.append("")

        sections.append("---\n")
        sections.append("*Generated by Physical AI Oncology Trials — 21 CFR Part 11 Audit Trail Generator*  ")
        sections.append(f"*Report ID: {report.report_id} | Generated: {report.generated_date[:10]}*  ")
        sections.append("*RESEARCH USE ONLY — Production audit trails require validated systems*\n")

        return "\n".join(sections)


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_audit_trail_demo() -> dict[str, Any]:
    """
    Demonstrate 21 CFR Part 11 audit trail generation.

    Returns:
        Dictionary with demo results.
    """
    logger.info("=" * 60)
    logger.info("21 CFR PART 11 AUDIT TRAIL DEMO")
    logger.info("=" * 60)

    generator = AuditTrailGenerator(
        system_name="OncoTwin AI Training Platform",
        system_id="SYS-ONCOTWIN-001",
    )

    # Record a training run
    generator.record_training_run(
        operator="ml_engineer_01",
        model_name="TumorSeg-3D",
        model_version="1.2.0",
        dataset_id="DS-NSCLC-2026-Q1",
        dataset_size=12500,
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 100,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "augmentation": "random_flip,rotation,elastic",
        },
        final_metrics={
            "dice_coefficient": 0.912,
            "sensitivity": 0.945,
            "specificity": 0.987,
            "training_loss": 0.0342,
        },
        training_duration_seconds=28800.0,
        hardware_description="NVIDIA A100 80GB x 4",
        framework_version="PyTorch 2.3.0 + MONAI 1.4.0",
        random_seed=42,
    )

    # Record a configuration change
    generator.record_event(
        event_type=AuditEventType.CONFIG_CHANGE,
        operator="ml_engineer_01",
        action="Update inference threshold",
        description="Adjusted detection confidence threshold based on validation ROC analysis",
        previous_value="threshold=0.50",
        new_value="threshold=0.55",
        reason_for_change="ROC analysis showed improved specificity at 0.55 threshold with acceptable sensitivity",
    )

    # Record validation
    generator.record_validation(
        operator="validation_team",
        model_name="TumorSeg-3D",
        model_version="1.2.0",
        test_dataset_id="TS-NSCLC-EXT-001",
        test_dataset_size=500,
        metrics={
            "dice_coefficient": 0.908,
            "sensitivity": 0.941,
            "specificity": 0.985,
        },
        acceptance_criteria={
            "dice_coefficient": ">= 0.85",
            "sensitivity": ">= 0.90",
            "specificity": ">= 0.95",
        },
        pass_fail="pass",
        reviewer="Dr. Regulatory Scientist",
        notes="External validation on hold-out test set from 3 institutions",
    )

    # Record deployment
    generator.record_event(
        event_type=AuditEventType.DEPLOYMENT,
        operator="devops_team",
        action="Deploy TumorSeg-3D v1.2.0 to staging",
        description="Deployed validated model v1.2.0 to staging environment for clinical UAT",
        severity=AuditSeverity.INFO,
        metadata={"environment": "staging", "model_version": "1.2.0"},
    )

    # Generate report
    report = generator.generate_report()

    print(f"\nAudit Report: {report.report_id}")
    print(f"System: {report.system_name}")
    print(f"Records: {len(report.records)}")
    print(f"Training Runs: {len(report.training_records)}")
    print(f"Validations: {len(report.validation_records)}")
    print(f"Document length: {len(report.generated_markdown):,} characters")

    return {
        "report_id": report.report_id,
        "record_count": len(report.records),
        "training_count": len(report.training_records),
        "validation_count": len(report.validation_records),
        "document_length": len(report.generated_markdown),
        "status": "demo_complete",
    }


if __name__ == "__main__":
    result = run_audit_trail_demo()
    print(f"\nDemo completed: {result}")
