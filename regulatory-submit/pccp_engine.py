"""
=============================================================================
Predetermined Change Control Plan (PCCP) Template Engine
for Physical AI Oncology Trials
=============================================================================

Generates Predetermined Change Control Plan documents for AI/ML-enabled
oncology devices, defining pre-authorized modification boundaries per
FDA's August 2025 finalized PCCP guidance.

CLINICAL CONTEXT:
-----------------
Adaptive AI/ML models in oncology devices may require post-market updates
including retraining on new data, architecture refinements, and threshold
adjustments. The PCCP framework allows sponsors to pre-specify a range of
modifications that can be implemented without a new marketing submission,
provided they fall within the approved change control boundaries and
verification/validation (V&V) protocols.

Key PCCP components per FDA guidance:
  - Description of Modifications: What changes are permitted
  - Modification Protocol: How changes are implemented and controlled
  - Impact Assessment: How each change affects safety and effectiveness
  - Verification and Validation: Testing required before deployment
  - Transparency: Communication to users about device updates

USE CASES COVERED:
    1. PCCP for model retraining with new institutional data
    2. PCCP for performance threshold adjustments
    3. PCCP for data preprocessing pipeline updates
    4. PCCP for multi-indication expansion within approved scope

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - FDA PCCP Guidance for AI/ML Devices (Aug 2025, finalized)
    - FDA Draft Guidance: AI-Enabled Device Software Functions (Jan 2025)
    - 21 CFR Part 820 / ISO 13485 — Quality Management System
    - IEC 62304:2015 — Software Lifecycle for Medical Devices
    - ISO 14971:2019 — Risk Management for Medical Devices

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Requires institutional validation and regulatory review before deployment.
    Generated PCCP documents require legal and regulatory review before filing.

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

from __future__ import annotations

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


class ModificationType(Enum):
    """Types of AI/ML model modifications covered by PCCP."""

    RETRAINING = "model_retraining"
    ARCHITECTURE_UPDATE = "architecture_update"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    PREPROCESSING_UPDATE = "preprocessing_update"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_AUGMENTATION = "data_augmentation"
    DRIFT_ADAPTATION = "drift_adaptation"
    INDICATION_EXPANSION = "indication_expansion"


class RiskLevel(Enum):
    """Risk level classification for modifications."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ApprovalCategory(Enum):
    """Whether a modification is pre-authorized under PCCP or requires new submission."""

    PRE_AUTHORIZED = "pre_authorized_under_pccp"
    REQUIRES_NOTIFICATION = "requires_fda_notification"
    REQUIRES_NEW_SUBMISSION = "requires_new_submission"


@dataclass
class ModificationBoundary:
    """Defines the permitted boundary for a specific modification type."""

    modification_type: ModificationType
    description: str
    permitted_scope: str
    risk_level: RiskLevel
    approval_category: ApprovalCategory
    performance_thresholds: dict[str, float] = field(default_factory=dict)
    validation_requirements: list[str] = field(default_factory=list)
    monitoring_requirements: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)


@dataclass
class ValidationProtocol:
    """Verification and validation protocol for a modification type."""

    protocol_name: str
    modification_types: list[ModificationType]
    test_dataset_requirements: str = ""
    performance_acceptance_criteria: dict[str, str] = field(default_factory=dict)
    statistical_test: str = ""
    sample_size_rationale: str = ""
    regression_testing: bool = True
    bias_testing: bool = True
    robustness_testing: bool = True
    clinical_review_required: bool = False


@dataclass
class TransparencyPlan:
    """Plan for communicating modifications to users and FDA."""

    notification_triggers: list[str] = field(default_factory=list)
    user_communication_method: str = ""
    labeling_update_process: str = ""
    fda_reporting_schedule: str = ""
    public_disclosure_method: str = ""


@dataclass
class PCCPDocument:
    """Complete Predetermined Change Control Plan document."""

    document_id: str
    device_name: str
    sponsor: str
    submission_reference: str
    ai_model_names: list[str] = field(default_factory=list)
    modification_boundaries: list[ModificationBoundary] = field(default_factory=list)
    validation_protocols: list[ValidationProtocol] = field(default_factory=list)
    transparency_plan: TransparencyPlan = field(default_factory=TransparencyPlan)
    lifecycle_duration: str = "3 years from initial authorization"
    review_schedule: str = "Annual PCCP review and reporting"
    generated_date: str = ""
    generated_markdown: str = ""


# =============================================================================
# SECTION 2: PCCP TEMPLATES
# =============================================================================

DEFAULT_MODIFICATION_BOUNDARIES: dict[str, dict[str, Any]] = {
    "retraining_same_architecture": {
        "type": ModificationType.RETRAINING,
        "description": "Model retraining using the same architecture with updated training data",
        "permitted_scope": (
            "Retraining the existing model architecture with additional training data from "
            "the same data modalities and patient populations. Training data must meet the "
            "same quality and demographic representation standards as the original dataset."
        ),
        "risk_level": RiskLevel.LOW,
        "approval_category": ApprovalCategory.PRE_AUTHORIZED,
        "performance_thresholds": {
            "primary_metric_minimum": 0.90,
            "regression_tolerance": 0.02,
            "subgroup_parity_threshold": 0.05,
        },
        "validation_requirements": [
            "Performance on locked test set must meet or exceed acceptance criteria",
            "Subgroup analysis across age, sex, and race/ethnicity",
            "Regression testing on prior validation cohort",
            "Bias assessment comparing new vs. prior model",
        ],
        "monitoring_requirements": [
            "Monthly performance monitoring on production data",
            "Quarterly bias monitoring across demographic subgroups",
            "Automated drift detection with configurable alerting",
        ],
        "exclusions": [
            "Changes to model architecture or fundamental algorithm",
            "Expansion to new data modalities not in original authorization",
            "Training on data from populations outside original scope",
        ],
    },
    "threshold_adjustment": {
        "type": ModificationType.THRESHOLD_ADJUSTMENT,
        "description": "Adjustment of decision thresholds within pre-specified bounds",
        "permitted_scope": (
            "Operating point adjustments for classification or detection thresholds "
            "within the pre-validated range. Thresholds must remain within the "
            "manufacturer-specified operating envelope."
        ),
        "risk_level": RiskLevel.LOW,
        "approval_category": ApprovalCategory.PRE_AUTHORIZED,
        "performance_thresholds": {
            "sensitivity_minimum": 0.85,
            "specificity_minimum": 0.80,
            "ppv_minimum": 0.70,
        },
        "validation_requirements": [
            "ROC curve analysis at new operating point",
            "Clinical impact assessment of sensitivity/specificity trade-off",
            "Verification on locked validation dataset",
        ],
        "monitoring_requirements": [
            "Weekly performance metrics at new threshold",
            "Alert if performance drops below acceptance criteria within 30 days",
        ],
        "exclusions": [
            "Threshold changes outside the pre-validated range defined in the PCCP",
            "Simultaneous threshold changes to multiple coupled models",
        ],
    },
    "preprocessing_update": {
        "type": ModificationType.PREPROCESSING_UPDATE,
        "description": "Updates to data preprocessing pipeline",
        "permitted_scope": (
            "Modifications to image normalization, augmentation parameters, or "
            "feature extraction pipelines that do not change the fundamental input "
            "representation to the AI/ML model."
        ),
        "risk_level": RiskLevel.MODERATE,
        "approval_category": ApprovalCategory.PRE_AUTHORIZED,
        "performance_thresholds": {
            "input_distribution_shift_max": 0.10,
            "end_to_end_performance_minimum": 0.88,
        },
        "validation_requirements": [
            "End-to-end performance validation with updated preprocessing",
            "Input distribution analysis comparing pre/post preprocessing change",
            "Regression testing on representative clinical cases",
        ],
        "monitoring_requirements": [
            "Input data quality monitoring post-deployment",
            "End-to-end performance tracking for 90 days post-change",
        ],
        "exclusions": [
            "Changes to input data modalities (e.g., CT to MRI)",
            "Fundamental changes to feature representation",
        ],
    },
    "architecture_change": {
        "type": ModificationType.ARCHITECTURE_UPDATE,
        "description": "Model architecture modifications",
        "permitted_scope": "NOT pre-authorized under this PCCP.",
        "risk_level": RiskLevel.HIGH,
        "approval_category": ApprovalCategory.REQUIRES_NEW_SUBMISSION,
        "performance_thresholds": {},
        "validation_requirements": [
            "Full de novo validation per original marketing submission requirements",
        ],
        "monitoring_requirements": [],
        "exclusions": [
            "All architecture changes require a new or supplemental marketing submission",
        ],
    },
    "drift_adaptation": {
        "type": ModificationType.DRIFT_ADAPTATION,
        "description": "Automated drift detection and model refresh",
        "permitted_scope": (
            "Automated retraining triggered by detected data drift or performance degradation, "
            "using the same architecture and data pipeline. Drift triggers and retraining "
            "parameters are pre-specified in this PCCP."
        ),
        "risk_level": RiskLevel.MODERATE,
        "approval_category": ApprovalCategory.REQUIRES_NOTIFICATION,
        "performance_thresholds": {
            "drift_trigger_threshold": 0.15,
            "post_retrain_minimum": 0.90,
            "regression_tolerance": 0.03,
        },
        "validation_requirements": [
            "Automated validation suite execution after each retraining cycle",
            "Human-in-the-loop review before production deployment",
            "Drift characterization report documenting trigger conditions",
        ],
        "monitoring_requirements": [
            "Continuous drift monitoring with statistical process control",
            "Real-time performance dashboarding",
            "FDA notification within 30 days of drift-triggered retraining",
        ],
        "exclusions": [
            "Drift requiring architecture changes or new data modalities",
            "Performance degradation below the minimum acceptable threshold",
        ],
    },
}

DEFAULT_VALIDATION_PROTOCOLS: list[dict[str, Any]] = [
    {
        "name": "Standard Retraining Validation",
        "types": [ModificationType.RETRAINING, ModificationType.DRIFT_ADAPTATION],
        "test_dataset": "Locked external test set (N >= 500) with demographic stratification",
        "acceptance_criteria": {
            "Primary metric": ">= 90% of original authorized performance",
            "Subgroup parity": "No subgroup performance drops > 5 percentage points",
            "Regression": "No statistically significant degradation on prior test cases (p < 0.05)",
        },
        "statistical_test": "McNemar's test for paired comparison with Bonferroni correction",
        "sample_size_rationale": "500 cases provides 90% power to detect 5% performance difference (alpha=0.05)",
    },
    {
        "name": "Threshold Adjustment Validation",
        "types": [ModificationType.THRESHOLD_ADJUSTMENT],
        "test_dataset": "Locked validation set with known ground truth (N >= 200)",
        "acceptance_criteria": {
            "Sensitivity": ">= 85% at new operating point",
            "Specificity": ">= 80% at new operating point",
            "Clinical net benefit": "Positive across decision-analytic model",
        },
        "statistical_test": "DeLong test for AUC comparison, Wilson score confidence intervals",
        "sample_size_rationale": "200 cases for 95% CI width <= 7% on sensitivity estimate",
    },
    {
        "name": "Preprocessing Update Validation",
        "types": [ModificationType.PREPROCESSING_UPDATE, ModificationType.FEATURE_ENGINEERING],
        "test_dataset": "Representative clinical dataset (N >= 300) spanning acquisition protocols",
        "acceptance_criteria": {
            "End-to-end performance": ">= 88% on primary metric",
            "Input distribution": "KL divergence < 0.10 vs. original input distribution",
            "Edge cases": "No new failure modes on curated edge case set",
        },
        "statistical_test": "Two-sample Kolmogorov-Smirnov test for distribution comparison",
        "sample_size_rationale": "300 cases for adequate representation of acquisition variability",
    },
]


# =============================================================================
# SECTION 3: PCCP ENGINE
# =============================================================================


class PCCPEngine:
    """
    Generates Predetermined Change Control Plan documents for AI/ML-enabled
    oncology devices.

    Produces structured Markdown PCCP documents defining modification
    boundaries, validation protocols, and transparency requirements per
    FDA's August 2025 finalized guidance.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Standard PCCP for locked models with retraining provisions
    2. PCCP for adaptive models with automated drift detection
    3. Multi-model PCCP covering multiple AI components
    4. PCCP lifecycle management with annual review cycles
    """

    def __init__(self, sponsor: str, regulatory_contact: str = ""):
        """
        Initialize PCCP engine.

        Args:
            sponsor: Sponsoring organization name.
            regulatory_contact: Regulatory affairs contact name.
        """
        self.sponsor = sponsor
        self.regulatory_contact = regulatory_contact
        self._document_counter = 0

        logger.info(f"PCCPEngine initialized for sponsor: {sponsor}")

    def create_pccp(
        self,
        device_name: str,
        submission_reference: str,
        ai_model_names: list[str],
        lifecycle_duration: str = "3 years from initial authorization",
    ) -> PCCPDocument:
        """
        Create a new PCCP document with default structure.

        Args:
            device_name: Name of the AI/ML-enabled device.
            submission_reference: Reference to the marketing submission.
            ai_model_names: Names of AI/ML models covered by the PCCP.
            lifecycle_duration: Duration of PCCP validity.

        Returns:
            PCCPDocument with initialized structure.
        """
        self._document_counter += 1
        doc_id = f"PCCP-{datetime.now().strftime('%Y%m%d')}-{self._document_counter:04d}"

        document = PCCPDocument(
            document_id=doc_id,
            device_name=device_name,
            sponsor=self.sponsor,
            submission_reference=submission_reference,
            ai_model_names=ai_model_names,
            lifecycle_duration=lifecycle_duration,
            generated_date=datetime.now().isoformat(),
        )

        logger.info(f"PCCP document created: {doc_id} for {device_name}")
        return document

    def populate_default_boundaries(self, document: PCCPDocument) -> None:
        """
        Populate document with default modification boundaries.

        Args:
            document: PCCP document to populate.
        """
        for key, tmpl in DEFAULT_MODIFICATION_BOUNDARIES.items():
            boundary = ModificationBoundary(
                modification_type=tmpl["type"],
                description=tmpl["description"],
                permitted_scope=tmpl["permitted_scope"],
                risk_level=tmpl["risk_level"],
                approval_category=tmpl["approval_category"],
                performance_thresholds=dict(tmpl["performance_thresholds"]),
                validation_requirements=list(tmpl["validation_requirements"]),
                monitoring_requirements=list(tmpl["monitoring_requirements"]),
                exclusions=list(tmpl["exclusions"]),
            )
            document.modification_boundaries.append(boundary)

        logger.info(
            f"Populated {len(document.modification_boundaries)} modification boundaries for {document.document_id}"
        )

    def populate_default_validation(self, document: PCCPDocument) -> None:
        """
        Populate document with default validation protocols.

        Args:
            document: PCCP document to populate.
        """
        for tmpl in DEFAULT_VALIDATION_PROTOCOLS:
            protocol = ValidationProtocol(
                protocol_name=tmpl["name"],
                modification_types=list(tmpl["types"]),
                test_dataset_requirements=tmpl["test_dataset"],
                performance_acceptance_criteria=dict(tmpl["acceptance_criteria"]),
                statistical_test=tmpl["statistical_test"],
                sample_size_rationale=tmpl["sample_size_rationale"],
            )
            document.validation_protocols.append(protocol)

        logger.info(f"Populated {len(document.validation_protocols)} validation protocols for {document.document_id}")

    def set_transparency_plan(
        self,
        document: PCCPDocument,
        notification_triggers: list[str] | None = None,
        user_communication_method: str = "In-app notification and updated labeling",
        labeling_update_process: str = "Updated Instructions for Use (IFU) within 30 days of change deployment",
        fda_reporting_schedule: str = "Annual PCCP performance report; immediate notification for drift-triggered retraining",
        public_disclosure_method: str = "Version release notes published on manufacturer website",
    ) -> None:
        """
        Set the transparency plan for the PCCP document.

        Args:
            document: PCCP document to update.
            notification_triggers: Events that trigger user/FDA notification.
            user_communication_method: How users are notified of changes.
            labeling_update_process: Process for updating device labeling.
            fda_reporting_schedule: Schedule for reporting to FDA.
            public_disclosure_method: How changes are publicly disclosed.
        """
        if notification_triggers is None:
            notification_triggers = [
                "Any model retraining event (pre-authorized or otherwise)",
                "Decision threshold adjustment",
                "Performance metric falling below acceptance criteria",
                "Drift detection triggering automated retraining",
                "Any modification requiring FDA notification",
            ]

        document.transparency_plan = TransparencyPlan(
            notification_triggers=notification_triggers,
            user_communication_method=user_communication_method,
            labeling_update_process=labeling_update_process,
            fda_reporting_schedule=fda_reporting_schedule,
            public_disclosure_method=public_disclosure_method,
        )

    def add_custom_boundary(self, document: PCCPDocument, boundary: ModificationBoundary) -> None:
        """
        Add a custom modification boundary to the PCCP.

        Args:
            document: PCCP document to update.
            boundary: Custom modification boundary.
        """
        document.modification_boundaries.append(boundary)
        logger.info(f"Custom boundary added to {document.document_id}: {boundary.description}")

    def generate_markdown(self, document: PCCPDocument) -> str:
        """
        Generate the complete PCCP document as Markdown.

        Args:
            document: PCCP document to render.

        Returns:
            Complete PCCP as Markdown string.
        """
        sections: list[str] = []

        # Header
        sections.append("# Predetermined Change Control Plan (PCCP)\n")
        sections.append(f"**Document ID:** {document.document_id}  ")
        sections.append(f"**Device:** {document.device_name}  ")
        sections.append(f"**Sponsor:** {document.sponsor}  ")
        sections.append(f"**Submission Reference:** {document.submission_reference}  ")
        sections.append(f"**PCCP Lifecycle:** {document.lifecycle_duration}  ")
        sections.append(f"**Review Schedule:** {document.review_schedule}  ")
        sections.append(f"**Date:** {document.generated_date[:10]}  ")
        sections.append("")

        sections.append("> **RESEARCH USE ONLY.** This PCCP document is generated for planning purposes. ")
        sections.append("> Consult regulatory counsel before including in any FDA submission.\n")

        # AI/ML Models Covered
        sections.append("## 1. AI/ML Models Covered by This PCCP\n")
        for i, name in enumerate(document.ai_model_names, 1):
            sections.append(f"{i}. {name}")
        sections.append("")

        # Modification Boundaries
        sections.append("## 2. Description of Permitted Modifications\n")

        # Summary table
        sections.append("| Modification | Risk | Authorization | Scope |")
        sections.append("|-------------|------|---------------|-------|")
        for b in document.modification_boundaries:
            scope_preview = b.permitted_scope[:60] + "..." if len(b.permitted_scope) > 60 else b.permitted_scope
            sections.append(
                f"| {b.description[:40]} | {b.risk_level.value} | "
                f"{b.approval_category.value.replace('_', ' ')} | {scope_preview} |"
            )
        sections.append("")

        for i, boundary in enumerate(document.modification_boundaries, 1):
            sections.append(f"### 2.{i}. {boundary.description}\n")
            sections.append(f"**Risk Level:** {boundary.risk_level.value.upper()}  ")
            sections.append(f"**Authorization:** {boundary.approval_category.value.replace('_', ' ').title()}  \n")
            sections.append(f"**Permitted Scope:**\n{boundary.permitted_scope}\n")

            if boundary.performance_thresholds:
                sections.append("**Performance Thresholds:**\n")
                sections.append("| Metric | Threshold |")
                sections.append("|--------|-----------|")
                for metric, value in boundary.performance_thresholds.items():
                    sections.append(f"| {metric.replace('_', ' ').title()} | {value} |")
                sections.append("")

            if boundary.validation_requirements:
                sections.append("**Validation Requirements:**\n")
                for req in boundary.validation_requirements:
                    sections.append(f"- {req}")
                sections.append("")

            if boundary.monitoring_requirements:
                sections.append("**Post-Deployment Monitoring:**\n")
                for req in boundary.monitoring_requirements:
                    sections.append(f"- {req}")
                sections.append("")

            if boundary.exclusions:
                sections.append("**Exclusions (NOT covered by this PCCP):**\n")
                for excl in boundary.exclusions:
                    sections.append(f"- {excl}")
                sections.append("")

        # Validation Protocols
        if document.validation_protocols:
            sections.append("## 3. Verification and Validation Protocols\n")
            for i, protocol in enumerate(document.validation_protocols, 1):
                sections.append(f"### 3.{i}. {protocol.protocol_name}\n")
                types_str = ", ".join(t.value.replace("_", " ").title() for t in protocol.modification_types)
                sections.append(f"**Applies To:** {types_str}  ")
                if protocol.test_dataset_requirements:
                    sections.append(f"**Test Dataset:** {protocol.test_dataset_requirements}  ")
                if protocol.statistical_test:
                    sections.append(f"**Statistical Test:** {protocol.statistical_test}  ")
                if protocol.sample_size_rationale:
                    sections.append(f"**Sample Size:** {protocol.sample_size_rationale}  ")
                sections.append("")

                if protocol.performance_acceptance_criteria:
                    sections.append("**Acceptance Criteria:**\n")
                    sections.append("| Criterion | Requirement |")
                    sections.append("|-----------|-------------|")
                    for criterion, requirement in protocol.performance_acceptance_criteria.items():
                        sections.append(f"| {criterion} | {requirement} |")
                    sections.append("")

                flags = []
                if protocol.regression_testing:
                    flags.append("Regression testing")
                if protocol.bias_testing:
                    flags.append("Bias testing")
                if protocol.robustness_testing:
                    flags.append("Robustness testing")
                if protocol.clinical_review_required:
                    flags.append("Clinical review required")
                if flags:
                    sections.append(f"**Additional Testing:** {', '.join(flags)}\n")

        # Transparency Plan
        tp = document.transparency_plan
        sections.append("## 4. Transparency and Communication Plan\n")
        if tp.notification_triggers:
            sections.append("**Notification Triggers:**\n")
            for trigger in tp.notification_triggers:
                sections.append(f"- {trigger}")
            sections.append("")
        if tp.user_communication_method:
            sections.append(f"**User Communication:** {tp.user_communication_method}\n")
        if tp.labeling_update_process:
            sections.append(f"**Labeling Updates:** {tp.labeling_update_process}\n")
        if tp.fda_reporting_schedule:
            sections.append(f"**FDA Reporting:** {tp.fda_reporting_schedule}\n")
        if tp.public_disclosure_method:
            sections.append(f"**Public Disclosure:** {tp.public_disclosure_method}\n")

        # Footer
        sections.append("---\n")
        sections.append("*Generated by Physical AI Oncology Trials — PCCP Template Engine*  ")
        sections.append(f"*Document ID: {document.document_id} | Generated: {document.generated_date[:10]}*  ")
        sections.append("*RESEARCH USE ONLY — Not an official PCCP filing*\n")

        markdown = "\n".join(sections)
        document.generated_markdown = markdown

        logger.info(
            f"PCCP Markdown generated for {document.document_id}: "
            f"{len(document.modification_boundaries)} boundaries, "
            f"{len(document.validation_protocols)} validation protocols"
        )

        return markdown


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_pccp_engine_demo() -> dict[str, Any]:
    """
    Demonstrate PCCP document generation for an adaptive AI oncology device.

    Returns:
        Dictionary with demo results.
    """
    logger.info("=" * 60)
    logger.info("PCCP ENGINE DEMO")
    logger.info("=" * 60)

    engine = PCCPEngine(
        sponsor="Physical AI Oncology Consortium",
        regulatory_contact="Regulatory Affairs Office",
    )

    document = engine.create_pccp(
        device_name="OncoTwin AI Surgical Planning System",
        submission_reference="DN-2026-0042",
        ai_model_names=["TumorSeg-3D", "PathOptimizer-RL"],
        lifecycle_duration="3 years from initial De Novo authorization",
    )

    engine.populate_default_boundaries(document)
    engine.populate_default_validation(document)
    engine.set_transparency_plan(document)

    markdown = engine.generate_markdown(document)

    print(f"\nPCCP Document: {document.document_id}")
    print(f"Device: {document.device_name}")
    print(f"Models: {', '.join(document.ai_model_names)}")
    print(f"Boundaries: {len(document.modification_boundaries)}")
    print(f"Validation Protocols: {len(document.validation_protocols)}")
    print(f"Document length: {len(markdown):,} characters")

    return {
        "document_id": document.document_id,
        "boundary_count": len(document.modification_boundaries),
        "validation_count": len(document.validation_protocols),
        "document_length": len(markdown),
        "status": "demo_complete",
    }


if __name__ == "__main__":
    result = run_pccp_engine_demo()
    print(f"\nDemo completed: {result}")
