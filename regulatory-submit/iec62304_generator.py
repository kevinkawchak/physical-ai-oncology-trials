"""
=============================================================================
IEC 62304 Software Lifecycle Documentation Generator
for Physical AI Oncology Trials
=============================================================================

Generates IEC 62304-compliant software lifecycle documents from repository
metadata and code structure, including Software Development Plans (SDP),
Software Requirements Specifications (SRS), Software Architecture Documents
(SAD), and risk analysis matrices for AI/ML-enabled oncology devices.

CLINICAL CONTEXT:
-----------------
IEC 62304:2015 (Medical device software — Software life cycle processes)
defines the software lifecycle activities required for medical device
software development. For AI/ML-enabled oncology devices, IEC 62304
compliance is a prerequisite for FDA submission and CE marking.

Key IEC 62304 deliverables:
  - Software Development Plan (SDP)
  - Software Requirements Specification (SRS)
  - Software Architecture Document (SAD)
  - Software Design Specification (SDS)
  - Software Unit/Integration/System Test Plans
  - Risk analysis with software safety classification

Software safety classification per IEC 62304:
  - Class A: No injury or damage to health possible
  - Class B: Non-serious injury possible
  - Class C: Death or serious injury possible

USE CASES COVERED:
    1. SDP generation for AI oncology device development projects
    2. SRS generation linking clinical needs to software requirements
    3. SAD generation documenting AI/ML system architecture
    4. Risk analysis matrix generation per ISO 14971 integration

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - IEC 62304:2015 — Medical device software lifecycle processes
    - IEC 62304:2015/AMD 1:2015 — Amendment 1
    - ISO 14971:2019 — Risk management for medical devices
    - IEC 82304-1:2016 — Health software product safety
    - FDA Guidance: Content of Premarket Submissions for Device Software Functions
    - EU MDR 2017/745 Annex I, Chapter II (software requirements)

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Generated documents require review by qualified software quality and
    regulatory professionals before use in actual submissions.

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


class SafetyClass(Enum):
    """IEC 62304 software safety classification."""

    CLASS_A = "A"
    CLASS_B = "B"
    CLASS_C = "C"


class RequirementType(Enum):
    """Software requirement categories."""

    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    INTERFACE = "interface"
    SAFETY = "safety"
    SECURITY = "security"
    USABILITY = "usability"
    REGULATORY = "regulatory"
    DATA = "data"


class RiskSeverity(Enum):
    """Risk severity levels per ISO 14971."""

    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    SERIOUS = "serious"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class RiskProbability(Enum):
    """Risk probability levels per ISO 14971."""

    INCREDIBLE = "incredible"
    IMPROBABLE = "improbable"
    REMOTE = "remote"
    OCCASIONAL = "occasional"
    PROBABLE = "probable"
    FREQUENT = "frequent"


@dataclass
class SoftwareItem:
    """Software item (module/component) in the architecture."""

    item_id: str
    name: str
    description: str
    safety_class: SafetyClass
    parent_item: str = ""
    language: str = "Python"
    dependencies: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    soup_components: list[str] = field(default_factory=list)


@dataclass
class SoftwareRequirement:
    """Individual software requirement."""

    req_id: str
    requirement_type: RequirementType
    title: str
    description: str
    acceptance_criteria: str = ""
    source: str = ""
    priority: str = "required"  # "required", "recommended", "optional"
    safety_relevant: bool = False
    traced_to: list[str] = field(default_factory=list)


@dataclass
class RiskEntry:
    """Risk analysis entry linking hazard to software cause."""

    risk_id: str
    hazard: str
    hazardous_situation: str
    harm: str
    severity: RiskSeverity
    probability: RiskProbability
    risk_level: str = ""
    software_cause: str = ""
    software_item: str = ""
    mitigation: str = ""
    residual_risk_acceptable: bool = True
    verification_method: str = ""


@dataclass
class IEC62304DocumentSet:
    """Complete IEC 62304 document set for a software project."""

    project_name: str
    device_name: str
    safety_class: SafetyClass
    version: str = "1.0.0"
    sponsor: str = ""
    software_items: list[SoftwareItem] = field(default_factory=list)
    requirements: list[SoftwareRequirement] = field(default_factory=list)
    risks: list[RiskEntry] = field(default_factory=list)
    development_model: str = "Agile with waterfall gates"
    configuration_management: str = "Git version control with protected branches"
    problem_resolution: str = "Issue tracker with severity classification"
    generated_date: str = ""
    sdp_markdown: str = ""
    srs_markdown: str = ""
    sad_markdown: str = ""
    risk_markdown: str = ""


# =============================================================================
# SECTION 2: TEMPLATES AND DEFAULTS
# =============================================================================

RISK_MATRIX: dict[str, dict[str, str]] = {
    "catastrophic": {
        "frequent": "UNACCEPTABLE",
        "probable": "UNACCEPTABLE",
        "occasional": "UNACCEPTABLE",
        "remote": "ALARP",
        "improbable": "ALARP",
        "incredible": "ACCEPTABLE",
    },
    "critical": {
        "frequent": "UNACCEPTABLE",
        "probable": "UNACCEPTABLE",
        "occasional": "ALARP",
        "remote": "ALARP",
        "improbable": "ACCEPTABLE",
        "incredible": "ACCEPTABLE",
    },
    "serious": {
        "frequent": "UNACCEPTABLE",
        "probable": "ALARP",
        "occasional": "ALARP",
        "remote": "ACCEPTABLE",
        "improbable": "ACCEPTABLE",
        "incredible": "ACCEPTABLE",
    },
    "minor": {
        "frequent": "ALARP",
        "probable": "ALARP",
        "occasional": "ACCEPTABLE",
        "remote": "ACCEPTABLE",
        "improbable": "ACCEPTABLE",
        "incredible": "ACCEPTABLE",
    },
    "negligible": {
        "frequent": "ACCEPTABLE",
        "probable": "ACCEPTABLE",
        "occasional": "ACCEPTABLE",
        "remote": "ACCEPTABLE",
        "improbable": "ACCEPTABLE",
        "incredible": "ACCEPTABLE",
    },
}

DEFAULT_SDP_ACTIVITIES: list[dict[str, str]] = [
    {"phase": "Planning", "activity": "Software Development Planning", "reference": "IEC 62304 Section 5.1"},
    {"phase": "Planning", "activity": "Software Safety Classification", "reference": "IEC 62304 Section 4.3"},
    {"phase": "Requirements", "activity": "Software Requirements Analysis", "reference": "IEC 62304 Section 5.2"},
    {"phase": "Architecture", "activity": "Software Architectural Design", "reference": "IEC 62304 Section 5.3"},
    {"phase": "Design", "activity": "Software Detailed Design", "reference": "IEC 62304 Section 5.4"},
    {"phase": "Implementation", "activity": "Software Unit Implementation", "reference": "IEC 62304 Section 5.5"},
    {"phase": "Verification", "activity": "Software Unit Verification", "reference": "IEC 62304 Section 5.5.5"},
    {
        "phase": "Integration",
        "activity": "Software Integration and Integration Testing",
        "reference": "IEC 62304 Section 5.6",
    },
    {"phase": "System Testing", "activity": "Software System Testing", "reference": "IEC 62304 Section 5.7"},
    {"phase": "Release", "activity": "Software Release", "reference": "IEC 62304 Section 5.8"},
    {"phase": "Maintenance", "activity": "Software Maintenance", "reference": "IEC 62304 Section 6"},
    {"phase": "Risk Management", "activity": "Software Risk Management", "reference": "IEC 62304 Section 7"},
    {"phase": "Configuration", "activity": "Software Configuration Management", "reference": "IEC 62304 Section 8"},
    {"phase": "Problem Resolution", "activity": "Software Problem Resolution", "reference": "IEC 62304 Section 9"},
]

ONCOLOGY_AI_REQUIREMENTS_TEMPLATE: list[dict[str, Any]] = [
    {
        "type": RequirementType.FUNCTIONAL,
        "title": "AI Model Inference",
        "description": "The software shall execute AI/ML model inference on input clinical data and return predictions within specified latency bounds.",
        "acceptance": "Model inference completes within 5 seconds for standard input volumes.",
        "safety": True,
    },
    {
        "type": RequirementType.FUNCTIONAL,
        "title": "Input Data Validation",
        "description": "The software shall validate all input clinical data against expected formats, ranges, and completeness before processing.",
        "acceptance": "Invalid inputs are rejected with informative error messages; no partial processing occurs.",
        "safety": True,
    },
    {
        "type": RequirementType.PERFORMANCE,
        "title": "Model Accuracy",
        "description": "The AI/ML model shall achieve performance metrics meeting or exceeding the acceptance criteria defined in the clinical validation protocol.",
        "acceptance": "Primary performance metric >= authorized threshold on locked test set.",
        "safety": True,
    },
    {
        "type": RequirementType.SAFETY,
        "title": "Graceful Degradation",
        "description": "The software shall degrade gracefully when encountering out-of-distribution inputs, incomplete data, or hardware failures.",
        "acceptance": "System displays appropriate warning and prevents output of unreliable results.",
        "safety": True,
    },
    {
        "type": RequirementType.SAFETY,
        "title": "Human Override",
        "description": "The software shall provide a mechanism for the clinician to override or reject any AI-generated recommendation.",
        "acceptance": "Override mechanism accessible within 2 interactions from any AI output screen.",
        "safety": True,
    },
    {
        "type": RequirementType.SECURITY,
        "title": "Data Encryption",
        "description": "The software shall encrypt all patient health information (PHI) at rest and in transit per HIPAA requirements.",
        "acceptance": "AES-256 encryption at rest; TLS 1.3 in transit.",
        "safety": False,
    },
    {
        "type": RequirementType.SECURITY,
        "title": "Authentication and Access Control",
        "description": "The software shall implement role-based access control with multi-factor authentication for clinical users.",
        "acceptance": "MFA enforced for all clinical user accounts; audit trail for access events.",
        "safety": False,
    },
    {
        "type": RequirementType.REGULATORY,
        "title": "Audit Trail",
        "description": "The software shall maintain a tamper-evident audit trail of all AI model predictions, user interactions, and configuration changes per 21 CFR Part 11.",
        "acceptance": "Audit records include timestamp, user ID, action, and data hash.",
        "safety": False,
    },
    {
        "type": RequirementType.USABILITY,
        "title": "Clinical Workflow Integration",
        "description": "The software shall integrate into the existing clinical workflow without requiring more than 3 additional user interactions per case.",
        "acceptance": "Usability study demonstrates < 3 additional clicks per case vs. standard workflow.",
        "safety": False,
    },
    {
        "type": RequirementType.DATA,
        "title": "DICOM Interoperability",
        "description": "The software shall accept DICOM-compliant imaging data and produce DICOM-compatible output where applicable.",
        "acceptance": "DICOM conformance statement validates compatibility with major PACS systems.",
        "safety": False,
    },
]


# =============================================================================
# SECTION 3: IEC 62304 DOCUMENT GENERATOR
# =============================================================================


class IEC62304Generator:
    """
    Generates IEC 62304-compliant software lifecycle documents for AI/ML-enabled
    oncology devices.

    Produces Software Development Plans (SDP), Software Requirements Specifications
    (SRS), Software Architecture Documents (SAD), and risk analysis matrices
    as structured Markdown documents.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Complete document set generation from project metadata
    2. Individual document generation (SDP, SRS, SAD, Risk)
    3. Custom requirement and architecture population
    4. Risk analysis matrix with ISO 14971 integration
    """

    def __init__(self, sponsor: str = "") -> None:
        """
        Initialize IEC 62304 document generator.

        Args:
            sponsor: Sponsoring organization name.
        """
        self.sponsor = sponsor
        logger.info("IEC62304Generator initialized")

    def create_document_set(
        self,
        project_name: str,
        device_name: str,
        safety_class: SafetyClass,
        version: str = "1.0.0",
    ) -> IEC62304DocumentSet:
        """
        Create a new IEC 62304 document set.

        Args:
            project_name: Software project name.
            device_name: Medical device name.
            safety_class: IEC 62304 software safety classification.
            version: Software version.

        Returns:
            IEC62304DocumentSet with initialized structure.
        """
        doc_set = IEC62304DocumentSet(
            project_name=project_name,
            device_name=device_name,
            safety_class=safety_class,
            version=version,
            sponsor=self.sponsor,
            generated_date=datetime.now().isoformat(),
        )

        logger.info(f"Document set created: {project_name} (Class {safety_class.value})")
        return doc_set

    def add_software_item(self, doc_set: IEC62304DocumentSet, item: SoftwareItem) -> None:
        """Add a software item to the architecture."""
        doc_set.software_items.append(item)

    def add_requirement(self, doc_set: IEC62304DocumentSet, requirement: SoftwareRequirement) -> None:
        """Add a software requirement."""
        doc_set.requirements.append(requirement)

    def add_risk(self, doc_set: IEC62304DocumentSet, risk: RiskEntry) -> None:
        """Add a risk analysis entry."""
        risk.risk_level = RISK_MATRIX.get(risk.severity.value, {}).get(risk.probability.value, "UNKNOWN")
        doc_set.risks.append(risk)

    def populate_default_requirements(self, doc_set: IEC62304DocumentSet) -> None:
        """
        Populate document set with default oncology AI requirements.

        Args:
            doc_set: Document set to populate.
        """
        for i, tmpl in enumerate(ONCOLOGY_AI_REQUIREMENTS_TEMPLATE, 1):
            req = SoftwareRequirement(
                req_id=f"SRS-{i:03d}",
                requirement_type=tmpl["type"],
                title=tmpl["title"],
                description=tmpl["description"],
                acceptance_criteria=tmpl["acceptance"],
                safety_relevant=tmpl["safety"],
                source="IEC 62304 / FDA AI/ML Guidance",
            )
            doc_set.requirements.append(req)

        logger.info(f"Populated {len(ONCOLOGY_AI_REQUIREMENTS_TEMPLATE)} default requirements")

    def populate_sample_risks(self, doc_set: IEC62304DocumentSet) -> None:
        """
        Populate sample risk entries for oncology AI devices.

        Args:
            doc_set: Document set to populate.
        """
        sample_risks = [
            {
                "hazard": "Incorrect AI prediction",
                "situation": "AI model produces incorrect tumor classification leading to wrong treatment",
                "harm": "Delayed or inappropriate cancer treatment",
                "severity": RiskSeverity.CRITICAL,
                "probability": RiskProbability.REMOTE,
                "sw_cause": "Model error on out-of-distribution input",
                "mitigation": "Input validation, confidence thresholds, clinician review requirement",
                "verification": "Clinical validation study with independent test set",
            },
            {
                "hazard": "Software crash during inference",
                "situation": "Application crashes during AI inference leaving clinician without result",
                "harm": "Treatment delay due to unavailable AI assistance",
                "severity": RiskSeverity.SERIOUS,
                "probability": RiskProbability.IMPROBABLE,
                "sw_cause": "Unhandled exception in model inference pipeline",
                "mitigation": "Exception handling, graceful degradation, fallback to manual workflow",
                "verification": "Fault injection testing, stress testing",
            },
            {
                "hazard": "Data corruption",
                "situation": "Patient data corrupted during processing produces erroneous output",
                "harm": "Clinical decision based on corrupted data",
                "severity": RiskSeverity.CRITICAL,
                "probability": RiskProbability.IMPROBABLE,
                "sw_cause": "Memory error, race condition, or storage failure",
                "mitigation": "Data integrity checksums, input/output validation, redundant processing",
                "verification": "Data integrity testing, checksums verification",
            },
            {
                "hazard": "Unauthorized access to PHI",
                "situation": "Patient health information accessed by unauthorized user via the software",
                "harm": "Privacy breach affecting patient confidentiality",
                "severity": RiskSeverity.SERIOUS,
                "probability": RiskProbability.REMOTE,
                "sw_cause": "Insufficient access control or authentication bypass",
                "mitigation": "RBAC, MFA, encryption, audit logging per 21 CFR Part 11",
                "verification": "Penetration testing, access control verification",
            },
            {
                "hazard": "Model performance degradation",
                "situation": "AI model performance degrades due to data drift without detection",
                "harm": "Clinician receives unreliable AI assistance over extended period",
                "severity": RiskSeverity.SERIOUS,
                "probability": RiskProbability.OCCASIONAL,
                "sw_cause": "Absence of performance monitoring or drift detection",
                "mitigation": "Continuous performance monitoring, drift detection alerts, PCCP retraining protocol",
                "verification": "Drift simulation testing, monitoring system validation",
            },
        ]

        for i, r in enumerate(sample_risks, 1):
            risk = RiskEntry(
                risk_id=f"RSK-{i:03d}",
                hazard=r["hazard"],
                hazardous_situation=r["situation"],
                harm=r["harm"],
                severity=r["severity"],
                probability=r["probability"],
                software_cause=r["sw_cause"],
                mitigation=r["mitigation"],
                verification_method=r["verification"],
            )
            self.add_risk(doc_set, risk)

        logger.info(f"Populated {len(sample_risks)} sample risk entries")

    def generate_sdp(self, doc_set: IEC62304DocumentSet) -> str:
        """
        Generate Software Development Plan as Markdown.

        Args:
            doc_set: Document set to generate SDP from.

        Returns:
            SDP as Markdown string.
        """
        sections: list[str] = []

        sections.append("# Software Development Plan (SDP)\n")
        sections.append(f"**Project:** {doc_set.project_name}  ")
        sections.append(f"**Device:** {doc_set.device_name}  ")
        sections.append(f"**Version:** {doc_set.version}  ")
        sections.append(f"**Safety Classification:** IEC 62304 Class {doc_set.safety_class.value}  ")
        sections.append(f"**Date:** {doc_set.generated_date[:10]}  ")
        if doc_set.sponsor:
            sections.append(f"**Sponsor:** {doc_set.sponsor}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** Generated document for planning purposes.\n")

        sections.append("## 1. Purpose and Scope\n")
        sections.append(
            f"This Software Development Plan defines the lifecycle activities, processes, and "
            f"deliverables for the development of {doc_set.project_name}, a Class {doc_set.safety_class.value} "
            f"medical device software component of {doc_set.device_name}.\n"
        )

        sections.append("## 2. Software Safety Classification\n")
        sections.append(f"**Classification:** Class {doc_set.safety_class.value}\n")
        class_desc = {
            SafetyClass.CLASS_A: "Software system that cannot contribute to a hazardous situation.",
            SafetyClass.CLASS_B: "Software system that can contribute to a hazardous situation that does not result in unacceptable risk after risk control measures are applied, where the resulting harm is non-serious injury.",
            SafetyClass.CLASS_C: "Software system that can contribute to a hazardous situation that results in unacceptable risk after risk control measures are applied, where the resulting harm is death or serious injury.",
        }
        sections.append(f"**Definition:** {class_desc[doc_set.safety_class]}\n")

        sections.append("## 3. Development Lifecycle Activities\n")
        sections.append("| Phase | Activity | IEC 62304 Reference |")
        sections.append("|-------|----------|-------------------|")
        for act in DEFAULT_SDP_ACTIVITIES:
            sections.append(f"| {act['phase']} | {act['activity']} | {act['reference']} |")
        sections.append("")

        sections.append("## 4. Development Model\n")
        sections.append(f"**Model:** {doc_set.development_model}\n")

        sections.append("## 5. Configuration Management\n")
        sections.append(f"**Approach:** {doc_set.configuration_management}\n")

        sections.append("## 6. Problem Resolution\n")
        sections.append(f"**Approach:** {doc_set.problem_resolution}\n")

        sections.append("---\n")
        sections.append("*Generated per IEC 62304:2015 Section 5.1 — Software Development Planning*\n")

        markdown = "\n".join(sections)
        doc_set.sdp_markdown = markdown
        logger.info(f"SDP generated for {doc_set.project_name}")
        return markdown

    def generate_srs(self, doc_set: IEC62304DocumentSet) -> str:
        """
        Generate Software Requirements Specification as Markdown.

        Args:
            doc_set: Document set to generate SRS from.

        Returns:
            SRS as Markdown string.
        """
        sections: list[str] = []

        sections.append("# Software Requirements Specification (SRS)\n")
        sections.append(f"**Project:** {doc_set.project_name}  ")
        sections.append(f"**Device:** {doc_set.device_name}  ")
        sections.append(f"**Version:** {doc_set.version}  ")
        sections.append(f"**Safety Classification:** IEC 62304 Class {doc_set.safety_class.value}  ")
        sections.append(f"**Date:** {doc_set.generated_date[:10]}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** Generated document for planning purposes.\n")

        # Group requirements by type
        by_type: dict[str, list[SoftwareRequirement]] = {}
        for req in doc_set.requirements:
            key = req.requirement_type.value
            by_type.setdefault(key, []).append(req)

        section_num = 1
        for req_type, reqs in by_type.items():
            sections.append(f"## {section_num}. {req_type.replace('_', ' ').title()} Requirements\n")
            for req in reqs:
                safety_tag = " [SAFETY]" if req.safety_relevant else ""
                sections.append(f"### {req.req_id}: {req.title}{safety_tag}\n")
                sections.append(f"**Description:** {req.description}\n")
                if req.acceptance_criteria:
                    sections.append(f"**Acceptance Criteria:** {req.acceptance_criteria}\n")
                if req.source:
                    sections.append(f"**Source:** {req.source}\n")
                sections.append(f"**Priority:** {req.priority}\n")
            section_num += 1

        # Requirements summary
        total = len(doc_set.requirements)
        safety_count = sum(1 for r in doc_set.requirements if r.safety_relevant)
        sections.append("## Requirements Summary\n")
        sections.append("| Metric | Count |")
        sections.append("|--------|-------|")
        sections.append(f"| Total Requirements | {total} |")
        sections.append(f"| Safety-Relevant | {safety_count} |")
        sections.append(f"| Non-Safety | {total - safety_count} |")
        sections.append("")

        sections.append("---\n")
        sections.append("*Generated per IEC 62304:2015 Section 5.2 — Software Requirements Analysis*\n")

        markdown = "\n".join(sections)
        doc_set.srs_markdown = markdown
        logger.info(f"SRS generated for {doc_set.project_name}: {total} requirements")
        return markdown

    def generate_sad(self, doc_set: IEC62304DocumentSet) -> str:
        """
        Generate Software Architecture Document as Markdown.

        Args:
            doc_set: Document set to generate SAD from.

        Returns:
            SAD as Markdown string.
        """
        sections: list[str] = []

        sections.append("# Software Architecture Document (SAD)\n")
        sections.append(f"**Project:** {doc_set.project_name}  ")
        sections.append(f"**Device:** {doc_set.device_name}  ")
        sections.append(f"**Version:** {doc_set.version}  ")
        sections.append(f"**Safety Classification:** IEC 62304 Class {doc_set.safety_class.value}  ")
        sections.append(f"**Date:** {doc_set.generated_date[:10]}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** Generated document for planning purposes.\n")

        # Architecture Overview
        sections.append("## 1. Architecture Overview\n")
        sections.append(
            f"The {doc_set.project_name} software architecture is organized into "
            f"{len(doc_set.software_items)} software items, each classified according to "
            f"IEC 62304 safety classification.\n"
        )

        # Software Items
        if doc_set.software_items:
            sections.append("## 2. Software Items\n")
            sections.append("| Item ID | Name | Safety Class | Language | Dependencies |")
            sections.append("|---------|------|-------------|----------|-------------|")
            for item in doc_set.software_items:
                deps = ", ".join(item.dependencies[:3]) if item.dependencies else "None"
                sections.append(
                    f"| {item.item_id} | {item.name} | Class {item.safety_class.value} | {item.language} | {deps} |"
                )
            sections.append("")

            for item in doc_set.software_items:
                sections.append(f"### {item.item_id}: {item.name}\n")
                sections.append(f"**Description:** {item.description}  ")
                sections.append(f"**Safety Class:** {item.safety_class.value}  ")
                if item.interfaces:
                    sections.append(f"**Interfaces:** {', '.join(item.interfaces)}  ")
                if item.soup_components:
                    sections.append(f"**SOUP Components:** {', '.join(item.soup_components)}  ")
                sections.append("")

        # SOUP (Software of Unknown Provenance)
        all_soup = list({s for item in doc_set.software_items for s in item.soup_components})
        if all_soup:
            sections.append("## 3. SOUP Components\n")
            sections.append("| Component | Used By |")
            sections.append("|-----------|---------|")
            for soup in sorted(all_soup):
                users = [item.name for item in doc_set.software_items if soup in item.soup_components]
                sections.append(f"| {soup} | {', '.join(users)} |")
            sections.append("")

        sections.append("---\n")
        sections.append("*Generated per IEC 62304:2015 Section 5.3 — Software Architectural Design*\n")

        markdown = "\n".join(sections)
        doc_set.sad_markdown = markdown
        logger.info(f"SAD generated for {doc_set.project_name}: {len(doc_set.software_items)} items")
        return markdown

    def generate_risk_analysis(self, doc_set: IEC62304DocumentSet) -> str:
        """
        Generate risk analysis matrix as Markdown.

        Args:
            doc_set: Document set to generate risk analysis from.

        Returns:
            Risk analysis as Markdown string.
        """
        sections: list[str] = []

        sections.append("# Software Risk Analysis\n")
        sections.append(f"**Project:** {doc_set.project_name}  ")
        sections.append(f"**Device:** {doc_set.device_name}  ")
        sections.append(f"**Version:** {doc_set.version}  ")
        sections.append(f"**Date:** {doc_set.generated_date[:10]}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** Generated per ISO 14971:2019 integration with IEC 62304.\n")

        # Risk Matrix Legend
        sections.append("## 1. Risk Acceptability Matrix\n")
        sections.append(
            "| Severity \\ Probability | Frequent | Probable | Occasional | Remote | Improbable | Incredible |"
        )
        sections.append(
            "|----------------------|----------|----------|------------|--------|------------|------------|"
        )
        for sev in ["catastrophic", "critical", "serious", "minor", "negligible"]:
            row = f"| {sev.title()} |"
            for prob in ["frequent", "probable", "occasional", "remote", "improbable", "incredible"]:
                level = RISK_MATRIX[sev][prob]
                row += f" {level} |"
            sections.append(row)
        sections.append("")

        # Risk Entries
        if doc_set.risks:
            sections.append("## 2. Identified Risks\n")
            sections.append("| ID | Hazard | Severity | Probability | Risk Level | Mitigation |")
            sections.append("|----|--------|----------|-------------|------------|------------|")
            for risk in doc_set.risks:
                mit_preview = risk.mitigation[:40] + "..." if len(risk.mitigation) > 40 else risk.mitigation
                sections.append(
                    f"| {risk.risk_id} | {risk.hazard} | {risk.severity.value} | "
                    f"{risk.probability.value} | {risk.risk_level} | {mit_preview} |"
                )
            sections.append("")

            for risk in doc_set.risks:
                sections.append(f"### {risk.risk_id}: {risk.hazard}\n")
                sections.append(f"**Hazardous Situation:** {risk.hazardous_situation}  ")
                sections.append(f"**Harm:** {risk.harm}  ")
                sections.append(f"**Severity:** {risk.severity.value} | **Probability:** {risk.probability.value}  ")
                sections.append(f"**Risk Level:** {risk.risk_level}  ")
                if risk.software_cause:
                    sections.append(f"**Software Cause:** {risk.software_cause}  ")
                sections.append(f"**Mitigation:** {risk.mitigation}  ")
                sections.append(f"**Residual Risk Acceptable:** {'Yes' if risk.residual_risk_acceptable else 'No'}  ")
                if risk.verification_method:
                    sections.append(f"**Verification:** {risk.verification_method}  ")
                sections.append("")

        # Summary
        sections.append("## 3. Risk Summary\n")
        total = len(doc_set.risks)
        unacceptable = sum(1 for r in doc_set.risks if r.risk_level == "UNACCEPTABLE")
        alarp = sum(1 for r in doc_set.risks if r.risk_level == "ALARP")
        acceptable = sum(1 for r in doc_set.risks if r.risk_level == "ACCEPTABLE")
        sections.append("| Category | Count |")
        sections.append("|----------|-------|")
        sections.append(f"| Total Risks | {total} |")
        sections.append(f"| Unacceptable | {unacceptable} |")
        sections.append(f"| ALARP (As Low As Reasonably Practicable) | {alarp} |")
        sections.append(f"| Acceptable | {acceptable} |")
        sections.append("")

        sections.append("---\n")
        sections.append("*Generated per IEC 62304:2015 Section 7 and ISO 14971:2019*\n")

        markdown = "\n".join(sections)
        doc_set.risk_markdown = markdown
        logger.info(f"Risk analysis generated for {doc_set.project_name}: {total} risks")
        return markdown

    def generate_all(self, doc_set: IEC62304DocumentSet) -> dict[str, str]:
        """
        Generate all IEC 62304 documents.

        Args:
            doc_set: Document set to generate all documents from.

        Returns:
            Dictionary mapping document name to Markdown content.
        """
        return {
            "sdp": self.generate_sdp(doc_set),
            "srs": self.generate_srs(doc_set),
            "sad": self.generate_sad(doc_set),
            "risk_analysis": self.generate_risk_analysis(doc_set),
        }


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_iec62304_demo() -> dict[str, Any]:
    """
    Demonstrate IEC 62304 document generation for an AI oncology device.

    Returns:
        Dictionary with demo results.
    """
    logger.info("=" * 60)
    logger.info("IEC 62304 DOCUMENT GENERATOR DEMO")
    logger.info("=" * 60)

    generator = IEC62304Generator(sponsor="Physical AI Oncology Consortium")

    doc_set = generator.create_document_set(
        project_name="OncoTwin-SW",
        device_name="OncoTwin AI Surgical Planning System",
        safety_class=SafetyClass.CLASS_C,
        version="1.0.0",
    )

    # Add software items
    items = [
        SoftwareItem(
            item_id="SWI-001",
            name="AI Inference Engine",
            description="Core AI/ML model inference pipeline for tumor segmentation and path planning",
            safety_class=SafetyClass.CLASS_C,
            dependencies=["Input Validator", "Model Registry"],
            interfaces=["DICOM Input", "Clinical Display Output"],
            soup_components=["PyTorch", "ONNX Runtime", "NumPy"],
        ),
        SoftwareItem(
            item_id="SWI-002",
            name="Input Validator",
            description="Validates incoming DICOM data for format compliance, completeness, and quality",
            safety_class=SafetyClass.CLASS_B,
            interfaces=["DICOM Input", "Validation Report"],
            soup_components=["pydicom", "NumPy"],
        ),
        SoftwareItem(
            item_id="SWI-003",
            name="Clinical Display",
            description="User interface for displaying AI results, surgical plans, and clinician interaction",
            safety_class=SafetyClass.CLASS_B,
            dependencies=["AI Inference Engine"],
            interfaces=["Clinician UI", "Override Controls"],
            soup_components=["VTK", "Qt"],
        ),
        SoftwareItem(
            item_id="SWI-004",
            name="Audit Logger",
            description="21 CFR Part 11 compliant audit trail for all predictions and user actions",
            safety_class=SafetyClass.CLASS_B,
            interfaces=["Audit Database", "Compliance Reports"],
            soup_components=["SQLite", "hashlib"],
        ),
    ]
    for item in items:
        generator.add_software_item(doc_set, item)

    generator.populate_default_requirements(doc_set)
    generator.populate_sample_risks(doc_set)

    docs = generator.generate_all(doc_set)

    print(f"\nProject: {doc_set.project_name}")
    print(f"Safety Class: {doc_set.safety_class.value}")
    print(f"Software Items: {len(doc_set.software_items)}")
    print(f"Requirements: {len(doc_set.requirements)}")
    print(f"Risks: {len(doc_set.risks)}")
    print("\nGenerated Documents:")
    for name, content in docs.items():
        print(f"  {name}: {len(content):,} characters")

    return {
        "project": doc_set.project_name,
        "documents_generated": list(docs.keys()),
        "requirement_count": len(doc_set.requirements),
        "risk_count": len(doc_set.risks),
        "status": "demo_complete",
    }


if __name__ == "__main__":
    result = run_iec62304_demo()
    print(f"\nDemo completed: {result}")
