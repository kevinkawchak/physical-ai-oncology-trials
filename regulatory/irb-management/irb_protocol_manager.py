"""
=============================================================================
IRB Protocol Manager for Physical AI Oncology Trials
=============================================================================

Manages IRB protocol submissions, review tracking, and compliance for
AI-enabled oncology clinical trials, incorporating AI-specific review
requirements per HHS SACHRP recommendations and the MRCT Center
Framework (July 2025).

CLINICAL CONTEXT:
-----------------
AI-enabled oncology trials require IRB review addressing:
  - How AI is disclosed to participants in informed consent
  - Algorithmic bias assessment and mitigation plans
  - AI-specific re-identification risks
  - Human oversight of AI-influenced treatment decisions
  - Community-level harms from algorithmic bias
  - Data use scope (clinical care vs. AI model training)

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - HHS SACHRP: IRB Considerations on AI (2025)
    - MRCT Center: Framework for Review of AI in Clinical Research (Jul 2025)
    - Common Rule: 45 CFR 46
    - FDA Regulations: 21 CFR 50, 56
    - SPIRIT-AI / CONSORT-AI Reporting Extensions

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

class ProtocolStatus(Enum):
    """IRB protocol submission status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    REVISIONS_REQUESTED = "revisions_requested"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    APPROVED = "approved"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    CLOSED = "closed"


class ReviewType(Enum):
    """Type of IRB review."""
    FULL_BOARD = "full_board"
    EXPEDITED = "expedited"
    EXEMPT = "exempt"
    CONTINUING = "continuing_review"
    AMENDMENT = "amendment"
    ADVERSE_EVENT = "adverse_event_report"


class RiskCategory(Enum):
    """Risk level per Common Rule."""
    MINIMAL_RISK = "minimal_risk"
    GREATER_THAN_MINIMAL = "greater_than_minimal_risk"


@dataclass
class AIComponent:
    """AI component in a clinical trial protocol."""
    name: str
    function: str  # "diagnostic", "treatment_planning", "monitoring", "decision_support"
    autonomy_level: str = "advisory"  # "advisory", "semi_autonomous", "autonomous"
    training_data_description: str = ""
    bias_assessment: str = ""
    human_override: bool = True
    model_locked: bool = True
    fda_status: str = ""


@dataclass
class ConsentElement:
    """Element required in informed consent for AI trials."""
    element: str
    description: str
    required: bool = True
    ai_specific: bool = False
    regulatory_basis: str = ""
    template_text: str = ""


@dataclass
class ReviewChecklistItem:
    """IRB review checklist item."""
    category: str
    item: str
    status: str = "not_reviewed"  # "not_reviewed", "satisfactory", "needs_revision", "not_applicable"
    reviewer_notes: str = ""
    regulatory_reference: str = ""
    ai_specific: bool = False


@dataclass
class Protocol:
    """IRB protocol record."""
    protocol_id: str
    title: str
    pi_name: str
    institution: str
    status: ProtocolStatus = ProtocolStatus.DRAFT
    review_type: ReviewType = ReviewType.FULL_BOARD
    risk_category: RiskCategory = RiskCategory.GREATER_THAN_MINIMAL
    ai_components: list[AIComponent] = field(default_factory=list)
    participant_count: int = 0
    trial_phase: str = ""
    consent_elements: list[ConsentElement] = field(default_factory=list)
    submission_date: str = ""
    approval_date: str = ""
    expiration_date: str = ""
    amendments: list[dict] = field(default_factory=list)
    created_date: str = ""

    def to_dict(self) -> dict:
        return {
            "protocol_id": self.protocol_id,
            "title": self.title,
            "pi_name": self.pi_name,
            "institution": self.institution,
            "status": self.status.value,
            "review_type": self.review_type.value,
            "ai_components": len(self.ai_components),
            "participant_count": self.participant_count,
            "trial_phase": self.trial_phase
        }


# =============================================================================
# SECTION 2: AI-SPECIFIC CONSENT AND REVIEW REQUIREMENTS
# =============================================================================

# AI-specific informed consent elements per SACHRP and MRCT guidance
AI_CONSENT_ELEMENTS: list[dict[str, Any]] = [
    {
        "element": "AI involvement disclosure",
        "description": "Clear statement that AI technology is used in the trial and its role in treatment/diagnosis",
        "required": True,
        "regulatory_basis": "45 CFR 46.116(b)(1), SACHRP Recommendations",
        "template_text": (
            "This clinical trial uses artificial intelligence (AI) technology as part of "
            "[describe function: treatment planning / diagnosis / monitoring]. The AI system "
            "analyzes [describe data used] to [describe what AI does]. All AI recommendations "
            "are reviewed by your clinical care team before any decisions are made."
        )
    },
    {
        "element": "AI limitations and risks",
        "description": "Description of known AI limitations, potential errors, and associated risks",
        "required": True,
        "regulatory_basis": "45 CFR 46.116(b)(2)",
        "template_text": (
            "The AI system may produce incorrect or incomplete recommendations. Like all "
            "medical technologies, it has limitations including [describe known limitations]. "
            "Your doctors will always review AI outputs and make final treatment decisions."
        )
    },
    {
        "element": "Human oversight assurance",
        "description": "How qualified clinicians oversee and can override AI recommendations",
        "required": True,
        "regulatory_basis": "SACHRP Recommendations, MRCT Framework",
        "template_text": (
            "A qualified physician will review all AI-generated recommendations before they "
            "are used in your care. Your treatment team can override or disregard AI "
            "recommendations at any time based on their clinical judgment."
        )
    },
    {
        "element": "Data use for AI training",
        "description": "Whether participant data may be used to train or improve AI models",
        "required": True,
        "regulatory_basis": "45 CFR 46.116(c)(7), SACHRP Recommendations",
        "template_text": (
            "Your de-identified data may be used to improve the AI system's performance. "
            "This data will be stripped of all identifying information before use. "
            "You may choose not to allow use of your data for AI improvement without "
            "affecting your participation in the trial."
        )
    },
    {
        "element": "AI-specific privacy risks",
        "description": "Disclosure of re-identification risks unique to AI processing of medical data",
        "required": True,
        "regulatory_basis": "SACHRP Recommendations",
        "template_text": (
            "Advanced AI systems can sometimes identify patterns in medical data that could "
            "potentially link back to individuals, even from de-identified data. We take "
            "additional precautions to protect against this risk, including [describe safeguards]."
        )
    },
    {
        "element": "Algorithmic bias disclosure",
        "description": "Statement about potential for AI bias and mitigation measures",
        "required": True,
        "regulatory_basis": "MRCT Framework, SACHRP Recommendations",
        "template_text": (
            "AI systems may perform differently across patient populations. We have tested "
            "this system across diverse populations and [describe bias assessment results]. "
            "Your care team is aware of these considerations."
        )
    },
    {
        "element": "Right to non-AI alternative",
        "description": "Whether participants can receive treatment without AI involvement",
        "required": False,
        "regulatory_basis": "45 CFR 46.116(b)(4)",
        "template_text": (
            "You may request that AI technology not be used in your treatment planning. "
            "Standard clinical approaches will be available as alternatives."
        )
    }
]

# AI-specific IRB review checklist items per MRCT Framework
AI_REVIEW_CHECKLIST: list[dict[str, Any]] = [
    {
        "category": "ai_transparency",
        "item": "AI involvement clearly described in protocol and consent",
        "reference": "SACHRP Recommendations"
    },
    {
        "category": "ai_transparency",
        "item": "AI model development stage documented (research, validated, cleared/approved)",
        "reference": "MRCT Framework"
    },
    {
        "category": "ai_transparency",
        "item": "Regulatory status of AI components specified (investigational, cleared, none)",
        "reference": "MRCT Framework"
    },
    {
        "category": "algorithmic_bias",
        "item": "Bias assessment conducted across demographic subgroups",
        "reference": "SACHRP, FDA AI/ML Guidance"
    },
    {
        "category": "algorithmic_bias",
        "item": "Bias mitigation strategy documented",
        "reference": "MRCT Framework"
    },
    {
        "category": "algorithmic_bias",
        "item": "Training data demographic representation described",
        "reference": "FDA AI/ML Guidance (Jan 2025)"
    },
    {
        "category": "privacy_data",
        "item": "AI-specific re-identification risks assessed",
        "reference": "SACHRP Recommendations"
    },
    {
        "category": "privacy_data",
        "item": "Data access by AI systems limited per minimum necessary principle",
        "reference": "45 CFR 164.502(b)"
    },
    {
        "category": "privacy_data",
        "item": "AI system access to demographic data limited where possible",
        "reference": "SACHRP Recommendations"
    },
    {
        "category": "human_oversight",
        "item": "Human oversight mechanism described for all AI-influenced decisions",
        "reference": "MRCT Framework"
    },
    {
        "category": "human_oversight",
        "item": "Override procedures for AI recommendations specified",
        "reference": "ICH E6(R3)"
    },
    {
        "category": "risk_benefit",
        "item": "AI-specific risks and benefits analyzed in protocol",
        "reference": "45 CFR 46.111(a)(2)"
    },
    {
        "category": "risk_benefit",
        "item": "Data Safety Monitoring Board (DSMB) plan includes AI performance review",
        "reference": "ICH E6(R3)"
    },
    {
        "category": "consent",
        "item": "Informed consent describes AI involvement in plain language",
        "reference": "45 CFR 46.116"
    },
    {
        "category": "consent",
        "item": "Consent addresses secondary use of data for AI training",
        "reference": "45 CFR 46.116(c)(7)"
    },
    {
        "category": "monitoring",
        "item": "AI performance monitoring plan during trial execution",
        "reference": "ICH E6(R3)"
    },
    {
        "category": "monitoring",
        "item": "Stopping rules defined for AI performance degradation",
        "reference": "MRCT Framework"
    }
]


# =============================================================================
# SECTION 3: IRB PROTOCOL MANAGER
# =============================================================================

class IRBProtocolManager:
    """
    Manages IRB protocol submissions for AI-enabled oncology trials.

    Incorporates AI-specific review requirements from HHS SACHRP
    recommendations and the MRCT Center Framework (July 2025).

    KEY FUNCTIONS:
    -------------
    1. Protocol creation with AI component documentation
    2. AI-specific informed consent template generation
    3. IRB review checklist management
    4. Amendment tracking for AI model updates
    5. Continuing review management
    """

    def __init__(
        self,
        institution: str,
        irb_type: str = "central",
        fwa_number: str = ""
    ):
        """
        Initialize IRB protocol manager.

        Args:
            institution: Institution name
            irb_type: "central", "local", or "commercial"
            fwa_number: Federalwide Assurance number
        """
        self.institution = institution
        self.irb_type = irb_type
        self.fwa_number = fwa_number

        self._protocols: dict[str, Protocol] = {}
        self._protocol_counter = 0

        logger.info(
            f"IRBProtocolManager initialized: institution={institution}, "
            f"irb_type={irb_type}"
        )

    def create_protocol(
        self,
        title: str,
        pi_name: str,
        ai_components: list[str],
        participant_count: int = 0,
        trial_phase: str = "pivotal",
        risk_category: str = "greater_than_minimal_risk"
    ) -> Protocol:
        """
        Create a new IRB protocol submission.

        Args:
            title: Protocol title
            pi_name: Principal Investigator name
            ai_components: List of AI component descriptions
            participant_count: Planned enrollment
            trial_phase: Trial phase (pilot, pivotal, post-market)
            risk_category: "minimal_risk" or "greater_than_minimal_risk"

        Returns:
            Protocol record
        """
        self._protocol_counter += 1
        protocol_id = f"IRB-{datetime.now().strftime('%Y%m%d')}-{self._protocol_counter:04d}"

        # Create AI component records
        components = [
            AIComponent(
                name=comp,
                function="decision_support",
                human_override=True
            )
            for comp in ai_components
        ]

        # Generate consent elements
        consent = [
            ConsentElement(
                element=elem["element"],
                description=elem["description"],
                required=elem["required"],
                ai_specific=True,
                regulatory_basis=elem["regulatory_basis"],
                template_text=elem["template_text"]
            )
            for elem in AI_CONSENT_ELEMENTS
        ]

        protocol = Protocol(
            protocol_id=protocol_id,
            title=title,
            pi_name=pi_name,
            institution=self.institution,
            review_type=ReviewType.FULL_BOARD,
            risk_category=RiskCategory(risk_category),
            ai_components=components,
            participant_count=participant_count,
            trial_phase=trial_phase,
            consent_elements=consent,
            created_date=datetime.now().isoformat()
        )

        self._protocols[protocol_id] = protocol

        logger.info(
            f"Protocol created: {protocol_id} - {title} "
            f"({len(components)} AI components)"
        )

        return protocol

    def generate_ai_review_checklist(
        self,
        protocol: Protocol
    ) -> SubmissionChecklist:
        """
        Generate AI-specific IRB review checklist.

        Creates a checklist based on SACHRP recommendations and
        MRCT Center Framework requirements.

        Args:
            protocol: Protocol to generate checklist for

        Returns:
            SubmissionChecklist with AI-specific items
        """
        checklist = SubmissionChecklist(
            submission_id=protocol.protocol_id,
            submission_type="irb_ai_review"
        )

        for item_def in AI_REVIEW_CHECKLIST:
            checklist.items.append(ReviewChecklistItem(
                category=item_def["category"],
                item=item_def["item"],
                regulatory_reference=item_def["reference"],
                ai_specific=True
            ))

        logger.info(
            f"AI review checklist generated for {protocol.protocol_id}: "
            f"{len(checklist.items)} items"
        )

        return checklist

    def generate_consent_template(self, protocol: Protocol) -> str:
        """
        Generate informed consent template with AI-specific elements.

        Produces a template incorporating all required and recommended
        AI disclosure elements per SACHRP and MRCT guidance.

        Args:
            protocol: Protocol to generate consent for

        Returns:
            Consent template text
        """
        template = f"""
INFORMED CONSENT DOCUMENT
==========================
Protocol: {protocol.title}
Protocol ID: {protocol.protocol_id}
Principal Investigator: {protocol.pi_name}
Institution: {protocol.institution}

NOTE: This is a template. All sections must be reviewed and
customized by the PI and IRB before use.

---

INTRODUCTION

You are being invited to participate in a research study. This document
describes the study, including its purpose, what you will be asked to do,
and the risks and benefits of participating.

---

AI TECHNOLOGY IN THIS STUDY
"""

        for element in protocol.consent_elements:
            if element.ai_specific:
                template += f"\n{element.element.upper()}\n"
                template += f"{element.template_text}\n"

        template += f"""
---

AI COMPONENTS USED IN THIS STUDY

The following AI technologies are part of this research:
"""

        for comp in protocol.ai_components:
            template += f"\n- {comp.name}: {comp.function}"
            if comp.human_override:
                template += " (all outputs reviewed by physician)"
            template += "\n"

        template += """
---

YOUR RIGHTS

- You can choose not to participate in this study.
- You can withdraw at any time without penalty.
- You can request that AI technology not be used in your care.
- Your decision will not affect the quality of your medical care.

---

SIGNATURES

Participant: _________________________ Date: ___________
PI or Designee: ______________________ Date: ___________
"""

        return template

    def file_amendment(
        self,
        protocol_id: str,
        amendment_type: str,
        description: str,
        ai_model_changed: bool = False
    ):
        """
        File a protocol amendment.

        Args:
            protocol_id: Protocol to amend
            amendment_type: "administrative", "minor", "major"
            description: Description of changes
            ai_model_changed: Whether AI model was updated
        """
        protocol = self._protocols.get(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")

        amendment = {
            "date": datetime.now().isoformat(),
            "type": amendment_type,
            "description": description,
            "ai_model_changed": ai_model_changed,
            "status": "submitted"
        }

        protocol.amendments.append(amendment)

        logger.info(
            f"Amendment filed for {protocol_id}: {amendment_type} "
            f"(AI model changed: {ai_model_changed})"
        )


# Reuse SubmissionChecklist from fda_submission_tracker for consistency
@dataclass
class SubmissionChecklist:
    """Checklist for IRB submission."""
    submission_id: str
    submission_type: str
    items: list = field(default_factory=list)
    completion_percentage: float = 0.0


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_irb_manager_demo():
    """
    Demonstrate IRB protocol management for AI oncology trials.
    """
    logger.info("=" * 60)
    logger.info("IRB PROTOCOL MANAGER DEMO")
    logger.info("=" * 60)

    irb = IRBProtocolManager(
        institution="Memorial Sloan Kettering Cancer Center",
        irb_type="central"
    )

    # Create protocol
    protocol = irb.create_protocol(
        title="Physical AI-Guided Surgical Resection for NSCLC: A Pivotal Trial",
        pi_name="Dr. Sarah Chen",
        ai_components=[
            "Real-time tumor boundary detection via intraoperative CT",
            "Robotic instrument path guidance for tumor resection",
            "Digital twin treatment response prediction"
        ],
        participant_count=200,
        trial_phase="pivotal"
    )

    print(f"\nProtocol: {protocol.protocol_id}")
    print(f"Title: {protocol.title}")
    print(f"PI: {protocol.pi_name}")
    print(f"AI Components: {len(protocol.ai_components)}")
    print(f"Consent Elements: {len(protocol.consent_elements)}")

    # Generate AI review checklist
    checklist = irb.generate_ai_review_checklist(protocol)
    print(f"\nAI Review Checklist: {len(checklist.items)} items")

    categories: dict[str, int] = {}
    for item in checklist.items:
        categories[item.category] = categories.get(item.category, 0) + 1

    for category, count in categories.items():
        print(f"  {category}: {count} items")

    # Show consent template excerpt
    consent = irb.generate_consent_template(protocol)
    lines = consent.strip().split("\n")
    print(f"\nConsent Template ({len(lines)} lines):")
    for line in lines[:10]:
        print(f"  {line}")
    print("  ...")

    return {"protocol_id": protocol.protocol_id, "status": "demo_complete"}


if __name__ == "__main__":
    result = run_irb_manager_demo()
    print(f"\nDemo completed: {result}")
