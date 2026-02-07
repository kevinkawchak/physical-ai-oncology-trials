"""
=============================================================================
EXAMPLE 06: Protocol-Grounded RAG Compliance Agent
=============================================================================

Implements a Retrieval-Augmented Generation (RAG) agent that grounds
clinical decisions in authoritative protocol documents, FDA guidance,
and institutional procedures to ensure regulatory compliance during
robotic oncology procedures.

CLINICAL CONTEXT:
-----------------
Robotic oncology trials must comply with multiple regulatory frameworks:
  - Trial-specific protocol (inclusion/exclusion, treatment arms, endpoints)
  - FDA guidance documents (IDE, 510(k), De Novo for surgical robots)
  - ICH E6(R3) Good Clinical Practice
  - IEC 62304 (medical device software lifecycle)
  - IEC 80601-2-77 (robotically assisted surgical equipment)
  - Institutional SOPs and IRB-approved procedures

A RAG agent retrieves relevant sections from these documents to:
  - Verify compliance of proposed clinical actions
  - Provide cited rationale for treatment decisions
  - Flag deviations from protocol before they occur
  - Support real-time regulatory queries during procedures

DISTINCTION FROM EXISTING EXAMPLES:
------------------------------------
- examples/04_agentic_clinical_workflow.py: Multi-agent orchestration
  for trial management (enrollment, AEs, data quality)
- This example: Document-grounded decision support with retrieval
  from regulatory knowledge bases, citation tracking, and compliance
  verification against specific protocol sections

ARCHITECTURE:
-------------
  Query  -->  Document     -->  Relevant    -->  LLM      -->  Cited
              Retriever        Chunks           Reasoning      Response
                  |                                |
             Regulatory                     Compliance
             Knowledge                      Verification
             Base (chunked,
              embedded)

FRAMEWORK REQUIREMENTS:
-----------------------
Required: (none - pure Python implementation)

Optional:
    - anthropic >= 0.40.0 (LLM reasoning)
    - langchain >= 1.0.0 (RAG pipeline)
    - numpy >= 1.24.0 (embedding similarity)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DOCUMENT AND KNOWLEDGE BASE MODELS
# =============================================================================


class DocumentType(Enum):
    """Types of regulatory and protocol documents."""

    TRIAL_PROTOCOL = "trial_protocol"
    FDA_GUIDANCE = "fda_guidance"
    ICH_GUIDELINE = "ich_guideline"
    IEC_STANDARD = "iec_standard"
    ISO_STANDARD = "iso_standard"
    INSTITUTIONAL_SOP = "institutional_sop"
    IRB_APPROVAL = "irb_approval"
    INFORMED_CONSENT = "informed_consent"


class ComplianceStatus(Enum):
    """Compliance assessment status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partially_compliant"
    REQUIRES_REVIEW = "requires_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class DocumentChunk:
    """A chunk of a regulatory document for retrieval."""

    chunk_id: str
    document_id: str
    document_type: DocumentType
    document_title: str
    section_number: str
    section_title: str
    content: str
    page_number: int = 0
    effective_date: str = ""
    keywords: list = field(default_factory=list)

    def get_citation(self) -> str:
        """Generate a formal citation string."""
        return f"{self.document_title}, Section {self.section_number}: {self.section_title}"


@dataclass
class RetrievalResult:
    """Result from document retrieval."""

    chunk: DocumentChunk
    relevance_score: float
    matched_keywords: list = field(default_factory=list)


@dataclass
class ComplianceCheck:
    """Result of checking an action against a specific requirement."""

    requirement_id: str
    requirement_text: str
    source_citation: str
    status: ComplianceStatus
    rationale: str
    evidence: str = ""
    remediation: str = ""


@dataclass
class ComplianceReport:
    """Complete compliance assessment report."""

    query: str
    timestamp: float
    overall_status: ComplianceStatus
    checks: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    retrieved_sections: int = 0


# =============================================================================
# SECTION 2: REGULATORY KNOWLEDGE BASE
# =============================================================================


class RegulatoryKnowledgeBase:
    """
    In-memory knowledge base of regulatory documents chunked for retrieval.

    In production, this would use a vector database (Pinecone, Weaviate,
    Chroma) with embeddings from a medical/regulatory-specific model.
    This implementation uses keyword-based retrieval to demonstrate
    the RAG pipeline without external dependencies.

    DOCUMENT SOURCES:
    -----------------
    The knowledge base contains curated chunks from:
    1. Trial protocol (eligibility, treatment, endpoints, safety)
    2. FDA guidance for robotic surgical devices
    3. ICH E6(R3) GCP requirements
    4. IEC 80601-2-77 robotic surgery safety
    5. Institutional SOPs
    """

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._keyword_index: dict[str, list[int]] = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load regulatory document chunks."""
        self._add_trial_protocol_chunks()
        self._add_fda_guidance_chunks()
        self._add_ich_gcp_chunks()
        self._add_iec_robotic_chunks()
        self._add_institutional_sop_chunks()
        self._build_keyword_index()

        logger.info(f"Knowledge base loaded: {len(self._chunks)} chunks from {self._count_documents()} documents")

    def _add_trial_protocol_chunks(self) -> None:
        """Add trial protocol document chunks."""
        doc_id = "PROT-2026-001"
        doc_title = "Protocol: AI-Guided Robotic Tumor Resection (Phase II)"

        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}-001",
                document_id=doc_id,
                document_type=DocumentType.TRIAL_PROTOCOL,
                document_title=doc_title,
                section_number="3.1",
                section_title="Inclusion Criteria",
                content=(
                    "Eligible patients must meet ALL of the following criteria: "
                    "(a) Age >= 18 years; (b) Histologically confirmed solid tumor amenable "
                    "to robotic resection; (c) ECOG performance status 0-1; (d) Adequate organ "
                    "function defined as: ANC >= 1500/uL, platelets >= 100,000/uL, hemoglobin "
                    ">= 9 g/dL, creatinine <= 1.5x ULN, total bilirubin <= 1.5x ULN, AST/ALT "
                    "<= 3x ULN; (e) Able to undergo general anesthesia; (f) Signed informed "
                    "consent."
                ),
                keywords=["eligibility", "inclusion", "criteria", "ECOG", "organ function", "enrollment"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-002",
                document_id=doc_id,
                document_type=DocumentType.TRIAL_PROTOCOL,
                document_title=doc_title,
                section_number="3.2",
                section_title="Exclusion Criteria",
                content=(
                    "Patients with ANY of the following are excluded: (a) Prior robotic surgery "
                    "within 30 days; (b) Active systemic infection requiring IV antibiotics; "
                    "(c) Tumor invading major vascular structures (T4 by imaging); (d) "
                    "Uncontrolled coagulopathy (INR > 2.0 or aPTT > 1.5x ULN); (e) Pregnancy "
                    "or lactation; (f) Contraindication to general anesthesia; (g) Inability "
                    "to comply with follow-up schedule."
                ),
                keywords=["exclusion", "criteria", "contraindication", "coagulopathy", "infection"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-003",
                document_id=doc_id,
                document_type=DocumentType.TRIAL_PROTOCOL,
                document_title=doc_title,
                section_number="5.1",
                section_title="Primary Endpoint",
                content=(
                    "The primary endpoint is R0 resection rate, defined as the proportion "
                    "of patients achieving microscopically negative margins on final "
                    "pathology. R0 is defined as no tumor cells within 1mm of any inked "
                    "margin per AJCC/CAP guidelines. The target R0 rate for the AI-guided "
                    "arm is >= 95%, compared to historical control of 85%."
                ),
                keywords=["endpoint", "R0", "margin", "resection", "pathology", "primary"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-004",
                document_id=doc_id,
                document_type=DocumentType.TRIAL_PROTOCOL,
                document_title=doc_title,
                section_number="6.3",
                section_title="Intraoperative Safety Stopping Rules",
                content=(
                    "The procedure must be paused or converted to standard technique if any "
                    "of the following occur: (a) Robot system fault or communication failure "
                    "lasting > 30 seconds; (b) Force sensor readings exceeding 15N sustained "
                    "for > 5 seconds; (c) Patient hemodynamic instability (MAP < 60 mmHg "
                    "for > 2 minutes despite intervention); (d) Uncontrolled bleeding > "
                    "500mL estimated blood loss; (e) AI planning system disagreement with "
                    "surgical team assessment of critical structure location."
                ),
                keywords=["safety", "stopping", "conversion", "force", "hemodynamic", "bleeding", "intraoperative"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-005",
                document_id=doc_id,
                document_type=DocumentType.TRIAL_PROTOCOL,
                document_title=doc_title,
                section_number="7.2",
                section_title="Adverse Event Reporting Requirements",
                content=(
                    "All adverse events must be graded per CTCAE v5.0. Serious adverse events "
                    "(SAEs) must be reported to the sponsor within 24 hours of awareness. "
                    "Device-related SAEs require concurrent MedWatch 3500A reporting to FDA. "
                    "Unanticipated adverse device effects (UADEs) require IRB notification "
                    "within 10 working days. Deaths must be reported within 24 hours "
                    "regardless of causality assessment."
                ),
                keywords=["adverse event", "SAE", "reporting", "CTCAE", "MedWatch", "IRB", "FDA"],
            ),
        ]
        self._chunks.extend(chunks)

    def _add_fda_guidance_chunks(self) -> None:
        """Add FDA guidance document chunks."""
        doc_id = "FDA-GUID-001"
        doc_title = "FDA Guidance: Computer-Assisted Surgical Equipment"

        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}-001",
                document_id=doc_id,
                document_type=DocumentType.FDA_GUIDANCE,
                document_title=doc_title,
                section_number="4.2",
                section_title="Software Validation Requirements",
                content=(
                    "The software controlling robotic surgical motion must undergo validation "
                    "testing per IEC 62304 classification. Class C software (can result in "
                    "death or serious injury) requires: (a) Complete requirements traceability; "
                    "(b) Software architecture documentation; (c) Unit testing with >= 80% "
                    "code coverage; (d) Integration testing across all hardware interfaces; "
                    "(e) System-level validation in simulated and cadaveric environments."
                ),
                keywords=["software", "validation", "IEC 62304", "testing", "Class C", "robot"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-002",
                document_id=doc_id,
                document_type=DocumentType.FDA_GUIDANCE,
                document_title=doc_title,
                section_number="5.1",
                section_title="AI/ML-Based Surgical Planning",
                content=(
                    "When AI/ML algorithms are used for surgical planning or intraoperative "
                    "guidance, the following apply: (a) The AI output must be presented as "
                    "advisory, not autonomous; (b) The surgeon must have ability to override "
                    "any AI recommendation; (c) Algorithm performance must be validated on "
                    "a representative patient population; (d) Known failure modes and "
                    "limitations must be disclosed to the surgical team; (e) Real-time "
                    "confidence indicators must be displayed for AI outputs."
                ),
                keywords=["AI", "ML", "surgical planning", "advisory", "override", "confidence", "autonomous"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-003",
                document_id=doc_id,
                document_type=DocumentType.FDA_GUIDANCE,
                document_title=doc_title,
                section_number="6.3",
                section_title="Cybersecurity for Networked Surgical Systems",
                content=(
                    "Networked surgical robot systems must implement: (a) Encrypted "
                    "communication channels (TLS 1.3 minimum); (b) Authentication for "
                    "all network interfaces; (c) Audit logging of all remote commands; "
                    "(d) Network segmentation isolating surgical control from hospital IT; "
                    "(e) Fail-safe behavior upon network disconnection maintaining safe "
                    "robot state."
                ),
                keywords=["cybersecurity", "network", "encryption", "authentication", "audit", "segmentation"],
            ),
        ]
        self._chunks.extend(chunks)

    def _add_ich_gcp_chunks(self) -> None:
        """Add ICH GCP guideline chunks."""
        doc_id = "ICH-E6R3"
        doc_title = "ICH E6(R3) Good Clinical Practice"

        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}-001",
                document_id=doc_id,
                document_type=DocumentType.ICH_GUIDELINE,
                document_title=doc_title,
                section_number="4.8",
                section_title="Informed Consent",
                content=(
                    "Before a subject's participation in the trial, the investigator should "
                    "obtain the subject's freely given informed consent after adequate "
                    "explanation of the aims, methods, anticipated benefits, potential hazards, "
                    "and the right to withdraw at any time. For robotic surgical trials, "
                    "specific risks related to robotic equipment malfunction, AI-guided "
                    "decision support limitations, and potential conversion to standard "
                    "technique must be disclosed."
                ),
                keywords=["informed consent", "disclosure", "risks", "withdrawal", "robotic", "participation"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-002",
                document_id=doc_id,
                document_type=DocumentType.ICH_GUIDELINE,
                document_title=doc_title,
                section_number="5.18",
                section_title="Monitoring",
                content=(
                    "The sponsor should ensure that trials are adequately monitored. "
                    "Risk-based monitoring should focus on: (a) Critical data and processes; "
                    "(b) Key risk indicators; (c) Protocol deviations; (d) Data integrity. "
                    "For robotic surgical trials, monitoring should include review of "
                    "device performance logs, AI decision logs, and adverse event patterns "
                    "across sites."
                ),
                keywords=["monitoring", "risk-based", "data integrity", "protocol deviation", "sponsor"],
            ),
        ]
        self._chunks.extend(chunks)

    def _add_iec_robotic_chunks(self) -> None:
        """Add IEC 80601-2-77 chunks for robotic surgery."""
        doc_id = "IEC-80601-2-77"
        doc_title = "IEC 80601-2-77: Robotically Assisted Surgical Equipment"

        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}-001",
                document_id=doc_id,
                document_type=DocumentType.IEC_STANDARD,
                document_title=doc_title,
                section_number="201.12.4.4",
                section_title="Restricted Motion Range",
                content=(
                    "The RASE (Robotically Assisted Surgical Equipment) must implement "
                    "restricted motion range limits that prevent the robotic actuator from "
                    "moving beyond the defined workspace boundaries. The restricted motion "
                    "range must be configurable per procedure type and patient anatomy. "
                    "The system must halt motion before reaching any workspace boundary "
                    "with a margin defined by the risk analysis."
                ),
                keywords=["workspace", "motion range", "boundary", "RASE", "limit", "safety"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-002",
                document_id=doc_id,
                document_type=DocumentType.IEC_STANDARD,
                document_title=doc_title,
                section_number="201.12.4.6",
                section_title="Force Limiting",
                content=(
                    "The RASE must implement force limiting to prevent excessive force "
                    "application to tissue. Force limits must be: (a) Set based on the "
                    "specific surgical task and tissue type; (b) Continuously monitored "
                    "at a rate no less than 1 kHz; (c) Enforced through immediate motion "
                    "cessation when exceeded; (d) Independently verified by a redundant "
                    "force sensing system for critical operations."
                ),
                keywords=["force", "limit", "tissue", "monitoring", "redundant", "cessation"],
            ),
        ]
        self._chunks.extend(chunks)

    def _add_institutional_sop_chunks(self) -> None:
        """Add institutional SOP chunks."""
        doc_id = "SOP-SURG-001"
        doc_title = "SOP: Robotic Surgery Setup and Verification"

        chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}-001",
                document_id=doc_id,
                document_type=DocumentType.INSTITUTIONAL_SOP,
                document_title=doc_title,
                section_number="4.1",
                section_title="Pre-Procedure Robot Verification",
                content=(
                    "Before each robotic procedure, the following verification steps "
                    "must be completed and documented: (1) Robot power-on self-test (POST) "
                    "must pass all checks; (2) Joint calibration verification within "
                    "0.1 degrees; (3) Force sensor zero calibration; (4) Communication "
                    "link verification between console and patient-side cart; (5) Emergency "
                    "stop function test; (6) Instrument insertion and recognition test; "
                    "(7) Endoscope white balance and focus verification."
                ),
                keywords=["pre-procedure", "verification", "calibration", "self-test", "checklist", "setup"],
            ),
            DocumentChunk(
                chunk_id=f"{doc_id}-002",
                document_id=doc_id,
                document_type=DocumentType.INSTITUTIONAL_SOP,
                document_title=doc_title,
                section_number="5.3",
                section_title="Specimen Handling During Robotic Procedures",
                content=(
                    "All specimens obtained during robotic procedures must be: (a) Placed "
                    "in specimen bags immediately upon removal; (b) Labeled with patient "
                    "MRN and specimen type at the surgical field; (c) Transferred to "
                    "pathology with chain-of-custody documentation; (d) Orientation "
                    "marked per surgeon instruction for margin assessment; (e) Fresh "
                    "tissue sent for frozen section if margin status is uncertain."
                ),
                keywords=[
                    "specimen",
                    "handling",
                    "pathology",
                    "margin",
                    "labeling",
                    "chain of custody",
                    "frozen section",
                ],
            ),
        ]
        self._chunks.extend(chunks)

    def _build_keyword_index(self) -> None:
        """Build keyword-to-chunk index for retrieval."""
        for idx, chunk in enumerate(self._chunks):
            for keyword in chunk.keywords:
                kw_lower = keyword.lower()
                if kw_lower not in self._keyword_index:
                    self._keyword_index[kw_lower] = []
                self._keyword_index[kw_lower].append(idx)

            # Also index words from section title
            for word in chunk.section_title.lower().split():
                if len(word) > 3:
                    if word not in self._keyword_index:
                        self._keyword_index[word] = []
                    self._keyword_index[word].append(idx)

    def _count_documents(self) -> int:
        """Count unique documents in knowledge base."""
        return len(set(c.document_id for c in self._chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Retrieve relevant document chunks for a query.

        Uses keyword matching with TF-IDF-style scoring.
        In production, replace with vector similarity search.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            Ranked list of relevant document chunks
        """
        query_words = set(re.findall(r"\b\w+\b", query.lower()))

        # Score each chunk by keyword overlap
        chunk_scores: dict[int, float] = {}
        matched_keywords_map: dict[int, list[str]] = {}

        for word in query_words:
            # Direct keyword match
            if word in self._keyword_index:
                for idx in self._keyword_index[word]:
                    chunk_scores[idx] = chunk_scores.get(idx, 0) + 2.0
                    if idx not in matched_keywords_map:
                        matched_keywords_map[idx] = []
                    matched_keywords_map[idx].append(word)

            # Partial match on keywords
            for kw, indices in self._keyword_index.items():
                if word in kw or kw in word:
                    for idx in indices:
                        chunk_scores[idx] = chunk_scores.get(idx, 0) + 1.0
                        if idx not in matched_keywords_map:
                            matched_keywords_map[idx] = []
                        if word not in matched_keywords_map[idx]:
                            matched_keywords_map[idx].append(word)

        # Content match (check for query words in content)
        for idx, chunk in enumerate(self._chunks):
            content_lower = chunk.content.lower()
            content_matches = sum(1 for w in query_words if w in content_lower and len(w) > 3)
            if content_matches > 0:
                chunk_scores[idx] = chunk_scores.get(idx, 0) + content_matches * 0.5

        # Sort by score and return top-k
        sorted_indices = sorted(chunk_scores.keys(), key=lambda i: chunk_scores[i], reverse=True)

        results = []
        for idx in sorted_indices[:top_k]:
            max_possible = len(query_words) * 3.0
            score = min(1.0, chunk_scores[idx] / max(1.0, max_possible))
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    relevance_score=round(score, 3),
                    matched_keywords=matched_keywords_map.get(idx, []),
                )
            )

        return results


# =============================================================================
# SECTION 3: COMPLIANCE VERIFICATION ENGINE
# =============================================================================


class ComplianceVerifier:
    """
    Verifies compliance of clinical actions against retrieved
    regulatory requirements.

    Uses rule-based checks with retrieved document context to
    assess whether proposed actions comply with applicable
    regulations and protocols.
    """

    def __init__(self):
        self._check_history: list[ComplianceCheck] = []

    def verify_action_compliance(
        self,
        action_description: str,
        action_context: dict,
        retrieved_chunks: list[RetrievalResult],
    ) -> list[ComplianceCheck]:
        """
        Verify an action against retrieved regulatory requirements.

        Args:
            action_description: Description of the proposed action
            action_context: Context including patient data, procedure state
            retrieved_chunks: Relevant regulatory document chunks

        Returns:
            List of compliance check results
        """
        checks = []

        for result in retrieved_chunks:
            chunk = result.chunk

            # Determine applicable compliance checks based on document type
            if chunk.document_type == DocumentType.TRIAL_PROTOCOL:
                check = self._check_protocol_compliance(action_description, action_context, chunk)
            elif chunk.document_type == DocumentType.FDA_GUIDANCE:
                check = self._check_fda_compliance(action_description, action_context, chunk)
            elif chunk.document_type == DocumentType.IEC_STANDARD:
                check = self._check_iec_compliance(action_description, action_context, chunk)
            elif chunk.document_type == DocumentType.ICH_GUIDELINE:
                check = self._check_gcp_compliance(action_description, action_context, chunk)
            elif chunk.document_type == DocumentType.INSTITUTIONAL_SOP:
                check = self._check_sop_compliance(action_description, action_context, chunk)
            else:
                check = ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text=chunk.content[:200],
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.REQUIRES_REVIEW,
                    rationale="Manual review required for this document type",
                )

            if check:
                checks.append(check)
                self._check_history.append(check)

        return checks

    def _check_protocol_compliance(self, action: str, context: dict, chunk: DocumentChunk) -> ComplianceCheck:
        """Check compliance against trial protocol requirements."""
        action_lower = action.lower()
        content_lower = chunk.content.lower()

        # Eligibility-related checks
        if "inclusion" in chunk.section_title.lower() or "exclusion" in chunk.section_title.lower():
            if "enroll" in action_lower or "eligib" in action_lower:
                ecog = context.get("ecog_status")
                if ecog is not None and ecog > 1:
                    return ComplianceCheck(
                        requirement_id=chunk.chunk_id,
                        requirement_text="ECOG performance status 0-1 required",
                        source_citation=chunk.get_citation(),
                        status=ComplianceStatus.NON_COMPLIANT,
                        rationale=f"Patient ECOG status is {ecog}, protocol requires 0-1",
                        remediation="Patient does not meet eligibility. Do not enroll.",
                    )
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text=chunk.content[:200],
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.REQUIRES_REVIEW,
                    rationale="Verify all inclusion/exclusion criteria before enrollment",
                )

        # Safety stopping rule checks
        if "stopping" in chunk.section_title.lower() or "safety" in chunk.section_title.lower():
            force = context.get("current_force_n", 0)
            if force > 15.0:
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="Force > 15N for > 5s triggers procedure pause",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale=f"Current force {force:.1f}N exceeds 15N protocol limit",
                    remediation="Pause procedure per protocol Section 6.3",
                )

            ebl = context.get("estimated_blood_loss_ml", 0)
            if ebl > 500:
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="EBL > 500mL triggers procedure conversion",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale=f"Estimated blood loss {ebl}mL exceeds 500mL protocol limit",
                    remediation="Convert to standard technique per protocol Section 6.3",
                )

        # AE reporting checks
        if "adverse" in content_lower and "report" in content_lower:
            if "adverse event" in action_lower or "ae" in action_lower:
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="SAEs must be reported within 24 hours",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.REQUIRES_REVIEW,
                    rationale="Verify SAE reporting timeline compliance",
                    evidence="CTCAE v5.0 grading required, MedWatch 3500A for device-related SAEs",
                )

        return ComplianceCheck(
            requirement_id=chunk.chunk_id,
            requirement_text=chunk.content[:200],
            source_citation=chunk.get_citation(),
            status=ComplianceStatus.COMPLIANT,
            rationale="No compliance issues identified for this action",
        )

    def _check_fda_compliance(self, action: str, context: dict, chunk: DocumentChunk) -> ComplianceCheck:
        """Check compliance against FDA guidance."""
        action_lower = action.lower()

        # AI advisory requirement
        if "ai" in chunk.content.lower() and "advisory" in chunk.content.lower():
            if "autonomous" in action_lower and not context.get("surgeon_override_available", True):
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="AI output must be advisory, not autonomous",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale="Action appears autonomous without surgeon override capability",
                    remediation="Ensure surgeon can override AI recommendation at all times",
                )

        return ComplianceCheck(
            requirement_id=chunk.chunk_id,
            requirement_text=chunk.content[:200],
            source_citation=chunk.get_citation(),
            status=ComplianceStatus.COMPLIANT,
            rationale="Action aligns with FDA guidance requirements",
        )

    def _check_iec_compliance(self, action: str, context: dict, chunk: DocumentChunk) -> ComplianceCheck:
        """Check compliance against IEC 80601-2-77."""
        # Force limit compliance
        if "force" in chunk.section_title.lower():
            force = context.get("current_force_n", 0)
            max_force = context.get("max_force_n", 10.0)
            if force > max_force:
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="Force must not exceed configured limit",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale=f"Current force {force:.1f}N exceeds limit {max_force:.1f}N",
                    remediation="Reduce applied force immediately per IEC 80601-2-77",
                )

        # Workspace compliance
        if "motion range" in chunk.section_title.lower():
            if not context.get("within_workspace", True):
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="Robot must operate within restricted motion range",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale="Robot target is outside defined workspace boundaries",
                    remediation="Reconfigure workspace or adjust target position",
                )

        return ComplianceCheck(
            requirement_id=chunk.chunk_id,
            requirement_text=chunk.content[:200],
            source_citation=chunk.get_citation(),
            status=ComplianceStatus.COMPLIANT,
            rationale="Action complies with IEC requirements",
        )

    def _check_gcp_compliance(self, action: str, context: dict, chunk: DocumentChunk) -> ComplianceCheck:
        """Check compliance against ICH GCP."""
        if "consent" in chunk.section_title.lower():
            if not context.get("informed_consent_signed", True):
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="Informed consent required before participation",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale="Informed consent not documented as signed",
                    remediation="Obtain and document informed consent before proceeding",
                )

        return ComplianceCheck(
            requirement_id=chunk.chunk_id,
            requirement_text=chunk.content[:200],
            source_citation=chunk.get_citation(),
            status=ComplianceStatus.COMPLIANT,
            rationale="Action complies with ICH GCP requirements",
        )

    def _check_sop_compliance(self, action: str, context: dict, chunk: DocumentChunk) -> ComplianceCheck:
        """Check compliance against institutional SOPs."""
        if "verification" in chunk.section_title.lower() or "pre-procedure" in chunk.section_title.lower():
            if not context.get("robot_self_test_passed", True):
                return ComplianceCheck(
                    requirement_id=chunk.chunk_id,
                    requirement_text="Robot POST must pass before procedure",
                    source_citation=chunk.get_citation(),
                    status=ComplianceStatus.NON_COMPLIANT,
                    rationale="Robot power-on self-test has not passed",
                    remediation="Complete robot self-test and verify all checks pass",
                )

        return ComplianceCheck(
            requirement_id=chunk.chunk_id,
            requirement_text=chunk.content[:200],
            source_citation=chunk.get_citation(),
            status=ComplianceStatus.COMPLIANT,
            rationale="Action complies with institutional SOP",
        )


# =============================================================================
# SECTION 4: RAG COMPLIANCE AGENT
# =============================================================================


class ProtocolRAGComplianceAgent:
    """
    RAG agent that retrieves regulatory context and verifies
    compliance for clinical actions in robotic oncology trials.

    AGENT WORKFLOW:
    ---------------
    1. Receive query about a proposed clinical action
    2. Retrieve relevant regulatory document sections
    3. Verify action compliance against retrieved requirements
    4. Generate cited compliance report with recommendations
    5. Log all queries and decisions for audit trail

    CITATION POLICY:
    ----------------
    Every compliance determination includes the source document,
    section number, and specific requirement text. This enables
    the surgical team and regulatory auditors to verify the
    agent's reasoning.
    """

    def __init__(self):
        self.knowledge_base = RegulatoryKnowledgeBase()
        self.verifier = ComplianceVerifier()
        self._query_history: list[ComplianceReport] = []

    def check_compliance(
        self,
        action_description: str,
        context: dict,
        top_k: int = 5,
    ) -> ComplianceReport:
        """
        Check compliance of a proposed action.

        Args:
            action_description: Natural language description of the action
            context: Current clinical context
            top_k: Number of regulatory sections to retrieve

        Returns:
            ComplianceReport with cited findings
        """
        # 1. Retrieve relevant regulatory sections
        retrieved = self.knowledge_base.retrieve(action_description, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved)} regulatory sections for: {action_description[:50]}...")

        # 2. Verify compliance against each section
        checks = self.verifier.verify_action_compliance(action_description, context, retrieved)

        # 3. Determine overall status
        statuses = [c.status for c in checks]
        if ComplianceStatus.NON_COMPLIANT in statuses:
            overall = ComplianceStatus.NON_COMPLIANT
        elif ComplianceStatus.REQUIRES_REVIEW in statuses:
            overall = ComplianceStatus.REQUIRES_REVIEW
        elif ComplianceStatus.PARTIAL in statuses:
            overall = ComplianceStatus.PARTIAL
        else:
            overall = ComplianceStatus.COMPLIANT

        # 4. Gather citations
        citations = [
            {
                "citation": c.source_citation,
                "status": c.status.value,
                "requirement": c.requirement_text[:150],
            }
            for c in checks
        ]

        # 5. Generate recommendations
        recommendations = []
        for check in checks:
            if check.status == ComplianceStatus.NON_COMPLIANT:
                recommendations.append(f"[NON-COMPLIANT] {check.remediation} (Ref: {check.source_citation})")
            elif check.status == ComplianceStatus.REQUIRES_REVIEW:
                recommendations.append(f"[REVIEW NEEDED] {check.rationale} (Ref: {check.source_citation})")

        report = ComplianceReport(
            query=action_description,
            timestamp=time.time(),
            overall_status=overall,
            checks=checks,
            citations=citations,
            recommendations=recommendations,
            retrieved_sections=len(retrieved),
        )

        self._query_history.append(report)
        return report

    def answer_regulatory_query(self, question: str) -> dict:
        """
        Answer a regulatory question using retrieved documents.

        Args:
            question: Natural language regulatory question

        Returns:
            Answer with citations
        """
        retrieved = self.knowledge_base.retrieve(question, top_k=3)

        if not retrieved:
            return {
                "answer": "No relevant regulatory guidance found for this query.",
                "citations": [],
                "confidence": 0.0,
            }

        # Build answer from retrieved chunks
        answer_parts = []
        citations = []

        for result in retrieved:
            chunk = result.chunk
            answer_parts.append(
                f"Per {chunk.document_title}, {chunk.section_title} "
                f"(Section {chunk.section_number}): {chunk.content[:300]}"
            )
            citations.append(chunk.get_citation())

        return {
            "answer": "\n\n".join(answer_parts),
            "citations": citations,
            "confidence": retrieved[0].relevance_score if retrieved else 0.0,
            "retrieved_count": len(retrieved),
        }

    def get_query_history(self) -> list[dict]:
        """Get all compliance queries and results."""
        return [
            {
                "query": r.query,
                "overall_status": r.overall_status.value,
                "checks_performed": len(r.checks),
                "citations": len(r.citations),
                "timestamp": r.timestamp,
            }
            for r in self._query_history
        ]


# =============================================================================
# SECTION 5: REPORT FORMATTER
# =============================================================================


def format_compliance_report(report: ComplianceReport) -> str:
    """Format compliance report as readable text."""
    lines = [
        "=" * 70,
        "COMPLIANCE ASSESSMENT REPORT",
        "=" * 70,
        f"Query: {report.query}",
        f"Overall Status: {report.overall_status.value.upper()}",
        f"Regulatory Sections Retrieved: {report.retrieved_sections}",
        f"Checks Performed: {len(report.checks)}",
        "",
        "DETAILED FINDINGS:",
        "-" * 50,
    ]

    for check in report.checks:
        status_marker = {
            ComplianceStatus.COMPLIANT: "[PASS]",
            ComplianceStatus.NON_COMPLIANT: "[FAIL]",
            ComplianceStatus.REQUIRES_REVIEW: "[REVIEW]",
            ComplianceStatus.PARTIAL: "[PARTIAL]",
            ComplianceStatus.NOT_APPLICABLE: "[N/A]",
        }.get(check.status, "[?]")

        lines.append(f"\n  {status_marker} {check.source_citation}")
        lines.append(f"    Requirement: {check.requirement_text[:120]}")
        lines.append(f"    Rationale: {check.rationale}")
        if check.remediation:
            lines.append(f"    Remediation: {check.remediation}")

    if report.recommendations:
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 50)
        for rec in report.recommendations:
            lines.append(f"  -> {rec}")

    lines.append("")
    lines.append("CITATIONS:")
    lines.append("-" * 50)
    for citation in report.citations:
        lines.append(f"  [{citation['status'].upper():12s}] {citation['citation']}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# SECTION 6: DEMO
# =============================================================================


def run_rag_compliance_demo():
    """
    Demonstrate the Protocol RAG Compliance Agent.

    Tests compliance checking for various clinical scenarios:
    1. Standard compliant procedure step
    2. Excessive force violation
    3. Eligibility verification
    4. Regulatory question answering
    """
    logger.info("=" * 60)
    logger.info("PROTOCOL RAG COMPLIANCE AGENT DEMO")
    logger.info("=" * 60)

    agent = ProtocolRAGComplianceAgent()

    # Scenario 1: Compliant action
    print("\n" + "=" * 60)
    print("SCENARIO 1: Standard Compliant Action")
    print("=" * 60)

    report1 = agent.check_compliance(
        action_description="Begin robotic dissection of hilar structures with AI guidance",
        context={
            "informed_consent_signed": True,
            "robot_self_test_passed": True,
            "current_force_n": 3.0,
            "max_force_n": 10.0,
            "within_workspace": True,
            "surgeon_override_available": True,
            "estimated_blood_loss_ml": 50,
        },
    )
    print(format_compliance_report(report1))

    # Scenario 2: Force limit violation
    print("\n" + "=" * 60)
    print("SCENARIO 2: Force Limit Violation")
    print("=" * 60)

    report2 = agent.check_compliance(
        action_description="Continue tissue retraction during safety stopping assessment",
        context={
            "current_force_n": 18.0,
            "max_force_n": 10.0,
            "within_workspace": True,
            "estimated_blood_loss_ml": 200,
            "surgeon_override_available": True,
        },
    )
    print(format_compliance_report(report2))

    # Scenario 3: Eligibility check
    print("\n" + "=" * 60)
    print("SCENARIO 3: Patient Eligibility Verification")
    print("=" * 60)

    report3 = agent.check_compliance(
        action_description="Enroll patient with ECOG status 2 who meets other inclusion criteria",
        context={
            "ecog_status": 2,
            "age": 65,
            "histology_confirmed": True,
            "anc": 2000,
            "platelets": 150000,
        },
    )
    print(format_compliance_report(report3))

    # Scenario 4: Regulatory question
    print("\n" + "=" * 60)
    print("SCENARIO 4: Regulatory Question")
    print("=" * 60)

    answer = agent.answer_regulatory_query("What are the requirements for AI-based surgical planning in FDA guidance?")
    print("\nQuestion: What are the requirements for AI-based surgical planning?")
    print(f"Confidence: {answer['confidence']:.0%}")
    print(f"Sources Retrieved: {answer['retrieved_count']}")
    print("\nAnswer:")
    print(answer["answer"][:500])
    print("\nCitations:")
    for citation in answer["citations"]:
        print(f"  - {citation}")

    # Summary
    print("\n" + "=" * 60)
    print("QUERY HISTORY")
    print("=" * 60)
    for entry in agent.get_query_history():
        print(
            f"  [{entry['overall_status'].upper():15s}] {entry['query'][:60]}... "
            f"({entry['checks_performed']} checks, {entry['citations']} citations)"
        )

    return {
        "queries_processed": len(agent.get_query_history()),
        "scenarios_tested": 4,
    }


if __name__ == "__main__":
    result = run_rag_compliance_demo()
    print(f"\nDemo result: {result}")
