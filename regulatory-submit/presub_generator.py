"""
=============================================================================
FDA Pre-Submission (Pre-Sub) Package Generator for Physical AI Oncology Trials
=============================================================================

Generates structured FDA pre-submission meeting request packages for
AI/ML-enabled medical devices used in physical AI oncology applications,
including surgical robotics, treatment planning, and digital twin systems.

CLINICAL CONTEXT:
-----------------
FDA Pre-Submissions (Q-Subs) enable sponsors to obtain FDA feedback before
a formal marketing submission. For AI/ML-enabled oncology devices, a
well-structured Pre-Sub package accelerates the regulatory pathway by:
  - Clarifying the appropriate regulatory pathway (510(k), De Novo, PMA)
  - Aligning on AI/ML-specific testing requirements
  - Discussing clinical evidence expectations
  - Reviewing proposed Predetermined Change Control Plans (PCCP)
  - Addressing novel risk considerations for physical AI systems

This module generates complete Pre-Sub packages as structured Markdown
documents, including device descriptions, intended use statements,
proposed testing protocols, and specific questions for FDA review.

USE CASES COVERED:
    1. Pre-Sub meeting request generation for AI surgical planning systems
    2. Pre-Sub for robotic-assisted oncology procedures with AI guidance
    3. Pre-Sub for AI-driven treatment response prediction from digital twins
    4. Pre-Sub for multi-modal AI diagnostic and prognostic tools

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - FDA Q-Submission Program Guidance (Rev. 2025)
    - FDA Draft Guidance: AI-Enabled Device Software Functions (Jan 2025)
    - FDA PCCP Guidance for AI Devices (Aug 2025, finalized)
    - 21 CFR 807 — Premarket Notification (510(k))
    - 21 CFR 814 — Premarket Approval (PMA)
    - 21 CFR 860 — De Novo Classification
    - SPIRIT-AI / CONSORT-AI Reporting Extensions

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Requires institutional validation and regulatory review before deployment.
    This tool does not constitute an FDA submission and is not FDA-cleared.

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


class PreSubType(Enum):
    """FDA Pre-Submission meeting types."""

    PRE_SUB = "pre_submission"
    INFORMATIONAL = "informational_meeting"
    AGREEMENT_DETERMINATION = "agreement_or_determination"
    STUDY_RISK = "study_risk_determination"


class DeviceCategory(Enum):
    """Physical AI oncology device categories."""

    SURGICAL_PLANNING = "ai_surgical_planning"
    ROBOTIC_GUIDANCE = "robotic_ai_guidance"
    TREATMENT_PREDICTION = "treatment_response_prediction"
    DIAGNOSTIC_IMAGING = "ai_diagnostic_imaging"
    DIGITAL_TWIN = "digital_twin_clinical"
    DOSE_OPTIMIZATION = "radiation_dose_optimization"
    PATHOLOGY = "computational_pathology"


class RegulatoryPathway(Enum):
    """Proposed FDA regulatory pathways."""

    FIVE_TEN_K = "510k"
    DE_NOVO = "de_novo"
    PMA = "pma"
    UNDETERMINED = "undetermined"


@dataclass
class AIModelDescription:
    """Description of an AI/ML model component within the device."""

    model_name: str
    model_type: str  # "classification", "segmentation", "regression", "generative", "reinforcement_learning"
    architecture: str = ""
    input_modalities: list[str] = field(default_factory=list)
    output_type: str = ""
    training_data_description: str = ""
    training_data_size: int = 0
    validation_approach: str = ""
    performance_metrics: dict[str, float] = field(default_factory=dict)
    locked_or_adaptive: str = "locked"
    clinical_role: str = ""


@dataclass
class TestingProtocol:
    """Proposed testing protocol for FDA discussion."""

    protocol_name: str
    objective: str
    study_design: str = ""
    sample_size_rationale: str = ""
    endpoints: list[str] = field(default_factory=list)
    comparator: str = ""
    subgroup_analyses: list[str] = field(default_factory=list)
    statistical_methods: str = ""


@dataclass
class PreSubQuestion:
    """Specific question for FDA in the Pre-Sub package."""

    question_number: int
    topic: str
    question_text: str
    context: str = ""
    proposed_approach: str = ""


@dataclass
class PreSubPackage:
    """Complete Pre-Sub meeting request package."""

    package_id: str
    pre_sub_type: PreSubType
    device_name: str
    sponsor: str
    device_category: DeviceCategory
    proposed_pathway: RegulatoryPathway
    intended_use: str
    device_description: str = ""
    indications_for_use: str = ""
    ai_models: list[AIModelDescription] = field(default_factory=list)
    testing_protocols: list[TestingProtocol] = field(default_factory=list)
    questions: list[PreSubQuestion] = field(default_factory=list)
    predicate_device: str = ""
    oncology_indication: str = ""
    risk_considerations: list[str] = field(default_factory=list)
    meeting_preference: str = "written_response_only"
    generated_date: str = ""
    generated_markdown: str = ""


# =============================================================================
# SECTION 2: PRE-SUB TEMPLATES
# =============================================================================

PRESUB_QUESTION_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "regulatory_pathway": [
        {
            "topic": "Regulatory Classification",
            "question": (
                "Based on the device description and intended use provided, does FDA agree that the "
                "{pathway} pathway is appropriate for this AI/ML-enabled device?"
            ),
            "context": "Device classification and regulatory pathway selection.",
        },
        {
            "topic": "Predicate Device Suitability",
            "question": (
                "Does FDA agree that {predicate} is an appropriate predicate device for "
                "demonstrating substantial equivalence, given the AI/ML software component?"
            ),
            "context": "Applicable only for 510(k) pathway.",
        },
    ],
    "ai_ml_specific": [
        {
            "topic": "AI/ML Algorithm Documentation",
            "question": (
                "Is the proposed level of detail for the AI/ML algorithm description "
                "(architecture, training methodology, feature engineering) sufficient for "
                "the {pathway} submission?"
            ),
            "context": "Per FDA draft guidance on AI-enabled device software functions (Jan 2025).",
        },
        {
            "topic": "Training Data Representativeness",
            "question": (
                "Does FDA agree that the proposed training dataset composition "
                "(N={data_size}, sourced from {data_sources}) provides adequate "
                "demographic representation for the intended use population?"
            ),
            "context": "Demographic subgroup analysis for bias assessment.",
        },
        {
            "topic": "Performance Metrics",
            "question": (
                "Are the proposed performance endpoints ({metrics}) appropriate for "
                "demonstrating the safety and effectiveness of the AI/ML component "
                "for {indication}?"
            ),
            "context": "Clinical performance validation metrics.",
        },
    ],
    "clinical_evidence": [
        {
            "topic": "Clinical Study Design",
            "question": (
                "Does FDA agree with the proposed clinical study design "
                "({study_design}) for generating clinical evidence to support "
                "the {pathway} submission?"
            ),
            "context": "Clinical evidence requirements per device risk classification.",
        },
        {
            "topic": "Endpoint Acceptability",
            "question": (
                "Are the proposed primary and secondary endpoints sufficient to "
                "support the intended use claims for {device_name}?"
            ),
            "context": "Endpoint alignment with intended use statement.",
        },
    ],
    "pccp": [
        {
            "topic": "PCCP Scope",
            "question": (
                "Does FDA agree that the proposed Predetermined Change Control Plan "
                "scope, covering {pccp_scope}, appropriately balances innovation "
                "flexibility with safety assurance?"
            ),
            "context": "Per FDA PCCP guidance for AI devices (Aug 2025, finalized).",
        },
    ],
    "physical_ai": [
        {
            "topic": "Physical AI Risk Assessment",
            "question": (
                "Given the physical interaction between the AI system and the patient "
                "during {procedure}, does FDA agree with the proposed risk mitigation "
                "strategy including {mitigations}?"
            ),
            "context": "Physical AI-specific safety considerations for robotic/surgical devices.",
        },
        {
            "topic": "Human-AI Teaming",
            "question": (
                "Does FDA agree that the proposed human-AI interaction model, where the "
                "clinician {clinician_role}, provides adequate human oversight for the "
                "intended clinical workflow?"
            ),
            "context": "Human factors and autonomy level assessment.",
        },
    ],
}

RISK_TEMPLATES: dict[str, list[str]] = {
    "surgical_planning": [
        "Incorrect tumor boundary delineation leading to inadequate surgical margins",
        "AI surgical path recommendation conflicting with intraoperative findings",
        "Failure to account for patient-specific anatomical variations",
        "Over-reliance on AI planning reducing surgeon situational awareness",
        "Data pipeline failure during surgical planning causing incomplete analysis",
    ],
    "robotic_guidance": [
        "AI guidance error during physical robot-patient interaction",
        "Latency in AI processing causing delayed safety intervention",
        "Sensor fusion failure leading to incorrect spatial localization",
        "Software fault in control loop causing unintended robot motion",
        "Communication failure between AI planner and robotic actuators",
    ],
    "treatment_prediction": [
        "Inaccurate treatment response prediction leading to suboptimal therapy",
        "Bias in prediction model across demographic subgroups",
        "Model drift due to evolving treatment protocols or patient populations",
        "Overconfident predictions without adequate uncertainty quantification",
        "Missing or corrupted input data producing unreliable predictions",
    ],
    "digital_twin": [
        "Digital twin model divergence from actual patient physiology",
        "Simulation-based treatment recommendation not validated clinically",
        "Patient data synchronization failure causing stale twin state",
        "Computational resource constraints limiting real-time twin updates",
        "Insufficient model personalization for rare cancer subtypes",
    ],
}


# =============================================================================
# SECTION 3: PRE-SUB PACKAGE GENERATOR
# =============================================================================


class PreSubGenerator:
    """
    Generates FDA Pre-Submission meeting request packages for AI/ML-enabled
    physical AI oncology devices.

    Produces structured Markdown documents containing device descriptions,
    intended use statements, AI/ML model documentation, proposed testing
    protocols, and specific questions for FDA review.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Single-device Pre-Sub for AI surgical planning systems
    2. Multi-component Pre-Sub for robotic AI systems
    3. Pre-Sub with PCCP discussion for adaptive AI devices
    4. Informational meeting requests for novel device categories
    """

    def __init__(self, sponsor: str, contact_name: str = "", contact_email: str = ""):
        """
        Initialize Pre-Sub package generator.

        Args:
            sponsor: Sponsoring organization name.
            contact_name: Primary regulatory contact name.
            contact_email: Primary regulatory contact email.
        """
        self.sponsor = sponsor
        self.contact_name = contact_name
        self.contact_email = contact_email
        self._package_counter = 0

        logger.info(f"PreSubGenerator initialized for sponsor: {sponsor}")

    def create_package(
        self,
        device_name: str,
        device_category: DeviceCategory,
        intended_use: str,
        proposed_pathway: RegulatoryPathway = RegulatoryPathway.UNDETERMINED,
        pre_sub_type: PreSubType = PreSubType.PRE_SUB,
        device_description: str = "",
        indications_for_use: str = "",
        oncology_indication: str = "",
        predicate_device: str = "",
        meeting_preference: str = "written_response_only",
    ) -> PreSubPackage:
        """
        Create a new Pre-Sub package.

        Args:
            device_name: Name of the AI/ML-enabled device.
            device_category: Category of physical AI device.
            intended_use: Intended use statement.
            proposed_pathway: Proposed regulatory pathway.
            pre_sub_type: Type of Pre-Sub meeting requested.
            device_description: Detailed device description.
            indications_for_use: Indications for use statement.
            oncology_indication: Specific cancer type or clinical indication.
            predicate_device: Predicate device name (for 510(k)).
            meeting_preference: Meeting format preference.

        Returns:
            PreSubPackage with initialized fields.
        """
        self._package_counter += 1
        package_id = f"PRESUB-{datetime.now().strftime('%Y%m%d')}-{self._package_counter:04d}"

        package = PreSubPackage(
            package_id=package_id,
            pre_sub_type=pre_sub_type,
            device_name=device_name,
            sponsor=self.sponsor,
            device_category=device_category,
            proposed_pathway=proposed_pathway,
            intended_use=intended_use,
            device_description=device_description,
            indications_for_use=indications_for_use or intended_use,
            predicate_device=predicate_device,
            oncology_indication=oncology_indication,
            meeting_preference=meeting_preference,
            generated_date=datetime.now().isoformat(),
        )

        logger.info(f"Pre-Sub package created: {package_id} for {device_name}")
        return package

    def add_ai_model(self, package: PreSubPackage, model: AIModelDescription) -> None:
        """
        Add an AI/ML model description to the package.

        Args:
            package: Pre-Sub package to add the model to.
            model: AI/ML model description.
        """
        package.ai_models.append(model)
        logger.info(f"AI model '{model.model_name}' added to {package.package_id}")

    def add_testing_protocol(self, package: PreSubPackage, protocol: TestingProtocol) -> None:
        """
        Add a proposed testing protocol to the package.

        Args:
            package: Pre-Sub package to add the protocol to.
            protocol: Proposed testing protocol.
        """
        package.testing_protocols.append(protocol)
        logger.info(f"Testing protocol '{protocol.protocol_name}' added to {package.package_id}")

    def add_question(
        self, package: PreSubPackage, topic: str, question_text: str, context: str = "", proposed_approach: str = ""
    ) -> None:
        """
        Add a specific question for FDA to the package.

        Args:
            package: Pre-Sub package to add the question to.
            topic: Question topic/category.
            question_text: The question text.
            context: Supporting context for the question.
            proposed_approach: Sponsor's proposed approach.
        """
        question_number = len(package.questions) + 1
        package.questions.append(
            PreSubQuestion(
                question_number=question_number,
                topic=topic,
                question_text=question_text,
                context=context,
                proposed_approach=proposed_approach,
            )
        )

    def populate_risk_considerations(self, package: PreSubPackage) -> None:
        """
        Populate standard risk considerations based on device category.

        Args:
            package: Pre-Sub package to populate risks for.
        """
        category_key_map = {
            DeviceCategory.SURGICAL_PLANNING: "surgical_planning",
            DeviceCategory.ROBOTIC_GUIDANCE: "robotic_guidance",
            DeviceCategory.TREATMENT_PREDICTION: "treatment_prediction",
            DeviceCategory.DIGITAL_TWIN: "digital_twin",
            DeviceCategory.DOSE_OPTIMIZATION: "treatment_prediction",
            DeviceCategory.DIAGNOSTIC_IMAGING: "treatment_prediction",
            DeviceCategory.PATHOLOGY: "treatment_prediction",
        }
        key = category_key_map.get(package.device_category, "treatment_prediction")
        package.risk_considerations = list(RISK_TEMPLATES.get(key, []))

    def auto_generate_questions(self, package: PreSubPackage) -> None:
        """
        Auto-generate relevant Pre-Sub questions based on package content.

        Populates the questions list with template-based questions
        tailored to the device category and proposed pathway.

        Args:
            package: Pre-Sub package to generate questions for.
        """
        pathway_name = package.proposed_pathway.value
        data_size_total = sum(m.training_data_size for m in package.ai_models)
        data_sources = (
            ", ".join({src for m in package.ai_models for src in m.input_modalities}) or "institutional datasets"
        )
        metric_names = (
            ", ".join({k for m in package.ai_models for k in m.performance_metrics}) or "sensitivity, specificity, AUC"
        )

        # Regulatory pathway questions
        for tmpl in PRESUB_QUESTION_TEMPLATES["regulatory_pathway"]:
            if tmpl["topic"] == "Predicate Device Suitability" and not package.predicate_device:
                continue
            text = tmpl["question"].format(
                pathway=pathway_name,
                predicate=package.predicate_device or "N/A",
            )
            self.add_question(package, tmpl["topic"], text, tmpl["context"])

        # AI/ML-specific questions
        for tmpl in PRESUB_QUESTION_TEMPLATES["ai_ml_specific"]:
            text = tmpl["question"].format(
                pathway=pathway_name,
                data_size=data_size_total,
                data_sources=data_sources,
                metrics=metric_names,
                indication=package.oncology_indication or "the target oncology indication",
            )
            self.add_question(package, tmpl["topic"], text, tmpl["context"])

        # Clinical evidence questions
        study_design = (
            package.testing_protocols[0].study_design if package.testing_protocols else "retrospective validation"
        )
        for tmpl in PRESUB_QUESTION_TEMPLATES["clinical_evidence"]:
            text = tmpl["question"].format(
                study_design=study_design,
                pathway=pathway_name,
                device_name=package.device_name,
            )
            self.add_question(package, tmpl["topic"], text, tmpl["context"])

        # PCCP questions (if any model is adaptive)
        has_adaptive = any(m.locked_or_adaptive == "adaptive" for m in package.ai_models)
        if has_adaptive:
            adaptive_models = [m.model_name for m in package.ai_models if m.locked_or_adaptive == "adaptive"]
            pccp_scope = f"retraining and performance updates for {', '.join(adaptive_models)}"
            for tmpl in PRESUB_QUESTION_TEMPLATES["pccp"]:
                text = tmpl["question"].format(pccp_scope=pccp_scope)
                self.add_question(package, tmpl["topic"], text, tmpl["context"])

        # Physical AI questions
        category = package.device_category
        if category in (DeviceCategory.SURGICAL_PLANNING, DeviceCategory.ROBOTIC_GUIDANCE):
            for tmpl in PRESUB_QUESTION_TEMPLATES["physical_ai"]:
                text = tmpl["question"].format(
                    procedure=package.oncology_indication or "oncology surgical procedure",
                    mitigations="real-time safety monitoring, workspace boundaries, and human override",
                    clinician_role="retains final decision authority and can override AI recommendations",
                )
                self.add_question(package, tmpl["topic"], text, tmpl["context"])

    def generate_markdown(self, package: PreSubPackage) -> str:
        """
        Generate the complete Pre-Sub package as a Markdown document.

        Args:
            package: Pre-Sub package to render.

        Returns:
            Complete Pre-Sub package as Markdown string.
        """
        sections: list[str] = []

        # Header
        sections.append("# FDA Pre-Submission Meeting Request\n")
        sections.append(f"**Package ID:** {package.package_id}  ")
        sections.append(f"**Date:** {package.generated_date[:10]}  ")
        sections.append(f"**Sponsor:** {package.sponsor}  ")
        if self.contact_name:
            sections.append(f"**Contact:** {self.contact_name}  ")
        if self.contact_email:
            sections.append(f"**Email:** {self.contact_email}  ")
        sections.append(f"**Meeting Type:** {package.pre_sub_type.value.replace('_', ' ').title()}  ")
        sections.append(f"**Meeting Preference:** {package.meeting_preference.replace('_', ' ').title()}  ")
        sections.append("")

        # Disclaimer
        sections.append("> **RESEARCH USE ONLY.** This document is generated for research and planning ")
        sections.append("> purposes. It does not constitute an official FDA submission. Consult qualified ")
        sections.append("> regulatory counsel before filing any submission with the FDA.\n")

        # Device Description
        sections.append("## 1. Device Description\n")
        sections.append(f"**Device Name:** {package.device_name}  ")
        sections.append(f"**Device Category:** {package.device_category.value.replace('_', ' ').title()}  ")
        sections.append(
            f"**Proposed Regulatory Pathway:** {package.proposed_pathway.value.replace('_', ' ').title()}  "
        )
        if package.predicate_device:
            sections.append(f"**Predicate Device:** {package.predicate_device}  ")
        sections.append("")
        if package.device_description:
            sections.append(package.device_description)
            sections.append("")

        # Intended Use / Indications for Use
        sections.append("## 2. Intended Use / Indications for Use\n")
        sections.append(f"**Intended Use:** {package.intended_use}\n")
        if package.indications_for_use and package.indications_for_use != package.intended_use:
            sections.append(f"**Indications for Use:** {package.indications_for_use}\n")
        if package.oncology_indication:
            sections.append(f"**Oncology Indication:** {package.oncology_indication}\n")

        # AI/ML Model Descriptions
        if package.ai_models:
            sections.append("## 3. AI/ML Component Descriptions\n")
            for i, model in enumerate(package.ai_models, 1):
                sections.append(f"### 3.{i}. {model.model_name}\n")
                sections.append("| Attribute | Value |")
                sections.append("|-----------|-------|")
                sections.append(f"| Model Type | {model.model_type} |")
                if model.architecture:
                    sections.append(f"| Architecture | {model.architecture} |")
                if model.input_modalities:
                    sections.append(f"| Input Modalities | {', '.join(model.input_modalities)} |")
                if model.output_type:
                    sections.append(f"| Output | {model.output_type} |")
                sections.append(f"| Algorithm Lock | {model.locked_or_adaptive} |")
                if model.training_data_description:
                    sections.append(f"| Training Data | {model.training_data_description} |")
                if model.training_data_size > 0:
                    sections.append(f"| Training Set Size | {model.training_data_size:,} |")
                if model.validation_approach:
                    sections.append(f"| Validation | {model.validation_approach} |")
                if model.clinical_role:
                    sections.append(f"| Clinical Role | {model.clinical_role} |")
                sections.append("")
                if model.performance_metrics:
                    sections.append("**Performance Metrics:**\n")
                    sections.append("| Metric | Value |")
                    sections.append("|--------|-------|")
                    for metric, value in model.performance_metrics.items():
                        sections.append(f"| {metric} | {value:.4f} |")
                    sections.append("")

        # Risk Considerations
        if package.risk_considerations:
            sections.append("## 4. Risk Considerations\n")
            sections.append("The following risks have been identified for this device category:\n")
            for i, risk in enumerate(package.risk_considerations, 1):
                sections.append(f"{i}. {risk}")
            sections.append("")

        # Proposed Testing Protocols
        if package.testing_protocols:
            sections.append("## 5. Proposed Testing Protocols\n")
            for i, protocol in enumerate(package.testing_protocols, 1):
                sections.append(f"### 5.{i}. {protocol.protocol_name}\n")
                sections.append(f"**Objective:** {protocol.objective}\n")
                if protocol.study_design:
                    sections.append(f"**Study Design:** {protocol.study_design}\n")
                if protocol.comparator:
                    sections.append(f"**Comparator:** {protocol.comparator}\n")
                if protocol.sample_size_rationale:
                    sections.append(f"**Sample Size Rationale:** {protocol.sample_size_rationale}\n")
                if protocol.endpoints:
                    sections.append("**Endpoints:**\n")
                    for ep in protocol.endpoints:
                        sections.append(f"- {ep}")
                    sections.append("")
                if protocol.subgroup_analyses:
                    sections.append("**Subgroup Analyses:**\n")
                    for sg in protocol.subgroup_analyses:
                        sections.append(f"- {sg}")
                    sections.append("")
                if protocol.statistical_methods:
                    sections.append(f"**Statistical Methods:** {protocol.statistical_methods}\n")

        # Questions for FDA
        if package.questions:
            sections.append("## 6. Specific Questions for FDA\n")
            for q in package.questions:
                sections.append(f"### Question {q.question_number}: {q.topic}\n")
                sections.append(f"**Question:** {q.question_text}\n")
                if q.context:
                    sections.append(f"**Context:** {q.context}\n")
                if q.proposed_approach:
                    sections.append(f"**Sponsor's Proposed Approach:** {q.proposed_approach}\n")

        # Footer
        sections.append("---\n")
        sections.append("*Generated by Physical AI Oncology Trials — Regulatory Submission Automation*  ")
        sections.append(f"*Package ID: {package.package_id} | Generated: {package.generated_date[:10]}*  ")
        sections.append("*RESEARCH USE ONLY — Not an official FDA submission*\n")

        markdown = "\n".join(sections)
        package.generated_markdown = markdown

        logger.info(
            f"Pre-Sub Markdown generated for {package.package_id}: "
            f"{len(package.questions)} questions, "
            f"{len(package.ai_models)} AI models, "
            f"{len(package.testing_protocols)} testing protocols"
        )

        return markdown


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_presub_generator_demo() -> dict[str, Any]:
    """
    Demonstrate Pre-Sub package generation for an AI surgical planning system.

    Creates a complete Pre-Sub package with device description, AI model
    documentation, testing protocols, and questions for FDA review.

    Returns:
        Dictionary with demo results including package ID and generated document.
    """
    logger.info("=" * 60)
    logger.info("PRE-SUB PACKAGE GENERATOR DEMO")
    logger.info("=" * 60)

    generator = PreSubGenerator(
        sponsor="Physical AI Oncology Consortium",
        contact_name="Regulatory Affairs Office",
        contact_email="regulatory@example-consortium.org",
    )

    # Create package
    package = generator.create_package(
        device_name="OncoTwin AI Surgical Planning System",
        device_category=DeviceCategory.SURGICAL_PLANNING,
        intended_use=(
            "AI-assisted surgical planning for solid tumor resection in adult "
            "patients with non-small cell lung cancer (NSCLC), using patient-specific "
            "digital twin models derived from pre-operative CT and MRI imaging"
        ),
        proposed_pathway=RegulatoryPathway.DE_NOVO,
        device_description=(
            "The OncoTwin AI Surgical Planning System is a software-only medical device "
            "that integrates deep learning-based tumor segmentation with patient-specific "
            "digital twin simulation to generate optimized surgical resection plans. The "
            "system processes pre-operative CT and MRI volumetric data, generates 3D tumor "
            "boundary maps, simulates multiple surgical approaches, and recommends optimal "
            "resection paths based on predicted oncological outcomes and surgical risk."
        ),
        oncology_indication="Non-small cell lung cancer (NSCLC), stages I-IIIA",
    )

    # Add AI models
    generator.add_ai_model(
        package,
        AIModelDescription(
            model_name="TumorSeg-3D",
            model_type="segmentation",
            architecture="3D U-Net with attention gates",
            input_modalities=["CT volumetric", "MRI T1-weighted", "MRI T2-weighted"],
            output_type="3D tumor boundary mask with confidence map",
            training_data_description="Multi-institutional dataset from 5 NCI-designated cancer centers",
            training_data_size=12500,
            validation_approach="5-fold cross-validation with external test set",
            performance_metrics={
                "dice_coefficient": 0.912,
                "sensitivity": 0.945,
                "specificity": 0.987,
                "hausdorff_95": 3.21,
            },
            locked_or_adaptive="locked",
            clinical_role="Automated tumor boundary delineation for surgical planning",
        ),
    )
    generator.add_ai_model(
        package,
        AIModelDescription(
            model_name="PathOptimizer-RL",
            model_type="reinforcement_learning",
            architecture="Proximal Policy Optimization (PPO) with anatomical constraints",
            input_modalities=["3D tumor segmentation", "vascular anatomy", "tissue density map"],
            output_type="Ranked surgical approach paths with predicted outcomes",
            training_data_description="Simulated surgical scenarios from digital twin environments",
            training_data_size=50000,
            validation_approach="Clinical expert review of top-3 recommended paths",
            performance_metrics={"path_safety_score": 0.934, "margin_adequacy": 0.891, "expert_agreement": 0.856},
            locked_or_adaptive="adaptive",
            clinical_role="Surgical path optimization with outcome prediction",
        ),
    )

    # Add testing protocol
    generator.add_testing_protocol(
        package,
        TestingProtocol(
            protocol_name="Retrospective Clinical Validation Study",
            objective="Validate AI surgical planning accuracy against expert surgeon consensus",
            study_design="Multi-reader multi-case retrospective study with paired comparison",
            sample_size_rationale="200 cases provides 80% power to detect 10% improvement in margin adequacy (alpha=0.05)",
            endpoints=[
                "Primary: Surgical margin adequacy (R0 resection rate) per pathology review",
                "Secondary: Expert agreement with AI-recommended approach (5-point Likert)",
                "Secondary: Time to complete surgical planning (AI-assisted vs. standard)",
                "Exploratory: Subgroup analysis by tumor size, location, and stage",
            ],
            comparator="Expert surgeon consensus panel (3 thoracic surgeons, blinded)",
            subgroup_analyses=[
                "Tumor size (<2cm, 2-5cm, >5cm)",
                "Tumor location (central vs. peripheral)",
                "Disease stage (I, II, IIIA)",
            ],
            statistical_methods="Non-inferiority with pre-specified margin of 5%, Bonferroni correction for multiplicity",
        ),
    )

    # Populate risks and auto-generate questions
    generator.populate_risk_considerations(package)
    generator.auto_generate_questions(package)

    # Generate Markdown
    markdown = generator.generate_markdown(package)

    print(f"\nPre-Sub Package: {package.package_id}")
    print(f"Device: {package.device_name}")
    print(f"Pathway: {package.proposed_pathway.value}")
    print(f"AI Models: {len(package.ai_models)}")
    print(f"Questions for FDA: {len(package.questions)}")
    print(f"Document length: {len(markdown):,} characters")
    print("\n--- Generated Document Preview (first 500 chars) ---")
    print(markdown[:500])
    print("...")

    return {
        "package_id": package.package_id,
        "device_name": package.device_name,
        "question_count": len(package.questions),
        "model_count": len(package.ai_models),
        "document_length": len(markdown),
        "status": "demo_complete",
    }


if __name__ == "__main__":
    result = run_presub_generator_demo()
    print(f"\nDemo completed: {result}")
