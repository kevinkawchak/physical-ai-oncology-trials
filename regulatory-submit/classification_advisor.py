"""
=============================================================================
510(k) / De Novo Classification Decision Advisor
for Physical AI Oncology Trials
=============================================================================

Provides interactive decision support for determining the appropriate
FDA regulatory pathway for AI/ML-enabled oncology devices, analyzing
device characteristics, predicate devices, and novel features to
recommend 510(k), De Novo, or PMA pathways with regulatory justification.

CLINICAL CONTEXT:
-----------------
Selecting the correct FDA regulatory pathway is critical for AI/ML-enabled
oncology devices. The pathway determines submission requirements, timeline,
and evidence expectations:
  - 510(k): Substantial equivalence to a legally marketed predicate device
  - De Novo: Novel device with no predicate, requesting Class I/II classification
  - PMA: Pre-Market Approval for Class III (high-risk) devices

As of December 2025, the majority of AI/ML oncology devices have been
authorized via 510(k) or De Novo pathways. Physical AI devices involving
direct patient contact (surgical robotics) may face additional scrutiny
under IEC 80601-2-77 and ISO 13482.

USE CASES COVERED:
    1. Pathway recommendation for software-only AI diagnostic tools
    2. Pathway analysis for AI-guided surgical robotics
    3. De Novo vs. 510(k) decision for novel AI algorithms
    4. Multi-factor risk classification for combination AI systems

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+

REFERENCES:
    - FDA Device Classification Procedures (21 CFR Part 860)
    - FDA 510(k) Program Guidance (2024)
    - FDA De Novo Classification Process Guidance (2024)
    - FDA Draft Guidance: AI-Enabled Device Software Functions (Jan 2025)
    - FDA CDRH Product Classification Database

DISCLAIMER: RESEARCH USE ONLY. Not approved for regulatory decision-making.
    Consult qualified regulatory counsel for pathway determination.
    This tool provides informational analysis only.

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


class RecommendedPathway(Enum):
    """FDA regulatory pathway recommendations."""

    FIVE_TEN_K = "510k"
    DE_NOVO = "de_novo"
    PMA = "pma"
    EXEMPT = "exempt"
    UNDETERMINED = "undetermined"


class DeviceRiskClass(Enum):
    """FDA device risk classification."""

    CLASS_I = "I"
    CLASS_II = "II"
    CLASS_III = "III"


class SoftwareRiskLevel(Enum):
    """IEC 62304 software safety classification."""

    CLASS_A = "A"  # No injury or damage to health
    CLASS_B = "B"  # Non-serious injury
    CLASS_C = "C"  # Death or serious injury


class AIDeviceType(Enum):
    """AI/ML device functional categories."""

    CADe = "computer_aided_detection"
    CADx = "computer_aided_diagnosis"
    CADt = "computer_aided_triage"
    TREATMENT_PLANNING = "treatment_planning"
    PROGNOSTIC = "prognostic"
    SURGICAL_GUIDANCE = "surgical_guidance"
    ROBOTIC_CONTROL = "robotic_control"
    DIGITAL_TWIN = "digital_twin"
    DOSE_OPTIMIZATION = "dose_optimization"


@dataclass
class DeviceCharacteristics:
    """Characteristics of the device under evaluation."""

    device_name: str
    device_type: AIDeviceType
    intended_use: str
    oncology_indication: str = ""
    software_only: bool = True
    patient_contact: bool = False
    autonomous_operation: bool = False
    clinician_in_loop: bool = True
    real_time_processing: bool = False
    locked_algorithm: bool = True
    novel_algorithm: bool = False
    training_data_size: int = 0
    clinical_evidence_available: bool = False
    life_threatening_condition: bool = False
    irreversibly_debilitating: bool = False


@dataclass
class PredicateAnalysis:
    """Analysis of potential predicate devices for 510(k)."""

    predicate_name: str
    predicate_510k_number: str = ""
    same_intended_use: bool = True
    same_technology: bool = True
    different_questions_of_safety_efficacy: bool = False
    predicate_has_ai: bool = False
    performance_comparison_feasible: bool = True
    notes: str = ""


@dataclass
class DecisionFactor:
    """Individual factor in the classification decision."""

    factor_name: str
    description: str
    value: str
    weight: float = 1.0
    pathway_implications: dict[str, str] = field(default_factory=dict)


@dataclass
class ClassificationRecommendation:
    """Complete classification recommendation with justification."""

    recommendation_id: str
    device_name: str
    recommended_pathway: RecommendedPathway
    recommended_risk_class: DeviceRiskClass
    software_risk_level: SoftwareRiskLevel
    confidence: str = "moderate"  # "high", "moderate", "low"
    primary_justification: str = ""
    decision_factors: list[DecisionFactor] = field(default_factory=list)
    predicate_analyses: list[PredicateAnalysis] = field(default_factory=list)
    special_considerations: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    alternative_pathways: list[str] = field(default_factory=list)
    generated_date: str = ""
    generated_markdown: str = ""


# =============================================================================
# SECTION 2: CLASSIFICATION RULES
# =============================================================================

DEVICE_TYPE_BASELINE_RISK: dict[AIDeviceType, dict[str, Any]] = {
    AIDeviceType.CADe: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.FIVE_TEN_K,
        "sw_risk": SoftwareRiskLevel.CLASS_B,
        "notes": "Established predicate devices exist for many CADe applications",
        "product_codes": ["QBS", "QDQ", "QMT"],
    },
    AIDeviceType.CADx: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_B,
        "notes": "CADx with autonomous diagnostic claims often requires De Novo",
        "product_codes": ["QBS", "QMT"],
    },
    AIDeviceType.CADt: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_C,
        "notes": "Triage devices prioritizing clinical review are often De Novo",
        "product_codes": ["QAS"],
    },
    AIDeviceType.TREATMENT_PLANNING: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_C,
        "notes": "AI treatment planning without predicate typically requires De Novo",
        "product_codes": ["MUJ", "NQR"],
    },
    AIDeviceType.PROGNOSTIC: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_B,
        "notes": "Prognostic AI tools (e.g., ArteraAI) have used De Novo pathway",
        "product_codes": ["QMT"],
    },
    AIDeviceType.SURGICAL_GUIDANCE: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_C,
        "notes": "AI-guided surgical navigation raises Class C software risk",
        "product_codes": ["NAY", "NQR"],
    },
    AIDeviceType.ROBOTIC_CONTROL: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_C,
        "notes": "Robotic AI with patient contact requires rigorous safety evidence",
        "product_codes": ["NAY"],
    },
    AIDeviceType.DIGITAL_TWIN: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.DE_NOVO,
        "sw_risk": SoftwareRiskLevel.CLASS_C,
        "notes": "Digital twin clinical decision tools are largely novel — De Novo expected",
        "product_codes": [],
    },
    AIDeviceType.DOSE_OPTIMIZATION: {
        "typical_class": DeviceRiskClass.CLASS_II,
        "typical_pathway": RecommendedPathway.FIVE_TEN_K,
        "sw_risk": SoftwareRiskLevel.CLASS_C,
        "notes": "Radiation dose optimization has established predicates in some categories",
        "product_codes": ["MUJ", "IYO"],
    },
}

ESCALATION_FACTORS: list[dict[str, Any]] = [
    {
        "name": "autonomous_operation",
        "description": "Device operates autonomously without real-time clinician oversight",
        "class_escalation": True,
        "pathway_impact": "Increases likelihood of PMA or De Novo with stringent special controls",
    },
    {
        "name": "patient_contact",
        "description": "Device physically interacts with or contacts the patient",
        "class_escalation": True,
        "pathway_impact": "Physical contact devices face IEC 80601-2-77 requirements",
    },
    {
        "name": "life_threatening_condition",
        "description": "Device addresses a life-threatening or irreversibly debilitating condition",
        "class_escalation": False,
        "pathway_impact": "May qualify for Breakthrough Device Designation — priority review",
    },
    {
        "name": "novel_algorithm",
        "description": "AI/ML algorithm is novel with no prior FDA-authorized similar technology",
        "class_escalation": False,
        "pathway_impact": "Novelty typically precludes 510(k) — De Novo or PMA required",
    },
]


# =============================================================================
# SECTION 3: CLASSIFICATION ADVISOR
# =============================================================================


class ClassificationAdvisor:
    """
    Provides regulatory pathway recommendations for AI/ML-enabled
    oncology devices based on device characteristics, predicate analysis,
    and FDA classification rules.

    SUPPORTED WORKFLOWS:
    -------------------
    1. Automated pathway recommendation from device characteristics
    2. Predicate device suitability analysis
    3. Risk classification determination
    4. Decision factor documentation with regulatory references
    """

    def __init__(self) -> None:
        """Initialize classification advisor."""
        self._recommendation_counter = 0
        logger.info("ClassificationAdvisor initialized")

    def analyze(
        self,
        characteristics: DeviceCharacteristics,
        predicate_analyses: list[PredicateAnalysis] | None = None,
    ) -> ClassificationRecommendation:
        """
        Analyze device characteristics and produce a pathway recommendation.

        Args:
            characteristics: Device characteristics for evaluation.
            predicate_analyses: Optional predicate device analyses.

        Returns:
            ClassificationRecommendation with pathway, justification, and next steps.
        """
        self._recommendation_counter += 1
        rec_id = f"REC-{datetime.now().strftime('%Y%m%d')}-{self._recommendation_counter:04d}"

        # Start with baseline from device type
        baseline = DEVICE_TYPE_BASELINE_RISK.get(
            characteristics.device_type,
            {
                "typical_class": DeviceRiskClass.CLASS_II,
                "typical_pathway": RecommendedPathway.DE_NOVO,
                "sw_risk": SoftwareRiskLevel.CLASS_B,
                "notes": "No baseline data for this device type",
            },
        )

        pathway = baseline["typical_pathway"]
        risk_class = baseline["typical_class"]
        sw_risk = baseline["sw_risk"]
        confidence = "moderate"
        factors: list[DecisionFactor] = []
        special: list[str] = []
        alternatives: list[str] = []

        # Factor 1: Device type baseline
        factors.append(
            DecisionFactor(
                factor_name="Device Type Baseline",
                description=f"Baseline for {characteristics.device_type.value}",
                value=baseline["notes"],
                pathway_implications={
                    pathway.value: "Baseline recommendation based on device type",
                },
            )
        )

        # Factor 2: Predicate availability
        suitable_predicates = [
            p
            for p in (predicate_analyses or [])
            if p.same_intended_use and p.same_technology and not p.different_questions_of_safety_efficacy
        ]
        if suitable_predicates:
            pathway = RecommendedPathway.FIVE_TEN_K
            confidence = "high"
            factors.append(
                DecisionFactor(
                    factor_name="Predicate Availability",
                    description=f"{len(suitable_predicates)} suitable predicate(s) identified",
                    value=suitable_predicates[0].predicate_name,
                    pathway_implications={
                        "510k": "Predicate device supports substantial equivalence argument",
                        "de_novo": "Alternative if predicate comparison raises new safety/efficacy questions",
                    },
                )
            )
            alternatives.append("De Novo (if FDA disagrees on predicate suitability)")
        else:
            if pathway == RecommendedPathway.FIVE_TEN_K:
                pathway = RecommendedPathway.DE_NOVO
            factors.append(
                DecisionFactor(
                    factor_name="Predicate Availability",
                    description="No suitable predicate device identified",
                    value="No predicate",
                    pathway_implications={
                        "de_novo": "No predicate requires De Novo or PMA pathway",
                    },
                )
            )

        # Factor 3: Novel algorithm
        if characteristics.novel_algorithm:
            if pathway == RecommendedPathway.FIVE_TEN_K:
                pathway = RecommendedPathway.DE_NOVO
                confidence = "moderate"
            factors.append(
                DecisionFactor(
                    factor_name="Algorithm Novelty",
                    description="Novel AI/ML algorithm without prior FDA authorization",
                    value="Novel",
                    pathway_implications={
                        "de_novo": "Novelty typically precludes 510(k) substantial equivalence",
                    },
                )
            )

        # Factor 4: Autonomous operation
        if characteristics.autonomous_operation:
            sw_risk = SoftwareRiskLevel.CLASS_C
            if not characteristics.clinician_in_loop:
                risk_class = DeviceRiskClass.CLASS_III
                pathway = RecommendedPathway.PMA
                confidence = "moderate"
            special.append("Autonomous operation without clinician-in-the-loop oversight raises Class III risk")
            factors.append(
                DecisionFactor(
                    factor_name="Autonomy Level",
                    description="Device autonomy and human oversight assessment",
                    value="Autonomous" if not characteristics.clinician_in_loop else "Supervised autonomous",
                    pathway_implications={
                        "pma": "Fully autonomous clinical devices typically require PMA",
                        "de_novo": "Supervised autonomy may be acceptable for De Novo with special controls",
                    },
                )
            )

        # Factor 5: Patient contact
        if characteristics.patient_contact:
            sw_risk = SoftwareRiskLevel.CLASS_C
            special.append("Physical patient contact requires IEC 80601-2-77 compliance assessment")
            special.append("ISO 13482 robot safety standards apply for robotic systems")
            factors.append(
                DecisionFactor(
                    factor_name="Patient Contact",
                    description="Device physically interacts with patient",
                    value="Direct contact",
                    pathway_implications={
                        "de_novo": "Physical AI devices with patient contact require additional safety evidence",
                    },
                )
            )

        # Factor 6: Life-threatening / irreversibly debilitating
        if characteristics.life_threatening_condition or characteristics.irreversibly_debilitating:
            special.append("Device may qualify for Breakthrough Device Designation — priority review pathway")
            alternatives.append("Breakthrough Device Designation (parallel with primary pathway)")
            factors.append(
                DecisionFactor(
                    factor_name="Condition Severity",
                    description="Device addresses life-threatening or irreversibly debilitating condition",
                    value="Life-threatening"
                    if characteristics.life_threatening_condition
                    else "Irreversibly debilitating",
                    pathway_implications={
                        "breakthrough": "Eligible for Breakthrough Device Designation (priority review)",
                    },
                )
            )

        # Factor 7: Software classification
        if not characteristics.software_only:
            special.append("Hardware component requires additional manufacturing and biocompatibility considerations")

        factors.append(
            DecisionFactor(
                factor_name="Software Risk Classification",
                description=f"IEC 62304 software safety class: {sw_risk.value}",
                value=sw_risk.value,
                pathway_implications={
                    "documentation": f"Class {sw_risk.value} software lifecycle documentation required per IEC 62304",
                },
            )
        )

        # Build justification
        justification = self._build_justification(characteristics, pathway, risk_class, suitable_predicates)

        # Build next steps
        next_steps = self._build_next_steps(pathway, characteristics)

        recommendation = ClassificationRecommendation(
            recommendation_id=rec_id,
            device_name=characteristics.device_name,
            recommended_pathway=pathway,
            recommended_risk_class=risk_class,
            software_risk_level=sw_risk,
            confidence=confidence,
            primary_justification=justification,
            decision_factors=factors,
            predicate_analyses=predicate_analyses or [],
            special_considerations=special,
            recommended_next_steps=next_steps,
            alternative_pathways=alternatives,
            generated_date=datetime.now().isoformat(),
        )

        logger.info(
            f"Classification recommendation {rec_id}: "
            f"{pathway.value} (Class {risk_class.value}, SW Class {sw_risk.value})"
        )

        return recommendation

    def _build_justification(
        self,
        chars: DeviceCharacteristics,
        pathway: RecommendedPathway,
        risk_class: DeviceRiskClass,
        predicates: list[PredicateAnalysis],
    ) -> str:
        """Build primary justification text."""
        if pathway == RecommendedPathway.FIVE_TEN_K:
            pred_name = predicates[0].predicate_name if predicates else "identified predicate"
            return (
                f"The {chars.device_name} is recommended for the 510(k) pathway based on "
                f"substantial equivalence to {pred_name}. The device shares the same intended "
                f"use and technological characteristics, and does not raise different questions "
                f"of safety and effectiveness."
            )
        elif pathway == RecommendedPathway.DE_NOVO:
            return (
                f"The {chars.device_name} is recommended for the De Novo classification pathway. "
                f"No suitable predicate device has been identified for 510(k) clearance, and the "
                f"device presents low-to-moderate risk appropriate for Class {risk_class.value} "
                f"classification with special controls."
            )
        elif pathway == RecommendedPathway.PMA:
            return (
                f"The {chars.device_name} is recommended for the PMA pathway due to its "
                f"Class III risk classification. The device's autonomous operation "
                f"and/or high-risk intended use require the highest level of regulatory scrutiny "
                f"with clinical evidence of safety and effectiveness."
            )
        return f"Regulatory pathway for {chars.device_name} requires further analysis."

    def _build_next_steps(self, pathway: RecommendedPathway, chars: DeviceCharacteristics) -> list[str]:
        """Build recommended next steps."""
        steps = ["Consult qualified regulatory counsel to validate pathway recommendation"]

        if pathway == RecommendedPathway.FIVE_TEN_K:
            steps.extend(
                [
                    "Prepare substantial equivalence comparison with identified predicate device",
                    "Document AI/ML algorithm per FDA draft guidance (Jan 2025)",
                    "Prepare 510(k) summary or statement",
                    "Consider Pre-Submission meeting to confirm predicate selection",
                ]
            )
        elif pathway == RecommendedPathway.DE_NOVO:
            steps.extend(
                [
                    "File Pre-Submission (Q-Sub) to discuss De Novo classification with FDA",
                    "Prepare risk assessment and proposed special controls",
                    "Develop clinical evidence plan (bench testing, clinical validation)",
                    "Document AI/ML algorithm per FDA draft guidance (Jan 2025)",
                    "Consider PCCP for adaptive algorithm components",
                ]
            )
        elif pathway == RecommendedPathway.PMA:
            steps.extend(
                [
                    "File Pre-Submission (Q-Sub) to discuss PMA requirements with FDA",
                    "Design and execute clinical trial per FDA-agreed protocol",
                    "Prepare comprehensive PMA application package",
                    "Plan for Advisory Panel meeting if applicable",
                ]
            )

        if chars.life_threatening_condition or chars.irreversibly_debilitating:
            steps.append("Consider filing Breakthrough Device Designation request")

        if not chars.locked_algorithm:
            steps.append("Prepare Predetermined Change Control Plan (PCCP) for adaptive algorithm")

        return steps

    def generate_markdown(self, recommendation: ClassificationRecommendation) -> str:
        """
        Generate classification recommendation as Markdown document.

        Args:
            recommendation: Classification recommendation to render.

        Returns:
            Complete recommendation as Markdown string.
        """
        sections: list[str] = []

        sections.append("# FDA Regulatory Pathway Classification Recommendation\n")
        sections.append(f"**Recommendation ID:** {recommendation.recommendation_id}  ")
        sections.append(f"**Device:** {recommendation.device_name}  ")
        sections.append(f"**Date:** {recommendation.generated_date[:10]}  ")
        sections.append("")
        sections.append("> **RESEARCH USE ONLY.** This recommendation is for informational and planning ")
        sections.append("> purposes. Consult qualified regulatory counsel for pathway determination.\n")

        # Recommendation Summary
        sections.append("## Recommendation Summary\n")
        sections.append("| Attribute | Recommendation |")
        sections.append("|-----------|---------------|")
        sections.append(
            f"| Regulatory Pathway | **{recommendation.recommended_pathway.value.replace('_', ' ').upper()}** |"
        )
        sections.append(f"| Device Risk Class | Class {recommendation.recommended_risk_class.value} |")
        sections.append(f"| Software Safety Class | IEC 62304 Class {recommendation.software_risk_level.value} |")
        sections.append(f"| Confidence | {recommendation.confidence.title()} |")
        sections.append("")

        sections.append(f"**Justification:** {recommendation.primary_justification}\n")

        # Decision Factors
        if recommendation.decision_factors:
            sections.append("## Decision Factors\n")
            for i, factor in enumerate(recommendation.decision_factors, 1):
                sections.append(f"### {i}. {factor.factor_name}\n")
                sections.append(f"{factor.description}\n")
                sections.append(f"**Assessment:** {factor.value}\n")
                if factor.pathway_implications:
                    for path, impl in factor.pathway_implications.items():
                        sections.append(f"- *{path}*: {impl}")
                    sections.append("")

        # Predicate Analysis
        if recommendation.predicate_analyses:
            sections.append("## Predicate Device Analysis\n")
            sections.append("| Predicate | Same Use | Same Tech | New S/E Questions | Suitable |")
            sections.append("|-----------|----------|-----------|-------------------|----------|")
            for p in recommendation.predicate_analyses:
                suitable = p.same_intended_use and p.same_technology and not p.different_questions_of_safety_efficacy
                sections.append(
                    f"| {p.predicate_name} | "
                    f"{'Yes' if p.same_intended_use else 'No'} | "
                    f"{'Yes' if p.same_technology else 'No'} | "
                    f"{'Yes' if p.different_questions_of_safety_efficacy else 'No'} | "
                    f"{'Yes' if suitable else 'No'} |"
                )
            sections.append("")

        # Special Considerations
        if recommendation.special_considerations:
            sections.append("## Special Considerations\n")
            for item in recommendation.special_considerations:
                sections.append(f"- {item}")
            sections.append("")

        # Next Steps
        if recommendation.recommended_next_steps:
            sections.append("## Recommended Next Steps\n")
            for i, step in enumerate(recommendation.recommended_next_steps, 1):
                sections.append(f"{i}. {step}")
            sections.append("")

        # Alternative Pathways
        if recommendation.alternative_pathways:
            sections.append("## Alternative Pathways\n")
            for alt in recommendation.alternative_pathways:
                sections.append(f"- {alt}")
            sections.append("")

        sections.append("---\n")
        sections.append("*Generated by Physical AI Oncology Trials — Classification Advisor*  ")
        sections.append(f"*Recommendation ID: {recommendation.recommendation_id}*  ")
        sections.append("*RESEARCH USE ONLY — Not official regulatory guidance*\n")

        markdown = "\n".join(sections)
        recommendation.generated_markdown = markdown

        logger.info(f"Classification Markdown generated for {recommendation.recommendation_id}")

        return markdown


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_classification_advisor_demo() -> dict[str, Any]:
    """
    Demonstrate classification decision support for two AI oncology devices.

    Returns:
        Dictionary with demo results.
    """
    logger.info("=" * 60)
    logger.info("CLASSIFICATION ADVISOR DEMO")
    logger.info("=" * 60)

    advisor = ClassificationAdvisor()

    # Device 1: AI surgical planning (novel, no predicate)
    chars1 = DeviceCharacteristics(
        device_name="OncoTwin AI Surgical Planning System",
        device_type=AIDeviceType.TREATMENT_PLANNING,
        intended_use="AI-assisted surgical planning for NSCLC tumor resection",
        oncology_indication="Non-small cell lung cancer (NSCLC)",
        software_only=True,
        clinician_in_loop=True,
        novel_algorithm=True,
        training_data_size=12500,
        life_threatening_condition=True,
    )
    rec1 = advisor.analyze(chars1)
    md1 = advisor.generate_markdown(rec1)

    print(f"\nDevice 1: {chars1.device_name}")
    print(f"  Pathway: {rec1.recommended_pathway.value}")
    print(f"  Class: {rec1.recommended_risk_class.value}")
    print(f"  SW Risk: {rec1.software_risk_level.value}")
    print(f"  Confidence: {rec1.confidence}")

    # Device 2: AI CADe with predicate
    chars2 = DeviceCharacteristics(
        device_name="OncoDetect CT Lung Nodule Detector",
        device_type=AIDeviceType.CADe,
        intended_use="Computer-aided detection of pulmonary nodules on CT imaging",
        oncology_indication="Lung cancer screening",
        software_only=True,
        clinician_in_loop=True,
        locked_algorithm=True,
        training_data_size=25000,
        clinical_evidence_available=True,
    )
    predicates = [
        PredicateAnalysis(
            predicate_name="Riverain ClearRead CT",
            predicate_510k_number="K201501",
            same_intended_use=True,
            same_technology=True,
            predicate_has_ai=True,
            performance_comparison_feasible=True,
        ),
    ]
    rec2 = advisor.analyze(chars2, predicate_analyses=predicates)
    md2 = advisor.generate_markdown(rec2)

    print(f"\nDevice 2: {chars2.device_name}")
    print(f"  Pathway: {rec2.recommended_pathway.value}")
    print(f"  Class: {rec2.recommended_risk_class.value}")
    print(f"  SW Risk: {rec2.software_risk_level.value}")
    print(f"  Confidence: {rec2.confidence}")

    return {
        "device_1": {
            "name": chars1.device_name,
            "pathway": rec1.recommended_pathway.value,
            "class": rec1.recommended_risk_class.value,
        },
        "device_2": {
            "name": chars2.device_name,
            "pathway": rec2.recommended_pathway.value,
            "class": rec2.recommended_risk_class.value,
        },
        "status": "demo_complete",
    }


if __name__ == "__main__":
    result = run_classification_advisor_demo()
    print(f"\nDemo completed: {result}")
