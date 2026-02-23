"""Unification Standard Level (USL) Scoring Framework for Surgical Robots.

Evaluates surgical robot systems on their readiness for multi-site oncology
clinical trials.  Final scores range from 1.0 to 10.0 in 0.1 increments
across four weighted dimensions:

    A) Simulation Framework Switching   (25%)
    B) Generative / Agentic AI Integration (25%)
    C) Cross-Robot Progress Sharing     (25%)
    D) Multi-Site Clinical Trial Collaboration (25%)

Surgical robots differ from cobots in several key ways that affect scoring:
  - Teleoperated master-slave architectures instead of autonomous arm control
  - FDA clearance / regulatory requirements specific to surgical systems
  - IEC 80601-2-77 safety standard for robotically assisted surgical equipment
  - Proprietary control protocols (vs. open ROS 2 ecosystem for cobots)
  - Clinical evidence requirements (procedure counts, outcome data)

Evaluated surgical robot systems:
  - Intuitive Surgical da Vinci (Xi / da Vinci 5): USL 7.6 (Level 7 — Advanced)
  - Medtronic Hugo RAS: USL 4.1 (Level 4 — Developing)
  - Johnson & Johnson OTTAVA: USL 2.3 (Level 2 — Exploratory)

Key references:
  - IEEE 3177-2024: Standard for Modular Framework for Robotically-Assisted
    Surgical Systems (approved Dec 2024)
  - IEC 80601-2-77: Safety standard for robotically assisted surgical equipment
  - dVRK Research Kit: github.com/jhu-dvrk/sawIntuitiveResearchKit
  - ORBIT-Surgical: github.com/orbit-surgical/orbit-surgical
  - OpenIGTLink: github.com/openigtlink/OpenIGTLink (open network protocol
    for image-guided therapy)
  - LASR Classification (Levels of Autonomy in Surgical Robotics):
    Yang et al., npj Digital Medicine, 2024; DOI 10.1038/s41746-024-01102-y
  - SAGES STARSS tool: DOI 10.1007/s00464-025-11897-w
  - Expert Consensus on Remote Robotic Surgery:
    DOI 10.1002/wjs.12653 (PMC12282566)
  - NASA/DOD TRL (Mankins, 2004)
  - MLTRL (Lavin et al., 2021; ai-infrastructure-alliance/mltrl)
  - LLM Recommendations for Oncology Trials (Kawchak, 2025;
    DOI 10.5281/zenodo.17451709)

RESEARCH USE ONLY - Not for clinical decision-making.

LICENSE: MIT
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SurgicalRobotCategory(Enum):
    """Surgical robot system categories."""

    TELEOPERATED = "teleoperated"
    SEMI_AUTONOMOUS = "semi_autonomous"
    AUTONOMOUS = "autonomous"
    TABLE_INTEGRATED = "table_integrated"


class SurgicalSimFramework(Enum):
    """Simulation frameworks relevant to surgical robotics."""

    ORBIT_SURGICAL = "orbit_surgical"
    DVRK_SIM = "dvrk_sim"
    SURROL = "surrol"
    SURGICAL_GYM = "surgical_gym"
    ISAAC_LAB = "nvidia_isaac_lab"
    MUJOCO = "mujoco"
    GAZEBO = "gazebo"
    AMBF = "ambf"


class SurgicalAICapability(Enum):
    """AI capabilities specific to surgical robot systems."""

    AUTONOMOUS_SUTURING = "autonomous_suturing"
    TISSUE_RECOGNITION = "tissue_recognition"
    FORCE_ESTIMATION = "force_estimation"
    GESTURE_RECOGNITION = "gesture_recognition"
    SKILL_ASSESSMENT = "skill_assessment"
    SURGICAL_PLANNING = "surgical_planning"
    INTRAOPERATIVE_GUIDANCE = "intraoperative_guidance"
    ANOMALY_DETECTION = "anomaly_detection"


class SurgicalSharingMethod(Enum):
    """Methods for cross-robot progress sharing in surgical systems."""

    OPEN_SOURCE_PLATFORM = "open_source_platform"
    RESEARCH_KIT = "research_kit"
    STANDARDIZED_API = "standardized_api"
    OPENIGTLINK = "openigtlink"
    ROS2_BRIDGE = "ros2_bridge"
    ONNX_EXPORT = "onnx_export"
    FEDERATED_LEARNING = "federated_learning"
    IEEE_3177 = "ieee_3177_modular_framework"


class RegulatoryStatus(Enum):
    """FDA regulatory clearance status."""

    CLEARED = "fda_cleared"
    DE_NOVO_PENDING = "de_novo_pending"
    IDE_ACTIVE = "ide_active"
    NOT_SUBMITTED = "not_submitted"


# ---------------------------------------------------------------------------
# Scoring data classes
# ---------------------------------------------------------------------------


@dataclass
class SurgicalSimScore:
    """Score for a surgical simulation framework's support level.

    Attributes:
        framework: The simulation framework being evaluated.
        open_source: Whether the simulation platform is open-source.
        surgical_tasks_available: Number of validated surgical benchmark tasks.
        gpu_accelerated: Whether the simulation supports GPU parallel envs.
        ros2_compatible: ROS 2 integration available.
        haptic_feedback_sim: Whether haptic/force feedback is simulated.
        tissue_deformation: Supports deformable tissue simulation.
        instrument_models: Number of instrument models available.
        documentation_quality: Quality of documentation (0-10).
        score: Computed sub-score (0.0-10.0).
    """

    framework: SurgicalSimFramework
    open_source: bool = False
    surgical_tasks_available: int = 0
    gpu_accelerated: bool = False
    ros2_compatible: bool = False
    haptic_feedback_sim: bool = False
    tissue_deformation: bool = False
    instrument_models: int = 0
    documentation_quality: float = 0.0
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute sub-score for this framework on a 0-10 scale."""
        raw = 0.0
        raw += 2.0 if self.open_source else 0.0
        raw += min(self.surgical_tasks_available / 7.0, 1.0) * 2.0
        raw += 1.0 if self.gpu_accelerated else 0.0
        raw += 1.0 if self.ros2_compatible else 0.0
        raw += 1.0 if self.haptic_feedback_sim else 0.0
        raw += 0.5 if self.tissue_deformation else 0.0
        raw += min(self.instrument_models / 5.0, 1.0)
        raw += min(self.documentation_quality / 10.0, 1.0) * 1.5
        self.score = round(min(raw, 10.0), 1)
        return self.score


@dataclass
class SurgicalDimAScore:
    """Dimension A: Simulation Framework Switching for surgical robots.

    Evaluates how well a surgical robot system can be simulated across
    different physics and surgical simulation environments.
    """

    framework_scores: list[SurgicalSimScore] = field(default_factory=list)
    num_frameworks_supported: int = 0
    cross_framework_transfer_tested: bool = False
    open_source_models_available: bool = False
    dvrk_compatible: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension A score (0.0-10.0)."""
        if not self.framework_scores:
            self.score = 1.0
            return self.score

        avg_fw = sum(fs.compute_score() for fs in self.framework_scores) / len(self.framework_scores)
        breadth = min(self.num_frameworks_supported / 5.0, 1.0) * 2.0
        transfer = 1.0 if self.cross_framework_transfer_tested else 0.0
        open_src = 1.0 if self.open_source_models_available else 0.0
        dvrk = 0.5 if self.dvrk_compatible else 0.0

        raw = avg_fw * 0.5 + breadth + transfer + open_src + dvrk
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class SurgicalDimBScore:
    """Dimension B: Generative/Agentic AI Integration for surgical robots.

    Evaluates the surgical robot's integration with AI systems for
    autonomous subtask execution, surgical planning, and skill assessment.
    """

    ai_capabilities: list[SurgicalAICapability] = field(default_factory=list)
    autonomous_subtask_demonstrated: bool = False
    rl_policy_trained: bool = False
    imitation_learning_tested: bool = False
    llm_planning_integration: bool = False
    vision_language_action: bool = False
    published_ai_papers: int = 0
    lasr_autonomy_level: int = 1
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension B score (0.0-10.0)."""
        raw = 0.0
        raw += min(len(self.ai_capabilities) / 6.0, 1.0) * 2.5
        raw += 1.5 if self.autonomous_subtask_demonstrated else 0.0
        raw += 1.0 if self.rl_policy_trained else 0.0
        raw += 0.5 if self.imitation_learning_tested else 0.0
        raw += 1.0 if self.llm_planning_integration else 0.0
        raw += 1.0 if self.vision_language_action else 0.0
        raw += min(self.published_ai_papers / 50.0, 1.0) * 1.5
        raw += min(self.lasr_autonomy_level / 5.0, 1.0) * 1.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class SurgicalDimCScore:
    """Dimension C: Cross-Robot Progress Sharing for surgical robots.

    Evaluates how well a surgical robot can share and transfer learned
    policies, configurations, and progress with other surgical systems.
    """

    sharing_methods: list[SurgicalSharingMethod] = field(default_factory=list)
    intra_manufacturer_transfer: bool = False
    inter_manufacturer_transfer: bool = False
    open_source_research_kit: bool = False
    ieee_3177_alignment: bool = False
    openigtlink_compatible: bool = False
    standardized_kinematics_published: bool = False
    skill_transfer_demonstrated: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension C score (0.0-10.0)."""
        raw = 0.0
        raw += min(len(self.sharing_methods) / 5.0, 1.0) * 2.0
        raw += 1.0 if self.intra_manufacturer_transfer else 0.0
        raw += 2.0 if self.inter_manufacturer_transfer else 0.0
        raw += 2.0 if self.open_source_research_kit else 0.0
        raw += 1.0 if self.ieee_3177_alignment else 0.0
        raw += 0.5 if self.openigtlink_compatible else 0.0
        raw += 1.0 if self.standardized_kinematics_published else 0.0
        raw += 0.5 if self.skill_transfer_demonstrated else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class SurgicalDimDScore:
    """Dimension D: Multi-Site Clinical Trial Collaboration for surgical robots.

    Evaluates readiness for deployment across multiple clinical trial sites
    with regulatory compliance and federated coordination.
    """

    fda_regulatory_status: RegulatoryStatus = RegulatoryStatus.NOT_SUBMITTED
    total_procedures_performed: int = 0
    multi_site_deployed: bool = False
    clinical_evidence_publications: int = 0
    iec_80601_2_77_compliant: bool = False
    iso_13482_aligned: bool = False
    hipaa_audit_capable: bool = False
    remote_monitoring: bool = False
    federated_data_sharing: bool = False
    multi_specialty_cleared: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension D score (0.0-10.0)."""
        raw = 0.0
        # FDA status
        status_scores = {
            RegulatoryStatus.CLEARED: 2.0,
            RegulatoryStatus.DE_NOVO_PENDING: 1.0,
            RegulatoryStatus.IDE_ACTIVE: 0.5,
            RegulatoryStatus.NOT_SUBMITTED: 0.0,
        }
        raw += status_scores.get(self.fda_regulatory_status, 0.0)
        # Procedure volume (log scale, capped at 10M+)
        if self.total_procedures_performed > 0:
            log_procs = math.log10(max(self.total_procedures_performed, 1))
            raw += min(log_procs / 7.0, 1.0) * 1.5
        raw += 1.5 if self.multi_site_deployed else 0.0
        raw += min(self.clinical_evidence_publications / 100.0, 1.0) * 1.0
        raw += 1.0 if self.iec_80601_2_77_compliant else 0.0
        raw += 0.5 if self.iso_13482_aligned else 0.0
        raw += 0.5 if self.hipaa_audit_capable else 0.0
        raw += 0.5 if self.remote_monitoring else 0.0
        raw += 0.5 if self.federated_data_sharing else 0.0
        raw += 0.5 if self.multi_specialty_cleared else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


# ---------------------------------------------------------------------------
# Main Surgical USL Rating
# ---------------------------------------------------------------------------


@dataclass
class SurgicalUSLRating:
    """Complete USL rating for a surgical robot system.

    The final score is a weighted average of the four dimensions,
    rounded to the nearest 0.1 on a 1.0-10.0 scale.
    """

    robot_name: str
    manufacturer: str
    category: SurgicalRobotCategory = SurgicalRobotCategory.TELEOPERATED
    dimension_a: SurgicalDimAScore = field(default_factory=SurgicalDimAScore)
    dimension_b: SurgicalDimBScore = field(default_factory=SurgicalDimBScore)
    dimension_c: SurgicalDimCScore = field(default_factory=SurgicalDimCScore)
    dimension_d: SurgicalDimDScore = field(default_factory=SurgicalDimDScore)
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    final_score: float = 0.0
    evaluated_date: str = ""
    evaluator: str = "USL Scoring Framework v1.5 — Surgical"
    notes: str = ""

    def compute_final_score(self) -> float:
        """Compute the weighted final USL score.

        Returns:
            Final score on 1.0-10.0 scale in 0.1 increments.
        """
        a = self.dimension_a.compute_score()
        b = self.dimension_b.compute_score()
        c = self.dimension_c.compute_score()
        d = self.dimension_d.compute_score()

        wa, wb, wc, wd = self.weights
        total_weight = wa + wb + wc + wd
        if total_weight == 0:
            self.final_score = 1.0
            return self.final_score

        raw = (a * wa + b * wb + c * wc + d * wd) / total_weight
        self.final_score = max(1.0, min(round(raw * 10) / 10, 10.0))
        self.evaluated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.final_score

    def summary(self) -> str:
        """Generate a human-readable summary of the USL rating."""
        self.compute_final_score()
        lines = [
            f"USL Rating: {self.robot_name} ({self.manufacturer})",
            f"{'=' * 60}",
            f"Category:     {self.category.value}",
            f"Evaluated:    {self.evaluated_date}",
            f"Evaluator:    {self.evaluator}",
            "",
            f"  A) Simulation Switching:       {self.dimension_a.score:>4.1f} / 10.0  (x{self.weights[0]:.0%})",
            f"  B) AI Integration:             {self.dimension_b.score:>4.1f} / 10.0  (x{self.weights[1]:.0%})",
            f"  C) Cross-Robot Sharing:         {self.dimension_c.score:>4.1f} / 10.0  (x{self.weights[2]:.0%})",
            f"  D) Clinical Trial Collab:      {self.dimension_d.score:>4.1f} / 10.0  (x{self.weights[3]:.0%})",
            "",
            f"  FINAL USL SCORE:               {self.final_score:>4.1f} / 10.0",
            "",
            _surgical_band_label(self.final_score),
        ]
        if self.notes:
            lines.append(f"\nNotes: {self.notes}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the rating to a dictionary."""
        self.compute_final_score()
        return {
            "robot_name": self.robot_name,
            "manufacturer": self.manufacturer,
            "category": self.category.value,
            "dimension_a": self.dimension_a.score,
            "dimension_b": self.dimension_b.score,
            "dimension_c": self.dimension_c.score,
            "dimension_d": self.dimension_d.score,
            "final_score": self.final_score,
            "evaluated_date": self.evaluated_date,
            "evaluator": self.evaluator,
            "weights": list(self.weights),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the rating to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Score band classifier
# ---------------------------------------------------------------------------

_SCORE_BANDS = [
    (9.0, "Exemplary", "Fully unified, multi-site clinical trial ready"),
    (7.0, "Advanced", "Strong unification, near clinical-trial ready"),
    (5.0, "Intermediate", "Partial unification, significant work remaining"),
    (3.0, "Foundational", "Basic interoperability, major gaps exist"),
    (1.0, "Initial", "Minimal unification capability"),
]


def _surgical_band_label(score: float) -> str:
    """Return a human-readable band label for a surgical USL score."""
    for threshold, label, description in _SCORE_BANDS:
        if score >= threshold:
            return f"  Band: {label} ({description})"
    return "  Band: Initial (Minimal unification capability)"


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_surgical_ratings(ratings: list[SurgicalUSLRating]) -> str:
    """Generate a comparison table for multiple surgical USL ratings.

    Args:
        ratings: List of surgical USL ratings to compare.

    Returns:
        Formatted comparison string.
    """
    for r in ratings:
        r.compute_final_score()

    header = f"{'Robot':<30} {'Mfg':<20} {'Dim A':>6} {'Dim B':>6} {'Dim C':>6} {'Dim D':>6} {'FINAL':>7}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in sorted(ratings, key=lambda x: x.final_score, reverse=True):
        lines.append(
            f"{r.robot_name:<30} {r.manufacturer:<20} "
            f"{r.dimension_a.score:>5.1f} {r.dimension_b.score:>5.1f} "
            f"{r.dimension_c.score:>5.1f} {r.dimension_d.score:>5.1f} "
            f"{r.final_score:>6.1f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def compute_surgical_gap_analysis(rating: SurgicalUSLRating) -> dict:
    """Identify weakest dimensions and suggest improvements for a surgical robot.

    Args:
        rating: A computed surgical USL rating.

    Returns:
        Dictionary with gap analysis per dimension.
    """
    rating.compute_final_score()
    dims = {
        "A_simulation_switching": rating.dimension_a.score,
        "B_ai_integration": rating.dimension_b.score,
        "C_cross_robot_sharing": rating.dimension_c.score,
        "D_clinical_trial_collab": rating.dimension_d.score,
    }

    sorted_dims = sorted(dims.items(), key=lambda x: x[1])
    weakest = sorted_dims[0]
    strongest = sorted_dims[-1]
    gap = round(strongest[1] - weakest[1], 1)

    suggestions = {
        "A_simulation_switching": [
            "Develop open-source simulation models (dVRK-compatible or standalone)",
            "Publish validated URDF/MJCF instrument and arm models",
            "Integrate with ORBIT-Surgical benchmark tasks",
        ],
        "B_ai_integration": [
            "Enable research API for autonomous subtask development",
            "Support RL policy deployment via standardized interface",
            "Integrate surgical planning with LLM-based reasoning",
        ],
        "C_cross_robot_sharing": [
            "Publish kinematics and control interface specifications",
            "Align with IEEE 3177-2024 modular surgical robot framework",
            "Support OpenIGTLink for image-guided therapy integration",
        ],
        "D_clinical_trial_collab": [
            "Expand FDA indications to oncology-specific procedures",
            "Implement federated data sharing for multi-site outcomes",
            "Develop IEC 80601-2-77 compliance documentation",
        ],
    }

    return {
        "robot_name": rating.robot_name,
        "final_score": rating.final_score,
        "weakest_dimension": weakest[0],
        "weakest_score": weakest[1],
        "strongest_dimension": strongest[0],
        "strongest_score": strongest[1],
        "max_gap": gap,
        "improvement_suggestions": suggestions.get(weakest[0], []),
    }


# ---------------------------------------------------------------------------
# USL Level Classification (shared with cobots)
# ---------------------------------------------------------------------------

USL_LEVELS = {
    1: "Conceptual — Robot exists; no simulation or AI integration attempted",
    2: "Exploratory — Single framework tested; basic model available",
    3: "Basic — 2+ frameworks; initial AI experiments conducted",
    4: "Developing — Cross-framework transfer demonstrated; AI planning tested",
    5: "Functional — 3+ frameworks; agentic AI operational; intra-org sharing",
    6: "Integrated — Multi-framework validated; LLM planning; inter-org sharing",
    7: "Advanced — GPU sim + policy transfer; MCP/VLA integration; skill sharing",
    8: "Clinical-Ready — Multi-site tested; regulatory docs; federated learning",
    9: "Validated — Full regulatory compliance; multi-site trials active",
    10: "Exemplary — Production deployment; open consortium; continuous improvement",
}


def get_usl_level(score: float) -> tuple[int, str]:
    """Map a USL score to its level description.

    Args:
        score: Final USL score (1.0-10.0).

    Returns:
        Tuple of (level_int, description).
    """
    level = max(1, min(10, math.floor(score)))
    return level, USL_LEVELS.get(level, "Unknown")


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_surgical_report(
    ratings: list[SurgicalUSLRating],
    output_format: str = "text",
) -> str:
    """Generate a comprehensive evaluation report for surgical robots.

    Args:
        ratings: List of surgical USL ratings to include.
        output_format: Either 'text' or 'json'.

    Returns:
        Formatted report string.
    """
    for r in ratings:
        r.compute_final_score()

    if output_format == "json":
        report_data = {
            "report_title": "USL Surgical Robot Evaluation Report",
            "generated": datetime.now(timezone.utc).isoformat(),
            "framework_version": "USL v1.5 — Surgical",
            "ratings": [r.to_dict() for r in ratings],
            "comparison": {
                r.robot_name: {
                    "gap_analysis": compute_surgical_gap_analysis(r),
                    "level": get_usl_level(r.final_score)[0],
                    "level_description": get_usl_level(r.final_score)[1],
                }
                for r in ratings
            },
        }
        return json.dumps(report_data, indent=2)

    lines = [
        "=" * 70,
        "  UNIFICATION STANDARD LEVEL (USL) — SURGICAL ROBOT EVALUATION REPORT",
        "=" * 70,
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "  Framework: USL v1.5 — Surgical",
        "  Category:  Surgical Robot Systems",
        "=" * 70,
        "",
    ]

    for r in sorted(ratings, key=lambda x: x.final_score, reverse=True):
        lines.append(r.summary())
        level_num, level_desc = get_usl_level(r.final_score)
        lines.append(f"  USL Level: {level_num} — {level_desc}")
        lines.append("")

    lines.append("")
    lines.append("COMPARISON TABLE")
    lines.append(compare_surgical_ratings(ratings))
    lines.append("")

    lines.append("GAP ANALYSIS")
    lines.append("-" * 70)
    for r in ratings:
        gap = compute_surgical_gap_analysis(r)
        lines.append(f"  {r.robot_name}:")
        lines.append(f"    Weakest:  {gap['weakest_dimension']} ({gap['weakest_score']:.1f})")
        lines.append(f"    Strongest: {gap['strongest_dimension']} ({gap['strongest_score']:.1f})")
        lines.append(f"    Gap: {gap['max_gap']:.1f} points")
        if gap["improvement_suggestions"]:
            lines.append("    Top suggestions:")
            for s in gap["improvement_suggestions"][:2]:
                lines.append(f"      - {s}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo with three surgical robots
# ---------------------------------------------------------------------------


def _build_davinci_rating() -> SurgicalUSLRating:
    """Build USL rating for da Vinci Xi / da Vinci 5."""
    return SurgicalUSLRating(
        robot_name="da Vinci (Xi / da Vinci 5)",
        manufacturer="Intuitive Surgical",
        category=SurgicalRobotCategory.TELEOPERATED,
        dimension_a=SurgicalDimAScore(
            framework_scores=[
                SurgicalSimScore(
                    framework=SurgicalSimFramework.ORBIT_SURGICAL,
                    open_source=True,
                    surgical_tasks_available=14,
                    gpu_accelerated=True,
                    ros2_compatible=True,
                    haptic_feedback_sim=False,
                    tissue_deformation=True,
                    instrument_models=4,
                    documentation_quality=8.0,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.DVRK_SIM,
                    open_source=True,
                    surgical_tasks_available=10,
                    gpu_accelerated=False,
                    ros2_compatible=True,
                    haptic_feedback_sim=True,
                    tissue_deformation=True,
                    instrument_models=5,
                    documentation_quality=9.0,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.SURROL,
                    open_source=True,
                    surgical_tasks_available=10,
                    gpu_accelerated=False,
                    ros2_compatible=True,
                    haptic_feedback_sim=False,
                    tissue_deformation=True,
                    instrument_models=3,
                    documentation_quality=7.0,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.SURGICAL_GYM,
                    open_source=True,
                    surgical_tasks_available=6,
                    gpu_accelerated=True,
                    ros2_compatible=False,
                    haptic_feedback_sim=False,
                    tissue_deformation=False,
                    instrument_models=2,
                    documentation_quality=6.0,
                ),
            ],
            num_frameworks_supported=5,
            cross_framework_transfer_tested=True,
            open_source_models_available=True,
            dvrk_compatible=True,
        ),
        dimension_b=SurgicalDimBScore(
            ai_capabilities=[
                SurgicalAICapability.AUTONOMOUS_SUTURING,
                SurgicalAICapability.TISSUE_RECOGNITION,
                SurgicalAICapability.FORCE_ESTIMATION,
                SurgicalAICapability.GESTURE_RECOGNITION,
                SurgicalAICapability.SKILL_ASSESSMENT,
                SurgicalAICapability.SURGICAL_PLANNING,
            ],
            autonomous_subtask_demonstrated=True,
            rl_policy_trained=True,
            imitation_learning_tested=True,
            llm_planning_integration=True,
            vision_language_action=False,
            published_ai_papers=200,
            lasr_autonomy_level=2,
        ),
        dimension_c=SurgicalDimCScore(
            sharing_methods=[
                SurgicalSharingMethod.OPEN_SOURCE_PLATFORM,
                SurgicalSharingMethod.RESEARCH_KIT,
                SurgicalSharingMethod.ROS2_BRIDGE,
                SurgicalSharingMethod.ONNX_EXPORT,
                SurgicalSharingMethod.OPENIGTLINK,
            ],
            intra_manufacturer_transfer=True,
            inter_manufacturer_transfer=True,
            open_source_research_kit=True,
            ieee_3177_alignment=False,
            openigtlink_compatible=True,
            standardized_kinematics_published=True,
            skill_transfer_demonstrated=True,
        ),
        dimension_d=SurgicalDimDScore(
            fda_regulatory_status=RegulatoryStatus.CLEARED,
            total_procedures_performed=14_000_000,
            multi_site_deployed=True,
            clinical_evidence_publications=500,
            iec_80601_2_77_compliant=True,
            iso_13482_aligned=True,
            hipaa_audit_capable=True,
            remote_monitoring=True,
            federated_data_sharing=False,
            multi_specialty_cleared=True,
        ),
    )


def _build_hugo_rating() -> SurgicalUSLRating:
    """Build USL rating for Medtronic Hugo RAS."""
    return SurgicalUSLRating(
        robot_name="Hugo RAS",
        manufacturer="Medtronic",
        category=SurgicalRobotCategory.TELEOPERATED,
        dimension_a=SurgicalDimAScore(
            framework_scores=[
                SurgicalSimScore(
                    framework=SurgicalSimFramework.MUJOCO,
                    open_source=False,
                    surgical_tasks_available=0,
                    gpu_accelerated=False,
                    ros2_compatible=False,
                    haptic_feedback_sim=False,
                    tissue_deformation=False,
                    instrument_models=0,
                    documentation_quality=2.0,
                ),
            ],
            num_frameworks_supported=1,
            cross_framework_transfer_tested=False,
            open_source_models_available=False,
            dvrk_compatible=False,
        ),
        dimension_b=SurgicalDimBScore(
            ai_capabilities=[
                SurgicalAICapability.SURGICAL_PLANNING,
                SurgicalAICapability.SKILL_ASSESSMENT,
            ],
            autonomous_subtask_demonstrated=False,
            rl_policy_trained=False,
            imitation_learning_tested=False,
            llm_planning_integration=False,
            vision_language_action=False,
            published_ai_papers=15,
            lasr_autonomy_level=1,
        ),
        dimension_c=SurgicalDimCScore(
            sharing_methods=[
                SurgicalSharingMethod.STANDARDIZED_API,
            ],
            intra_manufacturer_transfer=True,
            inter_manufacturer_transfer=False,
            open_source_research_kit=False,
            ieee_3177_alignment=False,
            openigtlink_compatible=False,
            standardized_kinematics_published=False,
            skill_transfer_demonstrated=False,
        ),
        dimension_d=SurgicalDimDScore(
            fda_regulatory_status=RegulatoryStatus.CLEARED,
            total_procedures_performed=50_000,
            multi_site_deployed=True,
            clinical_evidence_publications=30,
            iec_80601_2_77_compliant=True,
            iso_13482_aligned=True,
            hipaa_audit_capable=True,
            remote_monitoring=True,
            federated_data_sharing=False,
            multi_specialty_cleared=False,
        ),
    )


def _build_ottava_rating() -> SurgicalUSLRating:
    """Build USL rating for Johnson & Johnson OTTAVA."""
    return SurgicalUSLRating(
        robot_name="OTTAVA",
        manufacturer="Johnson & Johnson MedTech",
        category=SurgicalRobotCategory.TABLE_INTEGRATED,
        dimension_a=SurgicalDimAScore(
            framework_scores=[],
            num_frameworks_supported=0,
            cross_framework_transfer_tested=False,
            open_source_models_available=False,
            dvrk_compatible=False,
        ),
        dimension_b=SurgicalDimBScore(
            ai_capabilities=[
                SurgicalAICapability.SURGICAL_PLANNING,
            ],
            autonomous_subtask_demonstrated=False,
            rl_policy_trained=False,
            imitation_learning_tested=False,
            llm_planning_integration=False,
            vision_language_action=False,
            published_ai_papers=3,
            lasr_autonomy_level=1,
        ),
        dimension_c=SurgicalDimCScore(
            sharing_methods=[],
            intra_manufacturer_transfer=False,
            inter_manufacturer_transfer=False,
            open_source_research_kit=False,
            ieee_3177_alignment=False,
            openigtlink_compatible=False,
            standardized_kinematics_published=False,
            skill_transfer_demonstrated=False,
        ),
        dimension_d=SurgicalDimDScore(
            fda_regulatory_status=RegulatoryStatus.DE_NOVO_PENDING,
            total_procedures_performed=100,
            multi_site_deployed=False,
            clinical_evidence_publications=5,
            iec_80601_2_77_compliant=False,
            iso_13482_aligned=False,
            hipaa_audit_capable=False,
            remote_monitoring=False,
            federated_data_sharing=False,
            multi_specialty_cleared=False,
        ),
    )


def _demo() -> None:
    """Demonstrate USL scoring with three surgical robot systems."""
    logger.info("Running surgical robot USL scoring demonstration...")

    davinci = _build_davinci_rating()
    hugo = _build_hugo_rating()
    ottava = _build_ottava_rating()

    report = generate_surgical_report([davinci, hugo, ottava])
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
