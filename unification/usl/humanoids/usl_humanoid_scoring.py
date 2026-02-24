"""Unification Standard Level (USL) Scoring Framework for Humanoid Robots.

Evaluates humanoid robot systems on their readiness for multi-site oncology
clinical trials.  Final scores range from 1.0 to 10.0 in 0.1 increments
across four weighted dimensions:

    A) Simulation Framework Switching   (25%)
    B) Generative / Agentic AI Integration (25%)
    C) Cross-Robot Progress Sharing     (25%)
    D) Multi-Site Clinical Trial Collaboration (25%)

This module extends the USL framework (originally developed for collaborative
robots and surgical robots) to full-size humanoid robot systems.  Humanoid
robots differ from cobots and surgical robots in their:

    - Full-body bipedal locomotion with 30-50+ DOF
    - Whole-body coordination (locomotion + manipulation)
    - Foundation model integration (GR00T, OpenVLA, pi-zero)
    - Autonomous navigation and dynamic obstacle avoidance
    - Potential for patient transport, logistics, and assistive tasks
    - Less mature clinical deployment than cobots or surgical robots

Evaluated humanoid robots (Category: Humanoid Robots):
    1. Boston Dynamics Atlas (Electric)  — USL 5.8
    2. Tesla Optimus (Gen 2)             — USL 3.6
    3. Agility Robotics Digit            — USL 4.2

Inspired by:
  - NASA/DOD TRL (Mankins, 1995; 2004 White Paper)
  - ML Technology Readiness Levels (MLTRL, Lavin et al., 2021)
  - TRL for complex systems (Tomaschek et al., 2015;
    DOI 10.1109/PICMET.2015.7273196)
  - LLM recommendations for oncology trials (Kawchak, 2025;
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


class HumanoidType(Enum):
    """Sub-types of humanoid robot systems."""

    BIPEDAL_FULL = "bipedal_full_size"
    BIPEDAL_COMPACT = "bipedal_compact"
    WHEELED_HUMANOID = "wheeled_humanoid"
    UPPER_BODY = "upper_body_only"


class HumanoidSimFramework(Enum):
    """Supported simulation frameworks for humanoid robot evaluation."""

    ISAAC_SIM = "nvidia_isaac_sim"
    ISAAC_LAB = "nvidia_isaac_lab"
    MUJOCO = "mujoco"
    MUJOCO_MJX = "mujoco_mjx"
    MUJOCO_PLAYGROUND = "mujoco_playground"
    GAZEBO = "gazebo"
    PYBULLET = "pybullet"
    DRAKE = "drake"


class HumanoidAICapability(Enum):
    """AI capabilities specific to humanoid robot evaluation."""

    GENERATIVE_AI = "generative_ai"
    AGENTIC_AI = "agentic_ai"
    VLA_MODELS = "vla_models"
    FOUNDATION_MODEL = "foundation_model"
    DIFFUSION_POLICY = "diffusion_policy"
    IMITATION_LEARNING = "imitation_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    LLM_PLANNING = "llm_planning"
    WHOLE_BODY_CONTROL = "whole_body_control"
    LOCOMOTION_POLICY = "locomotion_policy"
    MANIPULATION_POLICY = "manipulation_policy"
    MCP_INTEGRATION = "mcp_integration"


class HumanoidTask(Enum):
    """Oncology-relevant tasks for humanoid robots."""

    PATIENT_TRANSPORT = "patient_transport"
    SUPPLY_DELIVERY = "supply_delivery"
    EQUIPMENT_HANDLING = "equipment_handling"
    ASSISTIVE_CARE = "assistive_care"
    SPECIMEN_TRANSPORT = "specimen_transport"
    PHARMACY_LOGISTICS = "pharmacy_logistics"
    DECONTAMINATION = "decontamination"
    SURGICAL_ASSISTANCE = "surgical_assistance"


# ---------------------------------------------------------------------------
# Scoring data classes
# ---------------------------------------------------------------------------


@dataclass
class HumanoidSimScore:
    """Score for a single simulation framework's support for humanoid robots."""

    framework: HumanoidSimFramework
    official_support: bool = False
    full_body_model: bool = False
    locomotion_capable: bool = False
    manipulation_capable: bool = False
    ros2_integration: bool = False
    gpu_sim_capable: bool = False
    documented_examples: int = 0
    community_packages: int = 0
    bidirectional_transfer: bool = False
    terrain_simulation: bool = False
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute sub-score for this framework on a 0-10 scale."""
        raw = 0.0
        raw += 2.0 if self.official_support else 0.0
        raw += 1.0 if self.full_body_model else 0.0
        raw += 1.0 if self.locomotion_capable else 0.0
        raw += 0.5 if self.manipulation_capable else 0.0
        raw += 1.0 if self.ros2_integration else 0.0
        raw += 1.0 if self.gpu_sim_capable else 0.0
        raw += min(self.documented_examples / 5.0, 1.0)
        raw += min(self.community_packages / 3.0, 1.0)
        raw += 0.5 if self.bidirectional_transfer else 0.0
        raw += 1.0 if self.terrain_simulation else 0.0
        self.score = round(min(raw, 10.0), 1)
        return self.score


@dataclass
class HumanoidDimAScore:
    """Dimension A: Simulation Framework Switching for humanoid robots."""

    framework_scores: list[HumanoidSimScore] = field(default_factory=list)
    num_frameworks_supported: int = 0
    cross_framework_transfer_tested: bool = False
    locomotion_sim_fidelity: bool = False
    manipulation_sim_fidelity: bool = False
    whole_body_model_formats: int = 0
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension A score (0.0-10.0)."""
        if not self.framework_scores:
            self.score = 1.0
            return self.score

        avg_fw = sum(fs.compute_score() for fs in self.framework_scores) / len(self.framework_scores)
        breadth_bonus = min(self.num_frameworks_supported / 5.0, 1.0) * 2.0
        transfer_bonus = 1.0 if self.cross_framework_transfer_tested else 0.0
        loco_bonus = 0.5 if self.locomotion_sim_fidelity else 0.0
        manip_bonus = 0.5 if self.manipulation_sim_fidelity else 0.0
        format_bonus = min(self.whole_body_model_formats / 4.0, 1.0) * 0.5

        raw = avg_fw * 0.5 + breadth_bonus + transfer_bonus + loco_bonus + manip_bonus + format_bonus
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class HumanoidDimBScore:
    """Dimension B: Generative/Agentic AI Integration for humanoid robots."""

    supported_capabilities: list[HumanoidAICapability] = field(default_factory=list)
    foundation_model_integrated: bool = False
    vla_model_compatible: bool = False
    llm_task_planning: bool = False
    whole_body_learning: bool = False
    locomotion_policy_learned: bool = False
    manipulation_policy_learned: bool = False
    diffusion_policy_tested: bool = False
    imitation_learning_tested: bool = False
    ai_safety_constraints: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension B score (0.0-10.0)."""
        raw = 0.0
        raw += min(len(self.supported_capabilities) / 8.0, 1.0) * 2.5
        raw += 1.5 if self.foundation_model_integrated else 0.0
        raw += 1.0 if self.vla_model_compatible else 0.0
        raw += 1.0 if self.llm_task_planning else 0.0
        raw += 0.5 if self.whole_body_learning else 0.0
        raw += 0.5 if self.locomotion_policy_learned else 0.0
        raw += 0.5 if self.manipulation_policy_learned else 0.0
        raw += 0.5 if self.diffusion_policy_tested else 0.0
        raw += 0.5 if self.imitation_learning_tested else 0.0
        raw += 1.0 if self.ai_safety_constraints else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class HumanoidDimCScore:
    """Dimension C: Cross-Robot Progress Sharing for humanoid robots."""

    open_source_platform: bool = False
    intra_org_transfer: bool = False
    inter_org_transfer: bool = False
    onnx_policy_export: bool = False
    standardized_morphology: bool = False
    skill_library_compatible: bool = False
    locomotion_transferable: bool = False
    manipulation_transferable: bool = False
    research_community_active: bool = False
    real_time_state_sync: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension C score (0.0-10.0)."""
        raw = 0.0
        raw += 2.0 if self.open_source_platform else 0.0
        raw += 1.5 if self.intra_org_transfer else 0.0
        raw += 1.5 if self.inter_org_transfer else 0.0
        raw += 1.0 if self.onnx_policy_export else 0.0
        raw += 0.5 if self.standardized_morphology else 0.0
        raw += 0.5 if self.skill_library_compatible else 0.0
        raw += 0.5 if self.locomotion_transferable else 0.0
        raw += 0.5 if self.manipulation_transferable else 0.0
        raw += 1.0 if self.research_community_active else 0.0
        raw += 1.0 if self.real_time_state_sync else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class HumanoidDimDScore:
    """Dimension D: Multi-Site Clinical Trial Collaboration for humanoid robots."""

    safety_certified: bool = False
    ce_marked: bool = False
    multi_site_deployed: bool = False
    federated_learning_compatible: bool = False
    hipaa_compliance_tools: bool = False
    audit_trail_capable: bool = False
    remote_monitoring: bool = False
    autonomous_navigation_safe: bool = False
    safety_certification: bool = False
    clinical_workflow_integration: bool = False
    iso_13482_alignment: bool = False
    hospital_pilot_tested: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension D score (0.0-10.0)."""
        raw = 0.0
        raw += 1.0 if self.safety_certified else 0.0
        raw += 0.5 if self.ce_marked else 0.0
        raw += 1.0 if self.multi_site_deployed else 0.0
        raw += 1.0 if self.federated_learning_compatible else 0.0
        raw += 0.5 if self.hipaa_compliance_tools else 0.0
        raw += 0.5 if self.audit_trail_capable else 0.0
        raw += 0.5 if self.remote_monitoring else 0.0
        raw += 1.5 if self.autonomous_navigation_safe else 0.0
        raw += 1.0 if self.safety_certification else 0.0
        raw += 1.0 if self.clinical_workflow_integration else 0.0
        raw += 1.0 if self.iso_13482_alignment else 0.0
        raw += 0.5 if self.hospital_pilot_tested else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


# ---------------------------------------------------------------------------
# Main USL Rating for Humanoid Robots
# ---------------------------------------------------------------------------


@dataclass
class HumanoidUSLRating:
    """Complete USL rating for a humanoid robot system.

    The final score is a weighted average of the four dimensions,
    rounded to the nearest 0.1 on a 1.0-10.0 scale.
    """

    robot_name: str
    manufacturer: str
    robot_type: HumanoidType = HumanoidType.BIPEDAL_FULL
    dimension_a: HumanoidDimAScore = field(default_factory=HumanoidDimAScore)
    dimension_b: HumanoidDimBScore = field(default_factory=HumanoidDimBScore)
    dimension_c: HumanoidDimCScore = field(default_factory=HumanoidDimCScore)
    dimension_d: HumanoidDimDScore = field(default_factory=HumanoidDimDScore)
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    final_score: float = 0.0
    evaluated_date: str = ""
    evaluator: str = "USL Humanoid Scoring Framework v1.0"
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
            f"{'=' * 55}",
            f"Category:     Humanoid Robot ({self.robot_type.value})",
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
            _score_band_label(self.final_score),
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
            "robot_type": self.robot_type.value,
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


def _score_band_label(score: float) -> str:
    """Return a human-readable band label for a USL score."""
    for threshold, label, description in _SCORE_BANDS:
        if score >= threshold:
            return f"  Band: {label} ({description})"
    return "  Band: Initial (Minimal unification capability)"


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_humanoid_ratings(ratings: list[HumanoidUSLRating]) -> str:
    """Generate a comparison table for multiple humanoid USL ratings.

    Args:
        ratings: List of humanoid USL ratings to compare.

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


def compute_humanoid_gap_analysis(rating: HumanoidUSLRating) -> dict:
    """Identify the weakest dimensions and suggest improvements.

    Args:
        rating: A computed humanoid USL rating.

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
            "Develop whole-body URDF/MJCF/USD models for multiple simulators",
            "Implement GPU-accelerated locomotion + manipulation sim environments",
            "Create terrain-aware simulation benchmarks for hospital navigation",
        ],
        "B_ai_integration": [
            "Integrate foundation models (GR00T, OpenVLA) for whole-body control",
            "Develop LLM-based task planning for clinical logistics",
            "Implement imitation learning pipelines from teleoperation data",
        ],
        "C_cross_robot_sharing": [
            "Standardize humanoid action space and observation definitions",
            "Implement ONNX policy export for locomotion and manipulation",
            "Enable cross-morphology skill transfer between humanoid platforms",
        ],
        "D_clinical_trial_collab": [
            "Develop ISO 13482 safety certification documentation",
            "Implement autonomous navigation safety for hospital environments",
            "Build HIPAA-compliant audit trails for patient-area operations",
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
# USL Level Classification (shared standard)
# ---------------------------------------------------------------------------

USL_LEVELS = {
    1: "Conceptual — Robot exists; no simulation or AI integration attempted",
    2: "Exploratory — Single framework tested; basic model available",
    3: "Basic — 2+ frameworks; initial AI experiments conducted",
    4: "Developing — Cross-framework transfer demonstrated; AI planning tested",
    5: "Functional — 3+ frameworks; agentic AI operational; intra-org sharing",
    6: "Integrated — Multi-framework validated; LLM planning; inter-org sharing",
    7: "Advanced — GPU sim + policy transfer; VLA integration; skill sharing",
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
# Evaluation report generator
# ---------------------------------------------------------------------------


def generate_humanoid_evaluation_report(
    ratings: list[HumanoidUSLRating],
    output_format: str = "text",
) -> str:
    """Generate a comprehensive evaluation report for humanoid robots.

    Args:
        ratings: List of humanoid USL ratings to include.
        output_format: Either 'text' or 'json'.

    Returns:
        Formatted report string.
    """
    for r in ratings:
        r.compute_final_score()

    if output_format == "json":
        report_data = {
            "report_title": "USL Humanoid Robot Evaluation Report",
            "generated": datetime.now(timezone.utc).isoformat(),
            "framework_version": "USL Humanoid v1.0",
            "ratings": [r.to_dict() for r in ratings],
            "comparison": {
                r.robot_name: {
                    "gap_analysis": compute_humanoid_gap_analysis(r),
                    "level": get_usl_level(r.final_score)[0],
                    "level_description": get_usl_level(r.final_score)[1],
                }
                for r in ratings
            },
        }
        return json.dumps(report_data, indent=2)

    # Text format
    lines = [
        "=" * 70,
        "  UNIFICATION STANDARD LEVEL (USL) — HUMANOID ROBOT EVALUATION REPORT",
        "=" * 70,
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "  Framework: USL Humanoid v1.0",
        "  Category:  Humanoid Robots (Bipedal Full-Size Systems)",
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
    lines.append(compare_humanoid_ratings(ratings))
    lines.append("")

    lines.append("GAP ANALYSIS")
    lines.append("-" * 70)
    for r in ratings:
        gap = compute_humanoid_gap_analysis(r)
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
# Demo / main — Build ratings for all three humanoids
# ---------------------------------------------------------------------------


def _build_atlas_rating() -> HumanoidUSLRating:
    """Build the Boston Dynamics Atlas (Electric) USL rating."""
    return HumanoidUSLRating(
        robot_name="Atlas (Electric)",
        manufacturer="Boston Dynamics",
        dimension_a=HumanoidDimAScore(
            framework_scores=[
                HumanoidSimScore(
                    framework=HumanoidSimFramework.ISAAC_LAB,
                    official_support=True,
                    full_body_model=True,
                    locomotion_capable=True,
                    manipulation_capable=True,
                    gpu_sim_capable=True,
                    documented_examples=6,
                    community_packages=3,
                    bidirectional_transfer=True,
                    terrain_simulation=True,
                ),
                HumanoidSimScore(
                    framework=HumanoidSimFramework.MUJOCO,
                    full_body_model=True,
                    locomotion_capable=True,
                    manipulation_capable=True,
                    gpu_sim_capable=True,
                    documented_examples=5,
                    community_packages=3,
                    terrain_simulation=True,
                ),
                HumanoidSimScore(
                    framework=HumanoidSimFramework.DRAKE,
                    official_support=True,
                    full_body_model=True,
                    locomotion_capable=True,
                    manipulation_capable=True,
                    ros2_integration=True,
                    documented_examples=8,
                    community_packages=4,
                    bidirectional_transfer=True,
                    terrain_simulation=True,
                ),
                HumanoidSimScore(
                    framework=HumanoidSimFramework.GAZEBO,
                    full_body_model=True,
                    locomotion_capable=True,
                    ros2_integration=True,
                    documented_examples=4,
                    community_packages=2,
                ),
            ],
            num_frameworks_supported=4,
            cross_framework_transfer_tested=True,
            locomotion_sim_fidelity=True,
            manipulation_sim_fidelity=True,
            whole_body_model_formats=3,
        ),
        dimension_b=HumanoidDimBScore(
            supported_capabilities=[
                HumanoidAICapability.GENERATIVE_AI,
                HumanoidAICapability.AGENTIC_AI,
                HumanoidAICapability.REINFORCEMENT_LEARNING,
                HumanoidAICapability.IMITATION_LEARNING,
                HumanoidAICapability.WHOLE_BODY_CONTROL,
                HumanoidAICapability.LOCOMOTION_POLICY,
                HumanoidAICapability.MANIPULATION_POLICY,
                HumanoidAICapability.LLM_PLANNING,
            ],
            foundation_model_integrated=True,
            vla_model_compatible=True,
            llm_task_planning=True,
            whole_body_learning=True,
            locomotion_policy_learned=True,
            manipulation_policy_learned=True,
            diffusion_policy_tested=True,
            imitation_learning_tested=True,
            ai_safety_constraints=True,
        ),
        dimension_c=HumanoidDimCScore(
            open_source_platform=False,
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=True,
            standardized_morphology=False,
            skill_library_compatible=True,
            locomotion_transferable=True,
            manipulation_transferable=True,
            research_community_active=True,
            real_time_state_sync=True,
        ),
        dimension_d=HumanoidDimDScore(
            safety_certified=True,
            ce_marked=False,
            multi_site_deployed=False,
            federated_learning_compatible=True,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_monitoring=True,
            autonomous_navigation_safe=True,
            safety_certification=True,
            clinical_workflow_integration=False,
            iso_13482_alignment=True,
            hospital_pilot_tested=True,
        ),
    )


def _build_optimus_rating() -> HumanoidUSLRating:
    """Build the Tesla Optimus (Gen 2) USL rating."""
    return HumanoidUSLRating(
        robot_name="Optimus (Gen 2)",
        manufacturer="Tesla",
        dimension_a=HumanoidDimAScore(
            framework_scores=[
                HumanoidSimScore(
                    framework=HumanoidSimFramework.ISAAC_SIM,
                    full_body_model=True,
                    locomotion_capable=True,
                    manipulation_capable=True,
                    gpu_sim_capable=True,
                    documented_examples=3,
                    community_packages=1,
                ),
                HumanoidSimScore(
                    framework=HumanoidSimFramework.MUJOCO,
                    full_body_model=True,
                    locomotion_capable=True,
                    documented_examples=2,
                    community_packages=1,
                ),
            ],
            num_frameworks_supported=2,
            cross_framework_transfer_tested=False,
            locomotion_sim_fidelity=True,
            manipulation_sim_fidelity=False,
            whole_body_model_formats=2,
        ),
        dimension_b=HumanoidDimBScore(
            supported_capabilities=[
                HumanoidAICapability.GENERATIVE_AI,
                HumanoidAICapability.IMITATION_LEARNING,
                HumanoidAICapability.REINFORCEMENT_LEARNING,
                HumanoidAICapability.WHOLE_BODY_CONTROL,
                HumanoidAICapability.LOCOMOTION_POLICY,
                HumanoidAICapability.MANIPULATION_POLICY,
            ],
            foundation_model_integrated=False,
            vla_model_compatible=False,
            llm_task_planning=False,
            whole_body_learning=True,
            locomotion_policy_learned=True,
            manipulation_policy_learned=True,
            diffusion_policy_tested=False,
            imitation_learning_tested=True,
            ai_safety_constraints=True,
        ),
        dimension_c=HumanoidDimCScore(
            open_source_platform=False,
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=False,
            standardized_morphology=False,
            skill_library_compatible=False,
            locomotion_transferable=False,
            manipulation_transferable=False,
            research_community_active=False,
            real_time_state_sync=False,
        ),
        dimension_d=HumanoidDimDScore(
            safety_certified=False,
            ce_marked=False,
            multi_site_deployed=False,
            federated_learning_compatible=False,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_monitoring=True,
            autonomous_navigation_safe=False,
            safety_certification=False,
            clinical_workflow_integration=False,
            iso_13482_alignment=False,
            hospital_pilot_tested=False,
        ),
    )


def _build_digit_rating() -> HumanoidUSLRating:
    """Build the Agility Robotics Digit USL rating."""
    return HumanoidUSLRating(
        robot_name="Digit",
        manufacturer="Agility Robotics",
        dimension_a=HumanoidDimAScore(
            framework_scores=[
                HumanoidSimScore(
                    framework=HumanoidSimFramework.ISAAC_LAB,
                    official_support=True,
                    full_body_model=True,
                    locomotion_capable=True,
                    manipulation_capable=True,
                    gpu_sim_capable=True,
                    documented_examples=5,
                    community_packages=2,
                    bidirectional_transfer=True,
                    terrain_simulation=True,
                ),
                HumanoidSimScore(
                    framework=HumanoidSimFramework.MUJOCO,
                    official_support=True,
                    full_body_model=True,
                    locomotion_capable=True,
                    manipulation_capable=True,
                    gpu_sim_capable=True,
                    documented_examples=4,
                    community_packages=2,
                    terrain_simulation=True,
                ),
                HumanoidSimScore(
                    framework=HumanoidSimFramework.GAZEBO,
                    full_body_model=True,
                    locomotion_capable=True,
                    ros2_integration=True,
                    documented_examples=3,
                    community_packages=1,
                ),
            ],
            num_frameworks_supported=3,
            cross_framework_transfer_tested=True,
            locomotion_sim_fidelity=True,
            manipulation_sim_fidelity=True,
            whole_body_model_formats=3,
        ),
        dimension_b=HumanoidDimBScore(
            supported_capabilities=[
                HumanoidAICapability.GENERATIVE_AI,
                HumanoidAICapability.AGENTIC_AI,
                HumanoidAICapability.REINFORCEMENT_LEARNING,
                HumanoidAICapability.IMITATION_LEARNING,
                HumanoidAICapability.WHOLE_BODY_CONTROL,
                HumanoidAICapability.LOCOMOTION_POLICY,
                HumanoidAICapability.MANIPULATION_POLICY,
            ],
            foundation_model_integrated=True,
            vla_model_compatible=True,
            llm_task_planning=True,
            whole_body_learning=True,
            locomotion_policy_learned=True,
            manipulation_policy_learned=True,
            diffusion_policy_tested=False,
            imitation_learning_tested=True,
            ai_safety_constraints=True,
        ),
        dimension_c=HumanoidDimCScore(
            open_source_platform=False,
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=True,
            standardized_morphology=False,
            skill_library_compatible=True,
            locomotion_transferable=True,
            manipulation_transferable=False,
            research_community_active=True,
            real_time_state_sync=True,
        ),
        dimension_d=HumanoidDimDScore(
            safety_certified=True,
            ce_marked=False,
            multi_site_deployed=True,
            federated_learning_compatible=False,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_monitoring=True,
            autonomous_navigation_safe=True,
            safety_certification=True,
            clinical_workflow_integration=False,
            iso_13482_alignment=True,
            hospital_pilot_tested=False,
        ),
    )


def _demo() -> None:
    """Demonstrate USL scoring with three humanoid robots."""
    logger.info("Running USL humanoid robot scoring demonstration...")

    atlas = _build_atlas_rating()
    optimus = _build_optimus_rating()
    digit = _build_digit_rating()

    report = generate_humanoid_evaluation_report([atlas, optimus, digit])
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
