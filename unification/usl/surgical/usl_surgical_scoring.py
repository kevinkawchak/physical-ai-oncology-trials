"""Unification Standard Level (USL) Scoring Framework for Surgical Robots.

Evaluates surgical robot systems on their readiness for multi-site oncology
clinical trials.  Final scores range from 1.0 to 10.0 in 0.1 increments
across four weighted dimensions:

    A) Simulation Framework Switching   (25%)
    B) Generative / Agentic AI Integration (25%)
    C) Cross-Robot Progress Sharing     (25%)
    D) Multi-Site Clinical Trial Collaboration (25%)

This module extends the USL framework (originally developed for collaborative
robots) to teleoperated surgical robot systems used in minimally invasive
surgery (MIS).  Surgical robots differ from cobots in their:

    - Master-slave teleoperation architecture
    - Sub-millimeter precision requirements (< 0.1 mm)
    - Instrument-level DOF (wrist articulation beyond arm DOF)
    - Stringent FDA/CE regulatory pathways (Class II/III medical devices)
    - Sterile field and clinical workflow integration requirements

Evaluated surgical robots (Category: Surgical Robots):
    1. Intuitive Surgical da Vinci (via dVRK) — USL 7.1
    2. Medtronic Hugo RAS                     — USL 4.5
    3. CMR Surgical Versius                   — USL 3.4

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


class SurgicalRobotType(Enum):
    """Sub-types of surgical robot systems."""

    TELEOPERATED = "teleoperated"
    SEMI_AUTONOMOUS = "semi_autonomous"
    SUPERVISORY = "supervisory"
    COOPERATIVE = "cooperative"


class SurgicalSimFramework(Enum):
    """Supported simulation frameworks for surgical robot evaluation."""

    ISAAC_SIM = "nvidia_isaac_sim"
    ISAAC_LAB = "nvidia_isaac_lab"
    ORBIT_SURGICAL = "orbit_surgical"
    MUJOCO = "mujoco"
    MUJOCO_MJX = "mujoco_mjx"
    GAZEBO = "gazebo"
    PYBULLET = "pybullet"
    SURROL = "surrol"
    AMBF = "ambf"


class SurgicalAICapability(Enum):
    """AI capabilities specific to surgical robot evaluation."""

    GENERATIVE_AI = "generative_ai"
    AGENTIC_AI = "agentic_ai"
    VLA_MODELS = "vla_models"
    DIFFUSION_POLICY = "diffusion_policy"
    IMITATION_LEARNING = "imitation_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    LLM_PLANNING = "llm_planning"
    SURGICAL_VIDEO_AI = "surgical_video_ai"
    INSTRUMENT_SEGMENTATION = "instrument_segmentation"
    PHASE_RECOGNITION = "phase_recognition"
    MCP_INTEGRATION = "mcp_integration"


class SurgicalProcedure(Enum):
    """Oncology surgical procedures for task evaluation."""

    PROSTATECTOMY = "prostatectomy"
    NEPHRECTOMY = "nephrectomy"
    LOBECTOMY = "lobectomy"
    HYSTERECTOMY = "hysterectomy"
    COLECTOMY = "colectomy"
    CHOLECYSTECTOMY = "cholecystectomy"
    BIOPSY = "biopsy"
    LYMPH_NODE_DISSECTION = "lymph_node_dissection"


# ---------------------------------------------------------------------------
# Scoring data classes
# ---------------------------------------------------------------------------


@dataclass
class SurgicalSimScore:
    """Score for a single simulation framework's support for surgical robots."""

    framework: SurgicalSimFramework
    official_support: bool = False
    surgical_instruments_modeled: bool = False
    tissue_deformation_capable: bool = False
    ros2_integration: bool = False
    gpu_sim_capable: bool = False
    documented_examples: int = 0
    community_packages: int = 0
    bidirectional_transfer: bool = False
    haptic_feedback_sim: bool = False
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute sub-score for this framework on a 0-10 scale."""
        raw = 0.0
        raw += 2.0 if self.official_support else 0.0
        raw += 1.5 if self.surgical_instruments_modeled else 0.0
        raw += 1.0 if self.tissue_deformation_capable else 0.0
        raw += 1.0 if self.ros2_integration else 0.0
        raw += 1.0 if self.gpu_sim_capable else 0.0
        raw += min(self.documented_examples / 5.0, 1.0)
        raw += min(self.community_packages / 3.0, 1.0)
        raw += 0.5 if self.bidirectional_transfer else 0.0
        raw += 1.0 if self.haptic_feedback_sim else 0.0
        self.score = round(min(raw, 10.0), 1)
        return self.score


@dataclass
class SurgicalDimAScore:
    """Dimension A: Simulation Framework Switching for surgical robots."""

    framework_scores: list[SurgicalSimScore] = field(default_factory=list)
    num_frameworks_supported: int = 0
    cross_framework_transfer_tested: bool = False
    tissue_sim_fidelity: bool = False
    instrument_model_coverage: int = 0
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
        tissue_bonus = 0.5 if self.tissue_sim_fidelity else 0.0
        instrument_bonus = min(self.instrument_model_coverage / 10.0, 1.0) * 0.5

        raw = avg_fw * 0.5 + breadth_bonus + transfer_bonus + tissue_bonus + instrument_bonus
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class SurgicalDimBScore:
    """Dimension B: Generative/Agentic AI Integration for surgical robots."""

    supported_capabilities: list[SurgicalAICapability] = field(default_factory=list)
    llm_task_planning: bool = False
    vla_model_compatible: bool = False
    surgical_video_ai: bool = False
    imitation_learning_tested: bool = False
    diffusion_policy_tested: bool = False
    instrument_segmentation: bool = False
    phase_recognition: bool = False
    ai_safety_constraints: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension B score (0.0-10.0)."""
        raw = 0.0
        raw += min(len(self.supported_capabilities) / 8.0, 1.0) * 2.5
        raw += 1.5 if self.llm_task_planning else 0.0
        raw += 1.0 if self.vla_model_compatible else 0.0
        raw += 1.0 if self.surgical_video_ai else 0.0
        raw += 1.0 if self.imitation_learning_tested else 0.0
        raw += 1.0 if self.diffusion_policy_tested else 0.0
        raw += 0.5 if self.instrument_segmentation else 0.0
        raw += 0.5 if self.phase_recognition else 0.0
        raw += 1.0 if self.ai_safety_constraints else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class SurgicalDimCScore:
    """Dimension C: Cross-Robot Progress Sharing for surgical robots."""

    open_source_platform: bool = False
    intra_org_transfer: bool = False
    inter_org_transfer: bool = False
    onnx_policy_export: bool = False
    standardized_kinematics: bool = False
    skill_library_compatible: bool = False
    instrument_interchangeable: bool = False
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
        raw += 0.5 if self.standardized_kinematics else 0.0
        raw += 0.5 if self.skill_library_compatible else 0.0
        raw += 1.0 if self.instrument_interchangeable else 0.0
        raw += 1.0 if self.research_community_active else 0.0
        raw += 1.0 if self.real_time_state_sync else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class SurgicalDimDScore:
    """Dimension D: Multi-Site Clinical Trial Collaboration for surgical robots."""

    fda_cleared: bool = False
    ce_marked: bool = False
    multi_site_deployed: bool = False
    federated_learning_compatible: bool = False
    hipaa_compliance_tools: bool = False
    audit_trail_capable: bool = False
    remote_proctoring: bool = False
    surgical_data_standard: bool = False
    safety_certification: bool = False
    clinical_workflow_integration: bool = False
    iec_80601_compliance: bool = False
    iso_13482_alignment: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension D score (0.0-10.0)."""
        raw = 0.0
        raw += 1.5 if self.fda_cleared else 0.0
        raw += 1.0 if self.ce_marked else 0.0
        raw += 1.0 if self.multi_site_deployed else 0.0
        raw += 1.0 if self.federated_learning_compatible else 0.0
        raw += 0.5 if self.hipaa_compliance_tools else 0.0
        raw += 0.5 if self.audit_trail_capable else 0.0
        raw += 0.5 if self.remote_proctoring else 0.0
        raw += 0.5 if self.surgical_data_standard else 0.0
        raw += 1.0 if self.safety_certification else 0.0
        raw += 1.0 if self.clinical_workflow_integration else 0.0
        raw += 0.5 if self.iec_80601_compliance else 0.0
        raw += 1.0 if self.iso_13482_alignment else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


# ---------------------------------------------------------------------------
# Main USL Rating for Surgical Robots
# ---------------------------------------------------------------------------


@dataclass
class SurgicalUSLRating:
    """Complete USL rating for a surgical robot system.

    The final score is a weighted average of the four dimensions,
    rounded to the nearest 0.1 on a 1.0-10.0 scale.
    """

    robot_name: str
    manufacturer: str
    robot_type: SurgicalRobotType = SurgicalRobotType.TELEOPERATED
    dimension_a: SurgicalDimAScore = field(default_factory=SurgicalDimAScore)
    dimension_b: SurgicalDimBScore = field(default_factory=SurgicalDimBScore)
    dimension_c: SurgicalDimCScore = field(default_factory=SurgicalDimCScore)
    dimension_d: SurgicalDimDScore = field(default_factory=SurgicalDimDScore)
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    final_score: float = 0.0
    evaluated_date: str = ""
    evaluator: str = "USL Surgical Scoring Framework v1.0"
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
            f"Category:     Surgical Robot ({self.robot_type.value})",
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
    """Identify the weakest dimensions and suggest improvements.

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
            "Develop open-source simulation models with tissue deformation",
            "Implement GPU-accelerated surgical task environments",
            "Create instrument-specific simulation models for oncology",
        ],
        "B_ai_integration": [
            "Integrate LLM-based surgical planning (preoperative + intraoperative)",
            "Develop surgical phase recognition with foundation models",
            "Implement VLA models for autonomous surgical subtask execution",
        ],
        "C_cross_robot_sharing": [
            "Standardize surgical kinematics description formats",
            "Implement ONNX policy export for surgical skill transfer",
            "Enable cross-platform instrument compatibility",
        ],
        "D_clinical_trial_collab": [
            "Develop FDA/CE regulatory pathway documentation",
            "Implement HIPAA-compliant surgical data sharing",
            "Build remote proctoring infrastructure for multi-site trials",
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
# USL Level Classification (same as cobots, shared standard)
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


def generate_surgical_evaluation_report(
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
            "framework_version": "USL Surgical v1.0",
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

    # Text format
    lines = [
        "=" * 70,
        "  UNIFICATION STANDARD LEVEL (USL) — SURGICAL ROBOT EVALUATION REPORT",
        "=" * 70,
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "  Framework: USL Surgical v1.0",
        "  Category:  Surgical Robots (Teleoperated MIS Systems)",
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
# Demo / main
# ---------------------------------------------------------------------------


def _build_davinci_rating() -> SurgicalUSLRating:
    """Build the da Vinci (dVRK) USL rating."""
    return SurgicalUSLRating(
        robot_name="da Vinci (dVRK)",
        manufacturer="Intuitive Surgical",
        dimension_a=SurgicalDimAScore(
            framework_scores=[
                SurgicalSimScore(
                    framework=SurgicalSimFramework.ORBIT_SURGICAL,
                    official_support=True,
                    surgical_instruments_modeled=True,
                    tissue_deformation_capable=True,
                    gpu_sim_capable=True,
                    documented_examples=10,
                    community_packages=3,
                    bidirectional_transfer=True,
                    haptic_feedback_sim=True,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.GAZEBO,
                    official_support=True,
                    surgical_instruments_modeled=True,
                    ros2_integration=True,
                    documented_examples=8,
                    community_packages=4,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.SURROL,
                    official_support=True,
                    surgical_instruments_modeled=True,
                    tissue_deformation_capable=True,
                    documented_examples=6,
                    community_packages=2,
                    haptic_feedback_sim=True,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.AMBF,
                    official_support=True,
                    surgical_instruments_modeled=True,
                    tissue_deformation_capable=True,
                    ros2_integration=True,
                    documented_examples=5,
                    community_packages=2,
                    haptic_feedback_sim=True,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.MUJOCO,
                    surgical_instruments_modeled=True,
                    documented_examples=4,
                    community_packages=2,
                ),
            ],
            num_frameworks_supported=5,
            cross_framework_transfer_tested=True,
            tissue_sim_fidelity=True,
            instrument_model_coverage=8,
        ),
        dimension_b=SurgicalDimBScore(
            supported_capabilities=[
                SurgicalAICapability.GENERATIVE_AI,
                SurgicalAICapability.AGENTIC_AI,
                SurgicalAICapability.VLA_MODELS,
                SurgicalAICapability.DIFFUSION_POLICY,
                SurgicalAICapability.IMITATION_LEARNING,
                SurgicalAICapability.REINFORCEMENT_LEARNING,
                SurgicalAICapability.SURGICAL_VIDEO_AI,
                SurgicalAICapability.INSTRUMENT_SEGMENTATION,
                SurgicalAICapability.PHASE_RECOGNITION,
            ],
            llm_task_planning=True,
            vla_model_compatible=True,
            surgical_video_ai=True,
            imitation_learning_tested=True,
            diffusion_policy_tested=True,
            instrument_segmentation=True,
            phase_recognition=True,
            ai_safety_constraints=True,
        ),
        dimension_c=SurgicalDimCScore(
            open_source_platform=True,
            intra_org_transfer=True,
            inter_org_transfer=True,
            onnx_policy_export=True,
            standardized_kinematics=True,
            skill_library_compatible=True,
            instrument_interchangeable=False,
            research_community_active=True,
            real_time_state_sync=True,
        ),
        dimension_d=SurgicalDimDScore(
            fda_cleared=True,
            ce_marked=True,
            multi_site_deployed=True,
            federated_learning_compatible=True,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_proctoring=True,
            surgical_data_standard=True,
            safety_certification=True,
            clinical_workflow_integration=True,
            iec_80601_compliance=True,
            iso_13482_alignment=True,
        ),
    )


def _build_hugo_rating() -> SurgicalUSLRating:
    """Build the Medtronic Hugo RAS USL rating."""
    return SurgicalUSLRating(
        robot_name="Hugo RAS",
        manufacturer="Medtronic",
        dimension_a=SurgicalDimAScore(
            framework_scores=[
                SurgicalSimScore(
                    framework=SurgicalSimFramework.ISAAC_SIM,
                    surgical_instruments_modeled=True,
                    gpu_sim_capable=True,
                    documented_examples=3,
                    community_packages=1,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.GAZEBO,
                    ros2_integration=True,
                    documented_examples=2,
                    community_packages=1,
                ),
            ],
            num_frameworks_supported=2,
            cross_framework_transfer_tested=False,
            tissue_sim_fidelity=True,
            instrument_model_coverage=4,
        ),
        dimension_b=SurgicalDimBScore(
            supported_capabilities=[
                SurgicalAICapability.GENERATIVE_AI,
                SurgicalAICapability.SURGICAL_VIDEO_AI,
                SurgicalAICapability.INSTRUMENT_SEGMENTATION,
                SurgicalAICapability.PHASE_RECOGNITION,
            ],
            llm_task_planning=False,
            vla_model_compatible=False,
            surgical_video_ai=True,
            imitation_learning_tested=False,
            diffusion_policy_tested=False,
            instrument_segmentation=True,
            phase_recognition=True,
            ai_safety_constraints=True,
        ),
        dimension_c=SurgicalDimCScore(
            open_source_platform=False,
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=False,
            standardized_kinematics=False,
            skill_library_compatible=False,
            instrument_interchangeable=True,
            research_community_active=False,
            real_time_state_sync=False,
        ),
        dimension_d=SurgicalDimDScore(
            fda_cleared=False,
            ce_marked=True,
            multi_site_deployed=True,
            federated_learning_compatible=False,
            hipaa_compliance_tools=True,
            audit_trail_capable=True,
            remote_proctoring=True,
            surgical_data_standard=True,
            safety_certification=True,
            clinical_workflow_integration=True,
            iec_80601_compliance=True,
            iso_13482_alignment=False,
        ),
    )


def _build_versius_rating() -> SurgicalUSLRating:
    """Build the CMR Surgical Versius USL rating."""
    return SurgicalUSLRating(
        robot_name="Versius",
        manufacturer="CMR Surgical",
        dimension_a=SurgicalDimAScore(
            framework_scores=[
                SurgicalSimScore(
                    framework=SurgicalSimFramework.GAZEBO,
                    ros2_integration=True,
                    documented_examples=2,
                    community_packages=1,
                ),
                SurgicalSimScore(
                    framework=SurgicalSimFramework.MUJOCO,
                    documented_examples=1,
                    community_packages=1,
                ),
            ],
            num_frameworks_supported=2,
            cross_framework_transfer_tested=False,
            tissue_sim_fidelity=False,
            instrument_model_coverage=3,
        ),
        dimension_b=SurgicalDimBScore(
            supported_capabilities=[
                SurgicalAICapability.GENERATIVE_AI,
                SurgicalAICapability.SURGICAL_VIDEO_AI,
                SurgicalAICapability.INSTRUMENT_SEGMENTATION,
            ],
            llm_task_planning=False,
            vla_model_compatible=False,
            surgical_video_ai=True,
            imitation_learning_tested=False,
            diffusion_policy_tested=False,
            instrument_segmentation=True,
            phase_recognition=False,
            ai_safety_constraints=True,
        ),
        dimension_c=SurgicalDimCScore(
            open_source_platform=False,
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=False,
            standardized_kinematics=False,
            skill_library_compatible=False,
            instrument_interchangeable=False,
            research_community_active=False,
            real_time_state_sync=False,
        ),
        dimension_d=SurgicalDimDScore(
            fda_cleared=False,
            ce_marked=True,
            multi_site_deployed=True,
            federated_learning_compatible=False,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_proctoring=True,
            surgical_data_standard=False,
            safety_certification=True,
            clinical_workflow_integration=True,
            iec_80601_compliance=True,
            iso_13482_alignment=False,
        ),
    )


def _demo() -> None:
    """Demonstrate USL scoring with three surgical robots."""
    logger.info("Running USL surgical robot scoring demonstration...")

    davinci = _build_davinci_rating()
    hugo = _build_hugo_rating()
    versius = _build_versius_rating()

    report = generate_surgical_evaluation_report([davinci, hugo, versius])
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
