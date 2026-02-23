"""Unification Standard Level (USL) Scoring Framework for Physical AI Cobots.

Evaluates collaborative robots on their readiness for multi-site oncology
clinical trials.  Final scores range from 1.0 to 10.0 in 0.1 increments
across four weighted dimensions:

    A) Simulation Framework Switching   (25%)
    B) Generative / Agentic AI Integration (25%)
    C) Cross-Robot Progress Sharing     (25%)
    D) Multi-Site Clinical Trial Collaboration (25%)

Inspired by:
  - NASA/DOD TRL (Mankins, 1995; 2004 White Paper)
  - ML Technology Readiness Levels (MLTRL, Lavin et al., 2021)
  - TRL adaptation for complex systems (Tomaschek et al., 2015;
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CobotCategory(Enum):
    """Robot categories supported by the USL standard."""

    COLLABORATIVE = "collaborative"
    SURGICAL = "surgical"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMANOID = "humanoid"


class SimFramework(Enum):
    """Supported simulation frameworks for dimension A evaluation."""

    ISAAC_SIM = "nvidia_isaac_sim"
    ISAAC_LAB = "nvidia_isaac_lab"
    MUJOCO = "mujoco"
    MUJOCO_MJX = "mujoco_mjx"
    GAZEBO = "gazebo"
    PYBULLET = "pybullet"
    WEBOTS = "webots"
    COPPELIASIM = "coppeliasim"


class AICapability(Enum):
    """AI integration capabilities for dimension B evaluation."""

    GENERATIVE_AI = "generative_ai"
    AGENTIC_AI = "agentic_ai"
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    VLA_MODELS = "vla_models"
    DIFFUSION_POLICY = "diffusion_policy"
    MCP_INTEGRATION = "mcp_integration"
    LLM_PLANNING = "llm_planning"


class ModelFormat(Enum):
    """Robot model description formats."""

    URDF = "urdf"
    MJCF = "mjcf"
    SDF = "sdf"
    USD = "usd"
    XACRO = "xacro"


class ProgressSharingMethod(Enum):
    """Methods for cross-robot progress sharing (dimension C)."""

    ONNX_EXPORT = "onnx_export"
    ROS2_ACTION = "ros2_action"
    POLICY_CHECKPOINT = "policy_checkpoint"
    FEDERATED_LEARNING = "federated_learning"
    SHARED_STATE_SPACE = "shared_state_space"
    STANDARDIZED_API = "standardized_api"


# ---------------------------------------------------------------------------
# Scoring data classes
# ---------------------------------------------------------------------------


@dataclass
class SimFrameworkScore:
    """Score for a single simulation framework's support level.

    Attributes:
        framework: The simulation framework being evaluated.
        official_support: Whether the manufacturer provides official support.
        urdf_available: URDF model available and validated.
        mjcf_available: MJCF model available and validated.
        sdf_available: SDF model available and validated.
        ros2_integration: Native ROS 2 integration exists.
        gpu_sim_capable: Supports GPU-accelerated parallel simulation.
        documented_examples: Number of documented usage examples (0-10+).
        community_packages: Number of community-maintained packages.
        bidirectional_transfer: Supports bidirectional policy transfer.
        score: Computed sub-score (0.0-10.0).
    """

    framework: SimFramework
    official_support: bool = False
    urdf_available: bool = False
    mjcf_available: bool = False
    sdf_available: bool = False
    ros2_integration: bool = False
    gpu_sim_capable: bool = False
    documented_examples: int = 0
    community_packages: int = 0
    bidirectional_transfer: bool = False
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute sub-score for this framework on a 0-10 scale."""
        raw = 0.0
        raw += 2.0 if self.official_support else 0.0
        raw += 1.0 if self.urdf_available else 0.0
        raw += 1.0 if self.mjcf_available else 0.0
        raw += 0.5 if self.sdf_available else 0.0
        raw += 1.5 if self.ros2_integration else 0.0
        raw += 1.0 if self.gpu_sim_capable else 0.0
        raw += min(self.documented_examples / 5.0, 1.0)
        raw += min(self.community_packages / 3.0, 1.0)
        raw += 1.0 if self.bidirectional_transfer else 0.0
        self.score = round(min(raw, 10.0), 1)
        return self.score


@dataclass
class DimensionAScore:
    """Dimension A: Simulation Framework Switching capability.

    Evaluates how well a cobot can transition between different physics
    simulation environments while preserving trained policies.
    """

    framework_scores: list[SimFrameworkScore] = field(default_factory=list)
    num_frameworks_supported: int = 0
    cross_framework_transfer_tested: bool = False
    model_format_coverage: list[ModelFormat] = field(default_factory=list)
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension A score (0.0-10.0)."""
        if not self.framework_scores:
            self.score = 1.0
            return self.score

        avg_fw = sum(fs.compute_score() for fs in self.framework_scores) / len(self.framework_scores)
        breadth_bonus = min(self.num_frameworks_supported / 6.0, 1.0) * 2.0
        transfer_bonus = 1.0 if self.cross_framework_transfer_tested else 0.0
        format_bonus = min(len(self.model_format_coverage) / 4.0, 1.0)

        raw = avg_fw * 0.5 + breadth_bonus + transfer_bonus + format_bonus
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class DimensionBScore:
    """Dimension B: Generative/Agentic AI Integration capability.

    Evaluates the cobot's ability to integrate with modern AI systems
    including LLMs, VLAs, diffusion policies, and agentic frameworks.
    """

    supported_capabilities: list[AICapability] = field(default_factory=list)
    llm_task_planning: bool = False
    vla_model_compatible: bool = False
    mcp_server_available: bool = False
    diffusion_policy_tested: bool = False
    agentic_framework_count: int = 0
    natural_language_control: bool = False
    ai_safety_constraints: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension B score (0.0-10.0)."""
        raw = 0.0
        raw += min(len(self.supported_capabilities) / 6.0, 1.0) * 3.0
        raw += 1.5 if self.llm_task_planning else 0.0
        raw += 1.5 if self.vla_model_compatible else 0.0
        raw += 1.0 if self.mcp_server_available else 0.0
        raw += 1.0 if self.diffusion_policy_tested else 0.0
        raw += min(self.agentic_framework_count / 3.0, 1.0)
        raw += 0.5 if self.natural_language_control else 0.0
        raw += 0.5 if self.ai_safety_constraints else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class DimensionCScore:
    """Dimension C: Cross-Robot Progress Sharing capability.

    Evaluates how well a cobot can share and continue progress with other
    robots in its category, both from the same and different manufacturers.
    """

    sharing_methods: list[ProgressSharingMethod] = field(default_factory=list)
    intra_org_transfer: bool = False
    inter_org_transfer: bool = False
    onnx_policy_export: bool = False
    standardized_action_space: bool = False
    skill_library_compatible: bool = False
    checkpoint_format_documented: bool = False
    real_time_state_sync: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension C score (0.0-10.0)."""
        raw = 0.0
        raw += min(len(self.sharing_methods) / 4.0, 1.0) * 2.5
        raw += 1.5 if self.intra_org_transfer else 0.0
        raw += 2.0 if self.inter_org_transfer else 0.0
        raw += 1.0 if self.onnx_policy_export else 0.0
        raw += 1.0 if self.standardized_action_space else 0.0
        raw += 0.5 if self.skill_library_compatible else 0.0
        raw += 1.0 if self.checkpoint_format_documented else 0.0
        raw += 0.5 if self.real_time_state_sync else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


@dataclass
class DimensionDScore:
    """Dimension D: Multi-Site Clinical Trial Collaboration capability.

    Evaluates the cobot's readiness for deployment across multiple clinical
    trial sites with federated coordination and regulatory compliance.
    """

    multi_site_tested: bool = False
    federated_learning_compatible: bool = False
    hipaa_compliance_tools: bool = False
    audit_trail_capable: bool = False
    remote_monitoring: bool = False
    regulatory_documentation: bool = False
    safety_certification_pathway: bool = False
    clinical_workflow_integration: bool = False
    iso_13482_alignment: bool = False
    iec_62304_documentation: bool = False
    notes: str = ""
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute dimension D score (0.0-10.0)."""
        raw = 0.0
        raw += 2.0 if self.multi_site_tested else 0.0
        raw += 1.5 if self.federated_learning_compatible else 0.0
        raw += 1.0 if self.hipaa_compliance_tools else 0.0
        raw += 0.5 if self.audit_trail_capable else 0.0
        raw += 0.5 if self.remote_monitoring else 0.0
        raw += 1.0 if self.regulatory_documentation else 0.0
        raw += 1.5 if self.safety_certification_pathway else 0.0
        raw += 0.5 if self.clinical_workflow_integration else 0.0
        raw += 1.0 if self.iso_13482_alignment else 0.0
        raw += 0.5 if self.iec_62304_documentation else 0.0
        self.score = round(max(1.0, min(raw, 10.0)), 1)
        return self.score


# ---------------------------------------------------------------------------
# Main USL Rating
# ---------------------------------------------------------------------------


@dataclass
class USLRating:
    """Complete USL rating for a collaborative robot.

    The final score is a weighted average of the four dimensions,
    rounded to the nearest 0.1 on a 1.0-10.0 scale.
    """

    robot_name: str
    manufacturer: str
    category: CobotCategory = CobotCategory.COLLABORATIVE
    dimension_a: DimensionAScore = field(default_factory=DimensionAScore)
    dimension_b: DimensionBScore = field(default_factory=DimensionBScore)
    dimension_c: DimensionCScore = field(default_factory=DimensionCScore)
    dimension_d: DimensionDScore = field(default_factory=DimensionDScore)
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    final_score: float = 0.0
    evaluated_date: str = ""
    evaluator: str = "USL Scoring Framework v1.0"
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
        # Round to nearest 0.1 and clamp to [1.0, 10.0]
        self.final_score = max(1.0, min(round(raw * 10) / 10, 10.0))
        self.evaluated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.final_score

    def summary(self) -> str:
        """Generate a human-readable summary of the USL rating."""
        self.compute_final_score()
        lines = [
            f"USL Rating: {self.robot_name} ({self.manufacturer})",
            f"{'=' * 55}",
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


def _score_band_label(score: float) -> str:
    """Return a human-readable band label for a USL score."""
    for threshold, label, description in _SCORE_BANDS:
        if score >= threshold:
            return f"  Band: {label} ({description})"
    return "  Band: Initial (Minimal unification capability)"


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_ratings(ratings: list[USLRating]) -> str:
    """Generate a comparison table for multiple USL ratings.

    Args:
        ratings: List of USL ratings to compare.

    Returns:
        Formatted comparison string.
    """
    for r in ratings:
        r.compute_final_score()

    header = f"{'Robot':<25} {'Mfg':<15} {'Dim A':>6} {'Dim B':>6} {'Dim C':>6} {'Dim D':>6} {'FINAL':>7}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in sorted(ratings, key=lambda x: x.final_score, reverse=True):
        lines.append(
            f"{r.robot_name:<25} {r.manufacturer:<15} "
            f"{r.dimension_a.score:>5.1f} {r.dimension_b.score:>5.1f} "
            f"{r.dimension_c.score:>5.1f} {r.dimension_d.score:>5.1f} "
            f"{r.final_score:>6.1f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def compute_gap_analysis(rating: USLRating) -> dict:
    """Identify the weakest dimensions and suggest improvements.

    Args:
        rating: A computed USL rating.

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
            "Add URDF/MJCF/SDF model support for more frameworks",
            "Implement bidirectional policy transfer testing",
            "Develop GPU simulation capability (Isaac Lab / MJX)",
        ],
        "B_ai_integration": [
            "Integrate LLM-based task planning (e.g., Claude Code, Codex)",
            "Implement MCP server for tool integration",
            "Test with VLA models (GR00T, OpenVLA)",
        ],
        "C_cross_robot_sharing": [
            "Standardize action space definitions",
            "Implement ONNX policy export pipeline",
            "Enable real-time state synchronization via ROS 2",
        ],
        "D_clinical_trial_collab": [
            "Develop FDA regulatory documentation",
            "Implement audit trail with 21 CFR Part 11 compliance",
            "Establish ISO 13482 safety certification pathway",
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
# USL Level Classification
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
# Evaluation report generator
# ---------------------------------------------------------------------------


def generate_evaluation_report(
    ratings: list[USLRating],
    output_format: str = "text",
) -> str:
    """Generate a comprehensive evaluation report for multiple cobots.

    Args:
        ratings: List of USL ratings to include.
        output_format: Either 'text' or 'json'.

    Returns:
        Formatted report string.
    """
    for r in ratings:
        r.compute_final_score()

    if output_format == "json":
        report_data = {
            "report_title": "USL Cobot Evaluation Report",
            "generated": datetime.now(timezone.utc).isoformat(),
            "framework_version": "USL v1.0",
            "ratings": [r.to_dict() for r in ratings],
            "comparison": {
                r.robot_name: {
                    "gap_analysis": compute_gap_analysis(r),
                    "level": get_usl_level(r.final_score)[0],
                    "level_description": get_usl_level(r.final_score)[1],
                }
                for r in ratings
            },
        }
        return json.dumps(report_data, indent=2)

    # Text format
    lines = [
        "=" * 65,
        "  UNIFICATION STANDARD LEVEL (USL) — COBOT EVALUATION REPORT",
        "=" * 65,
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "  Framework: USL v1.0",
        "  Category:  Collaborative Robots (Cobots)",
        "=" * 65,
        "",
    ]

    for r in sorted(ratings, key=lambda x: x.final_score, reverse=True):
        lines.append(r.summary())
        level_num, level_desc = get_usl_level(r.final_score)
        lines.append(f"  USL Level: {level_num} — {level_desc}")
        lines.append("")

    lines.append("")
    lines.append("COMPARISON TABLE")
    lines.append(compare_ratings(ratings))
    lines.append("")

    lines.append("GAP ANALYSIS")
    lines.append("-" * 65)
    for r in ratings:
        gap = compute_gap_analysis(r)
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


def _demo() -> None:
    """Demonstrate USL scoring with three example cobots."""
    logger.info("Running USL scoring demonstration...")

    # --- Franka Emika Panda ---
    franka = USLRating(
        robot_name="Franka Emika Panda",
        manufacturer="Franka Robotics",
        dimension_a=DimensionAScore(
            framework_scores=[
                SimFrameworkScore(
                    framework=SimFramework.MUJOCO,
                    official_support=True,
                    urdf_available=True,
                    mjcf_available=True,
                    ros2_integration=True,
                    gpu_sim_capable=True,
                    documented_examples=10,
                    community_packages=5,
                    bidirectional_transfer=True,
                ),
                SimFrameworkScore(
                    framework=SimFramework.ISAAC_LAB,
                    official_support=True,
                    urdf_available=True,
                    mjcf_available=True,
                    sdf_available=True,
                    ros2_integration=True,
                    gpu_sim_capable=True,
                    documented_examples=8,
                    community_packages=3,
                    bidirectional_transfer=True,
                ),
                SimFrameworkScore(
                    framework=SimFramework.GAZEBO,
                    official_support=True,
                    urdf_available=True,
                    sdf_available=True,
                    ros2_integration=True,
                    documented_examples=6,
                    community_packages=4,
                ),
                SimFrameworkScore(
                    framework=SimFramework.PYBULLET,
                    urdf_available=True,
                    ros2_integration=True,
                    documented_examples=5,
                    community_packages=3,
                ),
            ],
            num_frameworks_supported=5,
            cross_framework_transfer_tested=True,
            model_format_coverage=[ModelFormat.URDF, ModelFormat.MJCF, ModelFormat.SDF, ModelFormat.USD],
        ),
        dimension_b=DimensionBScore(
            supported_capabilities=[
                AICapability.GENERATIVE_AI,
                AICapability.AGENTIC_AI,
                AICapability.VLA_MODELS,
                AICapability.DIFFUSION_POLICY,
                AICapability.LLM_PLANNING,
            ],
            llm_task_planning=True,
            vla_model_compatible=True,
            mcp_server_available=False,
            diffusion_policy_tested=True,
            agentic_framework_count=2,
            natural_language_control=True,
            ai_safety_constraints=True,
        ),
        dimension_c=DimensionCScore(
            sharing_methods=[
                ProgressSharingMethod.ONNX_EXPORT,
                ProgressSharingMethod.ROS2_ACTION,
                ProgressSharingMethod.POLICY_CHECKPOINT,
                ProgressSharingMethod.SHARED_STATE_SPACE,
            ],
            intra_org_transfer=True,
            inter_org_transfer=True,
            onnx_policy_export=True,
            standardized_action_space=True,
            skill_library_compatible=True,
            checkpoint_format_documented=True,
            real_time_state_sync=True,
        ),
        dimension_d=DimensionDScore(
            multi_site_tested=False,
            federated_learning_compatible=True,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_monitoring=True,
            regulatory_documentation=True,
            safety_certification_pathway=True,
            clinical_workflow_integration=False,
            iso_13482_alignment=True,
            iec_62304_documentation=False,
        ),
    )

    # --- Kinova Gen3 ---
    kinova = USLRating(
        robot_name="Kinova Gen3 7DoF",
        manufacturer="Kinova Robotics",
        dimension_a=DimensionAScore(
            framework_scores=[
                SimFrameworkScore(
                    framework=SimFramework.GAZEBO,
                    official_support=True,
                    urdf_available=True,
                    sdf_available=True,
                    ros2_integration=True,
                    documented_examples=7,
                    community_packages=3,
                ),
                SimFrameworkScore(
                    framework=SimFramework.MUJOCO,
                    urdf_available=True,
                    mjcf_available=True,
                    ros2_integration=True,
                    documented_examples=4,
                    community_packages=2,
                    bidirectional_transfer=True,
                ),
                SimFrameworkScore(
                    framework=SimFramework.ISAAC_LAB,
                    urdf_available=True,
                    ros2_integration=True,
                    gpu_sim_capable=True,
                    documented_examples=3,
                    community_packages=1,
                ),
            ],
            num_frameworks_supported=4,
            cross_framework_transfer_tested=True,
            model_format_coverage=[ModelFormat.URDF, ModelFormat.MJCF, ModelFormat.SDF, ModelFormat.XACRO],
        ),
        dimension_b=DimensionBScore(
            supported_capabilities=[
                AICapability.GENERATIVE_AI,
                AICapability.AGENTIC_AI,
                AICapability.LLM_PLANNING,
            ],
            llm_task_planning=True,
            vla_model_compatible=False,
            mcp_server_available=False,
            diffusion_policy_tested=False,
            agentic_framework_count=1,
            natural_language_control=True,
            ai_safety_constraints=True,
        ),
        dimension_c=DimensionCScore(
            sharing_methods=[
                ProgressSharingMethod.ONNX_EXPORT,
                ProgressSharingMethod.ROS2_ACTION,
                ProgressSharingMethod.STANDARDIZED_API,
            ],
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=True,
            standardized_action_space=True,
            skill_library_compatible=False,
            checkpoint_format_documented=True,
            real_time_state_sync=True,
        ),
        dimension_d=DimensionDScore(
            multi_site_tested=False,
            federated_learning_compatible=False,
            hipaa_compliance_tools=False,
            audit_trail_capable=True,
            remote_monitoring=True,
            regulatory_documentation=True,
            safety_certification_pathway=True,
            clinical_workflow_integration=True,
            iso_13482_alignment=True,
            iec_62304_documentation=False,
        ),
    )

    # --- UFACTORY xArm 7 ---
    xarm = USLRating(
        robot_name="UFACTORY xArm 7",
        manufacturer="UFACTORY",
        dimension_a=DimensionAScore(
            framework_scores=[
                SimFrameworkScore(
                    framework=SimFramework.GAZEBO,
                    official_support=True,
                    urdf_available=True,
                    sdf_available=True,
                    ros2_integration=True,
                    documented_examples=5,
                    community_packages=2,
                ),
                SimFrameworkScore(
                    framework=SimFramework.MUJOCO,
                    urdf_available=True,
                    mjcf_available=True,
                    documented_examples=3,
                    community_packages=2,
                ),
                SimFrameworkScore(
                    framework=SimFramework.PYBULLET,
                    urdf_available=True,
                    documented_examples=4,
                    community_packages=2,
                ),
            ],
            num_frameworks_supported=3,
            cross_framework_transfer_tested=False,
            model_format_coverage=[ModelFormat.URDF, ModelFormat.MJCF, ModelFormat.SDF],
        ),
        dimension_b=DimensionBScore(
            supported_capabilities=[
                AICapability.GENERATIVE_AI,
                AICapability.AGENTIC_AI,
            ],
            llm_task_planning=False,
            vla_model_compatible=False,
            mcp_server_available=False,
            diffusion_policy_tested=False,
            agentic_framework_count=1,
            natural_language_control=False,
            ai_safety_constraints=False,
        ),
        dimension_c=DimensionCScore(
            sharing_methods=[
                ProgressSharingMethod.ONNX_EXPORT,
                ProgressSharingMethod.ROS2_ACTION,
            ],
            intra_org_transfer=True,
            inter_org_transfer=False,
            onnx_policy_export=True,
            standardized_action_space=False,
            skill_library_compatible=False,
            checkpoint_format_documented=False,
            real_time_state_sync=False,
        ),
        dimension_d=DimensionDScore(
            multi_site_tested=False,
            federated_learning_compatible=False,
            hipaa_compliance_tools=False,
            audit_trail_capable=False,
            remote_monitoring=True,
            regulatory_documentation=False,
            safety_certification_pathway=False,
            clinical_workflow_integration=False,
            iso_13482_alignment=False,
            iec_62304_documentation=False,
        ),
    )

    report = generate_evaluation_report([franka, kinova, xarm])
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
