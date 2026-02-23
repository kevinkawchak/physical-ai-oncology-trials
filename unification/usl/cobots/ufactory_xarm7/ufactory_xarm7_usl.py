"""UFACTORY xArm 7 — USL Unification Module.

Provides tools for integrating the UFACTORY xArm 7 (7-DOF collaborative
robot arm by UFACTORY) into the USL unification framework for physical AI
oncology clinical trials.

Key specifications:
    - DOF: 7 revolute joints
    - Payload: 3.5 kg
    - Reach: 700 mm
    - Repeatability: +/- 0.1 mm
    - Weight: 11.2 kg (arm only)
    - Torque sensors: Integrated in all 7 joints
    - Communication: Ethernet (250 Hz default, up to 500 Hz)
    - Built-in collision detection

Open-source ecosystem:
    - GitHub (SDK): https://github.com/xArm-Developer/xArm-Python-SDK
    - GitHub (ROS 2): https://github.com/xArm-Developer/xarm_ros2
    - MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
    - URDF models included in xarm_description package
    - ROS 2 Humble/Jazzy support via xarm_ros2

Simulation framework support:
    - MuJoCo: Model in Menagerie; research community use growing
    - Gazebo: Official URDF via xarm_ros2; ros2_control integration
    - PyBullet: Community URDF; basic support
    - Isaac Lab: Community URDF importable

Notable features:
    - Most affordable 7-DOF cobot in its class
    - Built-in collision detection without external sensors
    - xArm Studio GUI for intuitive programming
    - Python SDK for direct low-level control

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
# xArm 7 specifications
# ---------------------------------------------------------------------------


@dataclass
class XArm7Specs:
    """Hardware specifications for the UFACTORY xArm 7."""

    name: str = "UFACTORY xArm 7"
    manufacturer: str = "UFACTORY"
    dof: int = 7
    payload_kg: float = 3.5
    reach_mm: float = 700.0
    repeatability_mm: float = 0.1
    weight_kg: float = 11.2
    max_joint_velocity_rad_s: float = 3.14
    max_cartesian_velocity_m_s: float = 1.0
    max_cartesian_acceleration_m_s2: float = 20.0
    torque_sensors: bool = True
    torque_sensor_joints: int = 7
    communication_frequency_hz: int = 250
    max_communication_hz: int = 500
    collision_detection: bool = True
    ip_rating: str = "IP51"
    operating_temp_c: tuple[float, float] = (0.0, 50.0)
    power_supply_v: float = 24.0
    max_power_w: float = 200.0


# ---------------------------------------------------------------------------
# Joint specifications
# ---------------------------------------------------------------------------


@dataclass
class XArmJoint:
    """Specifications for a single xArm joint."""

    joint_index: int
    name: str
    lower_limit_rad: float
    upper_limit_rad: float
    max_velocity_rad_s: float
    max_torque_nm: float


XARM7_JOINTS = [
    XArmJoint(0, "joint1", -6.2832, 6.2832, 3.14, 50.0),
    XArmJoint(1, "joint2", -2.059, 2.094, 3.14, 50.0),
    XArmJoint(2, "joint3", -6.2832, 6.2832, 3.14, 32.0),
    XArmJoint(3, "joint4", -0.192, 3.927, 3.14, 32.0),
    XArmJoint(4, "joint5", -6.2832, 6.2832, 3.14, 12.0),
    XArmJoint(5, "joint6", -1.692, 3.141, 3.14, 12.0),
    XArmJoint(6, "joint7", -6.2832, 6.2832, 3.14, 12.0),
]


# ---------------------------------------------------------------------------
# Simulation framework configurations
# ---------------------------------------------------------------------------


class XArmSimFramework(Enum):
    """Simulation frameworks with xArm 7 support."""

    MUJOCO = "mujoco"
    GAZEBO = "gazebo"
    PYBULLET = "pybullet"
    ISAAC_LAB = "isaac_lab"


@dataclass
class FrameworkConfig:
    """Configuration for a single simulation framework."""

    framework: XArmSimFramework
    model_path: str
    model_format: str
    ros2_package: str = ""
    gpu_capable: bool = False
    control_frequency_hz: int = 250
    supported_controllers: list[str] = field(default_factory=list)
    notes: str = ""


XARM_FRAMEWORK_CONFIGS = {
    XArmSimFramework.GAZEBO: FrameworkConfig(
        framework=XArmSimFramework.GAZEBO,
        model_path="xarm_description/urdf/xarm7/xarm7.urdf.xacro",
        model_format="urdf",
        ros2_package="xarm_ros2",
        gpu_capable=False,
        control_frequency_hz=250,
        supported_controllers=["joint_trajectory", "effort", "velocity", "gripper"],
        notes="Official xarm_ros2 package; MoveIt2 integration; ros2_control",
    ),
    XArmSimFramework.MUJOCO: FrameworkConfig(
        framework=XArmSimFramework.MUJOCO,
        model_path="mujoco_menagerie/ufactory_xarm7/xarm7.xml",
        model_format="mjcf",
        gpu_capable=True,
        control_frequency_hz=500,
        supported_controllers=["position", "velocity", "torque"],
        notes="MuJoCo Menagerie model available; MJX GPU support",
    ),
    XArmSimFramework.PYBULLET: FrameworkConfig(
        framework=XArmSimFramework.PYBULLET,
        model_path="xarm7/xarm7.urdf",
        model_format="urdf",
        gpu_capable=False,
        control_frequency_hz=240,
        supported_controllers=["position", "velocity", "torque"],
        notes="Community URDF; basic Gymnasium environments",
    ),
    XArmSimFramework.ISAAC_LAB: FrameworkConfig(
        framework=XArmSimFramework.ISAAC_LAB,
        model_path="",
        model_format="usd",
        gpu_capable=True,
        control_frequency_hz=250,
        supported_controllers=["joint_position", "joint_effort"],
        notes="Community URDF to USD conversion; not officially supported",
    ),
}


# ---------------------------------------------------------------------------
# xArm Python SDK abstraction
# ---------------------------------------------------------------------------


@dataclass
class XArmCommand:
    """A command to send to the xArm via the Python SDK."""

    command_type: str  # "joint", "cartesian", "servo", "gripper"
    target_values: list[float] = field(default_factory=list)
    speed: float = 0.0
    acceleration: float = 0.0
    wait: bool = True
    timestamp: str = ""


class XArmSDKInterface:
    """Abstraction layer for the UFACTORY xArm Python SDK.

    Provides a standardized interface for controlling the xArm 7
    that mirrors the xArm-Python-SDK API structure for easy integration.
    """

    CONTROL_MODES = [
        "position",  # Position control mode (default)
        "servo_joint",  # Real-time servo (joint space)
        "servo_cartesian",  # Real-time servo (Cartesian space)
        "joint_teaching",  # Manual guidance / teach mode
        "cartesian_teaching",  # Cartesian teaching mode
    ]

    MOTION_STATES = [
        "idle",
        "moving",
        "paused",
        "error",
        "emergency_stop",
    ]

    SUPPORTED_GRIPPERS = [
        "UFACTORY xArm Gripper",
        "UFACTORY xArm Vacuum Gripper",
        "Robotiq 2F-85",
        "Robotiq 2F-140",
        "Custom End-Effector",
    ]

    ERROR_CODES = {
        1: "Emergency stop",
        2: "Joint communication error",
        3: "Joint angle limit",
        10: "Self-collision detected",
        11: "External collision detected",
        17: "Overload protection",
        18: "Overspeed protection",
        21: "Kinematic error",
    }

    def __init__(self, control_mode: str = "position") -> None:
        if control_mode not in self.CONTROL_MODES:
            logger.warning("Unsupported control mode: %s; defaulting to position", control_mode)
            control_mode = "position"
        self._control_mode = control_mode
        self._command_history: list[XArmCommand] = []
        self._state = "idle"
        self._error_code = 0

    def create_joint_command(
        self,
        target_angles_deg: list[float],
        speed_deg_s: float = 30.0,
        acceleration_deg_s2: float = 500.0,
    ) -> XArmCommand:
        """Create a joint-space motion command.

        Args:
            target_angles_deg: Target joint angles in degrees (7 values).
            speed_deg_s: Joint speed in degrees/second.
            acceleration_deg_s2: Joint acceleration in degrees/second^2.

        Returns:
            XArmCommand ready for execution.
        """
        if len(target_angles_deg) != 7:
            logger.error("Expected 7 joint values, got %d", len(target_angles_deg))
            return XArmCommand(command_type="error")

        # Validate against joint limits (convert to radians for check)
        for i, (angle_deg, joint_spec) in enumerate(zip(target_angles_deg, XARM7_JOINTS)):
            angle_rad = math.radians(angle_deg)
            if angle_rad < joint_spec.lower_limit_rad or angle_rad > joint_spec.upper_limit_rad:
                logger.warning(
                    "Joint %d angle %.1f deg outside limits [%.1f, %.1f] deg",
                    i,
                    angle_deg,
                    math.degrees(joint_spec.lower_limit_rad),
                    math.degrees(joint_spec.upper_limit_rad),
                )

        cmd = XArmCommand(
            command_type="joint",
            target_values=list(target_angles_deg),
            speed=speed_deg_s,
            acceleration=acceleration_deg_s2,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._command_history.append(cmd)
        return cmd

    def create_cartesian_command(
        self,
        position_mm: list[float],
        orientation_deg: list[float],
        speed_mm_s: float = 100.0,
    ) -> XArmCommand:
        """Create a Cartesian-space motion command.

        Args:
            position_mm: Target [x, y, z] in millimeters.
            orientation_deg: Target [roll, pitch, yaw] in degrees.
            speed_mm_s: Cartesian speed in mm/s.

        Returns:
            XArmCommand ready for execution.
        """
        cmd = XArmCommand(
            command_type="cartesian",
            target_values=list(position_mm) + list(orientation_deg),
            speed=speed_mm_s,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._command_history.append(cmd)
        return cmd

    def get_error_description(self, code: int) -> str:
        """Get human-readable error description.

        Args:
            code: Error code from the xArm controller.

        Returns:
            Error description string.
        """
        return self.ERROR_CODES.get(code, f"Unknown error (code {code})")

    def get_command_history(self) -> list[dict]:
        """Return history of all commands."""
        return [
            {
                "type": c.command_type,
                "values": c.target_values,
                "speed": c.speed,
                "time": c.timestamp,
            }
            for c in self._command_history
        ]


# ---------------------------------------------------------------------------
# Oncology task definitions
# ---------------------------------------------------------------------------


@dataclass
class XArmOncologyTask:
    """Task specification for xArm 7 in oncology applications."""

    task: str
    obs_dim: int
    act_dim: int
    success_threshold: float
    max_force_n: float
    description: str = ""
    requires_vision: bool = False
    requires_gripper: bool = True


XARM_ONCOLOGY_TASKS = {
    "vial_handling": XArmOncologyTask(
        task="vial_handling",
        obs_dim=18,
        act_dim=8,
        success_threshold=0.99,
        max_force_n=1.5,
        description="Pick-and-place vials between racks and instruments",
        requires_vision=True,
        requires_gripper=True,
    ),
    "plate_stacking": XArmOncologyTask(
        task="plate_stacking",
        obs_dim=16,
        act_dim=8,
        success_threshold=0.99,
        max_force_n=2.0,
        description="Stack and unstack well plates for automated assays",
        requires_vision=True,
        requires_gripper=True,
    ),
    "pipette_operation": XArmOncologyTask(
        task="pipette_operation",
        obs_dim=22,
        act_dim=8,
        success_threshold=0.98,
        max_force_n=1.0,
        description="Operate electronic pipettes for liquid handling",
        requires_vision=True,
        requires_gripper=True,
    ),
    "equipment_loading": XArmOncologyTask(
        task="equipment_loading",
        obs_dim=20,
        act_dim=7,
        success_threshold=0.97,
        max_force_n=3.0,
        description="Load/unload centrifuges, analyzers, and imaging systems",
        requires_vision=True,
        requires_gripper=True,
    ),
}


# ---------------------------------------------------------------------------
# Cross-organization sharing
# ---------------------------------------------------------------------------


class XArmCrossOrgSharing:
    """Manage cross-organization sharing for xArm 7 robots.

    Supports sharing trained policies and configurations between
    xArm robots at different organizations for oncology trials.
    """

    SHARING_CAPABILITIES = [
        {
            "method": "ONNX Policy Export",
            "supported": True,
            "tested_with": [],
            "notes": "Standard ONNX export; less community validation than Franka/Kinova",
        },
        {
            "method": "ROS 2 Action Server",
            "supported": True,
            "tested_with": ["Franka Panda"],
            "notes": "xarm_ros2 provides standard ROS 2 interfaces",
        },
        {
            "method": "xArm Python SDK",
            "supported": True,
            "tested_with": ["xArm 5", "xArm 6", "xArm Lite 6"],
            "notes": "Intra-organization sharing via unified SDK API",
        },
        {
            "method": "MuJoCo Menagerie Models",
            "supported": True,
            "tested_with": ["Franka Panda", "Kinova Gen3"],
            "notes": "Standardized MJCF enables simulation-level policy transfer",
        },
    ]

    def __init__(self) -> None:
        self._shared_items: list[dict] = []

    def get_capabilities(self) -> list[dict]:
        """Return sharing capabilities."""
        return list(self.SHARING_CAPABILITIES)

    def get_intra_org_robots(self) -> list[str]:
        """Return UFACTORY robots compatible for intra-org sharing."""
        return [
            "UFACTORY xArm 5 (5-DOF)",
            "UFACTORY xArm 6 (6-DOF)",
            "UFACTORY xArm 7 (7-DOF)",
            "UFACTORY Lite 6 (6-DOF)",
            "UFACTORY 850 (6-DOF)",
        ]


# ---------------------------------------------------------------------------
# USL Evaluation for xArm 7
# ---------------------------------------------------------------------------


def evaluate_ufactory_xarm7() -> dict:
    """Run the full USL evaluation for the UFACTORY xArm 7.

    Returns:
        Dictionary containing scores, specs, and recommendations.
    """
    specs = XArm7Specs()
    sharing = XArmCrossOrgSharing()

    evaluation = {
        "robot": specs.name,
        "manufacturer": specs.manufacturer,
        "category": "Collaborative Robot (Cobot)",
        "specs": {
            "dof": specs.dof,
            "payload_kg": specs.payload_kg,
            "reach_mm": specs.reach_mm,
            "repeatability_mm": specs.repeatability_mm,
            "weight_kg": specs.weight_kg,
            "torque_sensors": specs.torque_sensors,
            "collision_detection": specs.collision_detection,
            "control_frequency_hz": specs.communication_frequency_hz,
            "max_control_frequency_hz": specs.max_communication_hz,
            "ip_rating": specs.ip_rating,
        },
        "joints": [
            {
                "index": j.joint_index,
                "name": j.name,
                "limits_deg": [math.degrees(j.lower_limit_rad), math.degrees(j.upper_limit_rad)],
                "max_torque_nm": j.max_torque_nm,
            }
            for j in XARM7_JOINTS
        ],
        "framework_support": {
            fw.value: {
                "model_format": cfg.model_format,
                "gpu_capable": cfg.gpu_capable,
                "ros2_package": cfg.ros2_package,
                "controllers": cfg.supported_controllers,
            }
            for fw, cfg in XARM_FRAMEWORK_CONFIGS.items()
        },
        "oncology_tasks": {
            k: {
                "description": v.description,
                "obs_dim": v.obs_dim,
                "act_dim": v.act_dim,
                "requires_vision": v.requires_vision,
            }
            for k, v in XARM_ONCOLOGY_TASKS.items()
        },
        "sharing_capabilities": sharing.get_capabilities(),
        "intra_org_robots": sharing.get_intra_org_robots(),
        "usl_scores": {
            "dimension_a_simulation_switching": 4.6,
            "dimension_b_ai_integration": 3.2,
            "dimension_c_cross_robot_sharing": 3.8,
            "dimension_d_clinical_trial_collab": 2.1,
            "final_score": 3.4,
        },
        "strengths": [
            "Most affordable 7-DOF cobot — lowest barrier to entry for new sites",
            "Built-in collision detection simplifies safety implementation",
            "IP51 rating — better environmental protection than competitors",
            "Wider operating temperature range (0-50 C)",
            "xArm Python SDK provides simple direct control interface",
            "MuJoCo Menagerie model now available for standardized simulation",
            "Strong intra-organization sharing across xArm family (5/6/7/Lite 6/850)",
        ],
        "gaps": [
            "Smallest open-source research community of the three cobots",
            "No published VLA or diffusion policy experiments",
            "Lower control frequency (250 Hz default vs 1 kHz competitors)",
            "No MCP, Claude Code, or Codex integration demonstrated",
            "Limited clinical/medical research publications",
            "No regulatory pathway documentation",
            "No federated learning demonstrations",
        ],
        "recommendations": [
            "Develop Isaac Lab USD asset for GPU-accelerated training",
            "Implement LLM-based task planning via xArm Python SDK",
            "Create cross-manufacturer ONNX policy transfer pipeline",
            "Build clinical deployment documentation (ISO 13482, IEC 62304)",
            "Leverage built-in collision detection for safety certification",
            "Establish xArm-to-Franka and xArm-to-Kinova policy sharing",
            "Publish oncology lab automation benchmarks to grow community",
        ],
    }

    return evaluation


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Demonstrate UFACTORY xArm 7 USL evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 65)
    print("  UFACTORY xArm 7 — USL Evaluation")
    print("=" * 65)

    evaluation = evaluate_ufactory_xarm7()
    print(json.dumps(evaluation, indent=2))

    print("\n--- xArm SDK Interface Demo ---")
    sdk = XArmSDKInterface(control_mode="position")
    home = [0.0, 0.0, 0.0, 90.0, 0.0, -90.0, 0.0]
    cmd = sdk.create_joint_command(home, speed_deg_s=30.0)
    print(f"  Command: {cmd.command_type} -> {cmd.target_values}")
    print(f"  History: {len(sdk.get_command_history())} commands")


if __name__ == "__main__":
    _demo()
