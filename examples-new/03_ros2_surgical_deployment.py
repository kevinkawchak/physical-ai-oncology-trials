"""
=============================================================================
EXAMPLE 03: ROS 2 Deployment Architecture for Surgical Robots
=============================================================================

WHAT THIS CODE DOES:
--------------------
Provides the complete ROS 2 node architecture for deploying a trained
surgical policy to a physical robot in an oncology operating room. This
includes lifecycle management, real-time control loop structure, state
machine for procedure phases, and integration patterns for dVRK and
Kinova hardware.

This is the code that runs on the robot's control computer during the
actual surgical procedure.

WHEN TO USE THIS:
-----------------
- You have a trained policy (from examples/ 01) and need to deploy it
- You are building the ROS 2 control stack for a surgical robot
- You need a state machine for managing procedure phases (setup, approach,
  operate, retract, emergency)
- You must integrate with dVRK 2.4.0 or Kinova Gen3 via ROS 2

HARDWARE REQUIREMENTS:
----------------------
    - ROS 2 Jazzy or Kilted on Ubuntu 24.04
    - Real-time kernel (PREEMPT_RT) for control loop guarantees
    - dVRK 2.4.0 (da Vinci) or Kinova Gen3 with ros2_control drivers
    - Force-torque sensor with ROS 2 driver
    - NVIDIA GPU (for policy inference if using neural network policy)

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - ROS 2 Jazzy Jalisco or Kilted Kaiju
    - ros2_control (for hardware abstraction)
    - NumPy 1.24.0+

Optional:
    - MoveIt 2 (for motion planning fallback)
    - dVRK 2.4.0 (sawIntuitiveResearchKit)
    - ONNX Runtime 1.17+ (for policy inference)
    - Nav2 lifecycle manager (for node lifecycle)

ARCHITECTURE NOTES:
-------------------
This example defines the node structure. In production, each class
would be a separate ROS 2 node or component in a composable container.

    Nodes in the surgical robot control stack:
    ==========================================
    1. PolicyInferenceNode    - Runs trained policy at control rate
    2. SafetyMonitorNode      - Checks all safety invariants (example 01)
    3. ProcedureStateMachine  - Manages procedure phase transitions
    4. HardwareInterfaceNode  - ros2_control interface to robot hardware
    5. PerceptionNode         - Sensor fusion (example 02)
    6. RecorderNode           - Logs all data for post-op analysis

    Data flow:
    ==========
    Sensors ──> Perception ──> Policy ──> SafetyMonitor ──> Hardware
                                 ^            |
                                 |            v
                              StateMachine  E-stop

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: PROCEDURE STATE MACHINE
# =============================================================================
# Manages the phases of a surgical procedure. The robot's behavior
# (velocity limits, autonomy level, allowed motions) changes with each phase.
#
# INSTRUCTIONS:
# - The state machine transitions are triggered by the surgeon via
#   foot pedal, voice command, or console button.
# - Some transitions are automatic (e.g., OPERATE -> EMERGENCY on fault).
# - Each state has different safety parameters (see SafetyLimits in ex 01).
# =============================================================================


class ProcedurePhase(Enum):
    """
    Surgical procedure phases.

    IDLE: Robot powered but not engaged. Used during OR setup.
    HOMING: Robot moving to home configuration. No patient contact.
    REGISTRATION: Patient-to-robot registration in progress (see ex 04).
    APPROACH: Robot moving toward surgical site. Slow, careful motion.
    OPERATE: Active surgical manipulation. Full autonomy or teleop.
    RETRACT: Robot retracting from surgical site. Controlled withdrawal.
    EMERGENCY: Fault detected. Robot holding position or powered off.
    COMPLETE: Procedure finished. Robot moves to park position.
    """

    IDLE = auto()
    HOMING = auto()
    REGISTRATION = auto()
    APPROACH = auto()
    OPERATE = auto()
    RETRACT = auto()
    EMERGENCY = auto()
    COMPLETE = auto()


@dataclass
class PhaseConfig:
    """
    Configuration for a procedure phase.

    Each phase has different velocity limits, autonomy levels, and
    allowed transitions. Set these based on clinical safety requirements.
    """

    max_velocity_m_s: float = 0.25
    max_force_n: float = 5.0
    autonomy_enabled: bool = False
    teleop_enabled: bool = True
    allowed_transitions: list = field(default_factory=list)
    requires_confirmation: bool = False


# Phase configurations: these define robot behavior in each phase
PHASE_CONFIGS: dict[ProcedurePhase, PhaseConfig] = {
    ProcedurePhase.IDLE: PhaseConfig(
        max_velocity_m_s=0.0,
        max_force_n=0.0,
        autonomy_enabled=False,
        teleop_enabled=False,
        allowed_transitions=[ProcedurePhase.HOMING],
    ),
    ProcedurePhase.HOMING: PhaseConfig(
        max_velocity_m_s=0.10,
        max_force_n=2.0,
        autonomy_enabled=True,
        teleop_enabled=False,
        allowed_transitions=[ProcedurePhase.REGISTRATION, ProcedurePhase.IDLE],
    ),
    ProcedurePhase.REGISTRATION: PhaseConfig(
        max_velocity_m_s=0.05,
        max_force_n=1.0,
        autonomy_enabled=False,
        teleop_enabled=True,
        allowed_transitions=[ProcedurePhase.APPROACH, ProcedurePhase.IDLE],
        requires_confirmation=True,
    ),
    ProcedurePhase.APPROACH: PhaseConfig(
        max_velocity_m_s=0.05,
        max_force_n=3.0,
        autonomy_enabled=True,
        teleop_enabled=True,
        allowed_transitions=[
            ProcedurePhase.OPERATE,
            ProcedurePhase.RETRACT,
        ],
    ),
    ProcedurePhase.OPERATE: PhaseConfig(
        max_velocity_m_s=0.25,
        max_force_n=5.0,
        autonomy_enabled=True,
        teleop_enabled=True,
        allowed_transitions=[
            ProcedurePhase.RETRACT,
            ProcedurePhase.APPROACH,
        ],
    ),
    ProcedurePhase.RETRACT: PhaseConfig(
        max_velocity_m_s=0.05,
        max_force_n=2.0,
        autonomy_enabled=True,
        teleop_enabled=False,
        allowed_transitions=[
            ProcedurePhase.APPROACH,
            ProcedurePhase.COMPLETE,
        ],
    ),
    ProcedurePhase.EMERGENCY: PhaseConfig(
        max_velocity_m_s=0.0,
        max_force_n=0.0,
        autonomy_enabled=False,
        teleop_enabled=False,
        allowed_transitions=[ProcedurePhase.IDLE],
        requires_confirmation=True,
    ),
    ProcedurePhase.COMPLETE: PhaseConfig(
        max_velocity_m_s=0.10,
        max_force_n=1.0,
        autonomy_enabled=True,
        teleop_enabled=False,
        allowed_transitions=[ProcedurePhase.IDLE],
    ),
}


class ProcedureStateMachine:
    """
    State machine for managing surgical procedure phases.

    INTEGRATION INSTRUCTIONS:
    -------------------------
    1. In ROS 2, this runs as a lifecycle node.
    2. Transitions are triggered by:
       - Surgeon foot pedal (via /surgeon/foot_pedal topic)
       - Voice command (via /surgeon/voice_command topic)
       - Automatic triggers (e.g., registration complete, fault detected)
    3. Each transition publishes to /procedure/phase for all nodes to consume.
    4. EMERGENCY transition is always allowed from any state.

    Example (ROS 2 integration):
        # In your ROS 2 node:
        self.phase_pub = self.create_publisher(String, '/procedure/phase', 10)
        self.sm = ProcedureStateMachine()
        self.sm.register_transition_callback(self._on_phase_change)

        def _on_phase_change(self, old_phase, new_phase, config):
            msg = String()
            msg.data = new_phase.name
            self.phase_pub.publish(msg)
    """

    def __init__(self):
        self._current_phase = ProcedurePhase.IDLE
        self._phase_configs = PHASE_CONFIGS
        self._transition_callbacks: list[Callable] = []
        self._transition_history: list[dict] = []

        logger.info("ProcedureStateMachine initialized in IDLE phase")

    @property
    def current_phase(self) -> ProcedurePhase:
        return self._current_phase

    @property
    def current_config(self) -> PhaseConfig:
        return self._phase_configs[self._current_phase]

    def register_transition_callback(
        self, callback: Callable[[ProcedurePhase, ProcedurePhase, PhaseConfig], None]
    ):
        """Register callback for phase transitions."""
        self._transition_callbacks.append(callback)

    def request_transition(
        self,
        target_phase: ProcedurePhase,
        operator_confirmed: bool = False,
    ) -> bool:
        """
        Request a phase transition.

        Args:
            target_phase: Desired next phase.
            operator_confirmed: Whether operator has confirmed (required
                for some transitions).

        Returns:
            True if transition succeeded, False if rejected.
        """
        # EMERGENCY is always allowed
        if target_phase == ProcedurePhase.EMERGENCY:
            return self._execute_transition(target_phase)

        # Check if transition is allowed
        config = self._phase_configs[self._current_phase]
        if target_phase not in config.allowed_transitions:
            logger.warning(
                "Transition %s -> %s not allowed",
                self._current_phase.name,
                target_phase.name,
            )
            return False

        # Check confirmation requirement
        target_config = self._phase_configs[target_phase]
        if target_config.requires_confirmation and not operator_confirmed:
            logger.warning(
                "Transition to %s requires operator confirmation",
                target_phase.name,
            )
            return False

        return self._execute_transition(target_phase)

    def _execute_transition(self, target_phase: ProcedurePhase) -> bool:
        """Execute a validated phase transition."""
        old_phase = self._current_phase
        new_config = self._phase_configs[target_phase]

        self._current_phase = target_phase
        self._transition_history.append(
            {
                "from": old_phase.name,
                "to": target_phase.name,
                "timestamp": time.monotonic(),
            }
        )

        logger.info(
            "Phase transition: %s -> %s (vel=%.2f m/s, force=%.1f N, "
            "autonomy=%s, teleop=%s)",
            old_phase.name,
            target_phase.name,
            new_config.max_velocity_m_s,
            new_config.max_force_n,
            new_config.autonomy_enabled,
            new_config.teleop_enabled,
        )

        for callback in self._transition_callbacks:
            try:
                callback(old_phase, target_phase, new_config)
            except Exception as e:
                logger.error("Transition callback failed: %s", e)

        return True


# =============================================================================
# SECTION 2: POLICY INFERENCE NODE
# =============================================================================
# Runs the trained neural network policy at the control loop rate.
# In production, this is a ROS 2 component with real-time priority.
#
# INSTRUCTIONS:
# - Load ONNX model exported from training (see examples/ 01).
# - Pin to a dedicated CPU core using cpu_affinity for deterministic timing.
# - Pre-allocate input/output tensors to avoid memory allocation in loop.
# - Measure inference latency; must be < 50% of control period.
# =============================================================================


class PolicyInferenceEngine:
    """
    Run trained surgical policy for real-time robot control.

    SETUP INSTRUCTIONS:
    -------------------
    1. Export policy to ONNX from training (examples/ 01, Sim2RealExporter).
    2. Verify ONNX model with onnxruntime.InferenceSession on target machine.
    3. Measure inference latency: must be <5 ms for 100 Hz control.
    4. If using GPU inference, ensure CUDA context is initialized before
       the control loop starts (first inference is slow).

    Observation normalization MUST match training:
    - Use the same running mean/std from training.
    - Mismatch causes policy to produce garbage actions.

    Example:
        >>> engine = PolicyInferenceEngine("policy.onnx")
        >>> engine.set_normalization(obs_mean, obs_std)
        >>> action = engine.infer(observation)
    """

    def __init__(self, model_path: str = "", obs_dim: int = 35, act_dim: int = 8):
        self.model_path = model_path
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._session = None
        self._obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self._obs_std = np.ones(obs_dim, dtype=np.float32)

        # Pre-allocate buffers to avoid allocation in control loop
        self._input_buffer = np.zeros((1, obs_dim), dtype=np.float32)
        self._output_buffer = np.zeros((1, act_dim), dtype=np.float32)

        self._load_model()

    def _load_model(self):
        """
        Load ONNX model for inference.

        In production:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1  # Deterministic timing
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                self.model_path, sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        """
        logger.info(
            "PolicyInferenceEngine: obs_dim=%d, act_dim=%d", self.obs_dim, self.act_dim
        )

    def set_normalization(self, obs_mean: np.ndarray, obs_std: np.ndarray):
        """
        Set observation normalization parameters from training.

        CRITICAL: These must exactly match the running statistics
        used during policy training. Typically saved alongside the
        policy checkpoint.
        """
        self._obs_mean = obs_mean.astype(np.float32)
        self._obs_std = np.maximum(obs_std.astype(np.float32), 1e-8)

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """
        Run policy inference.

        Args:
            observation: Raw observation vector (unnormalized).

        Returns:
            Action vector, clipped to [-1, 1].
        """
        # Normalize observation
        self._input_buffer[0] = (observation - self._obs_mean) / self._obs_std

        if self._session is not None:
            # ONNX Runtime inference
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(
                None, {input_name: self._input_buffer}
            )
            action = output[0].flatten()
        else:
            # Simulated inference
            action = np.tanh(self._input_buffer[0, : self.act_dim] * 0.1)

        return np.clip(action, -1.0, 1.0)


# =============================================================================
# SECTION 3: HARDWARE INTERFACE
# =============================================================================
# Abstraction layer for robot hardware via ros2_control.
#
# INSTRUCTIONS:
# - For dVRK: use crtk_msgs and dvrk_ros2 packages.
# - For Kinova Gen3: use kortex_driver.
# - For UR5e: use ur_robot_driver.
# - All interfaces expose the same command/state pattern.
# =============================================================================


class RobotHardwareInterface:
    """
    Hardware abstraction for surgical robot control via ros2_control.

    SETUP INSTRUCTIONS FOR dVRK:
    ----------------------------
    1. Launch dVRK ROS 2 console:
         ros2 launch dvrk_robot dvrk_robot.launch.py config:=jhu-daVinci
    2. Verify robot state:
         ros2 topic echo /PSM1/measured_cp
    3. Enable robot:
         ros2 service call /PSM1/enable std_srvs/srv/Trigger

    SETUP INSTRUCTIONS FOR KINOVA GEN3:
    ------------------------------------
    1. Launch Kinova driver:
         ros2 launch kortex_driver gen3.launch.py
    2. Switch to position controller:
         ros2 control switch_controllers --activate joint_trajectory_controller

    The interface sends commands at the control rate and reads state from
    the hardware at the sensor rate. Commands are position-based with
    velocity feed-forward.
    """

    def __init__(self, robot_type: str = "dvrk_psm", control_rate_hz: int = 100):
        self.robot_type = robot_type
        self.control_rate_hz = control_rate_hz
        self._is_enabled = False
        self._current_position_m = np.zeros(3)
        self._current_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._current_joint_positions = np.zeros(7)

        # ros2_control topic names by robot type
        self._topic_map = {
            "dvrk_psm": {
                "state": "/PSM1/measured_cp",
                "command": "/PSM1/servo_cp",
                "enable": "/PSM1/enable",
                "joint_state": "/PSM1/measured_js",
            },
            "kinova_gen3": {
                "state": "/joint_states",
                "command": "/joint_trajectory_controller/joint_trajectory",
                "enable": "/controller_manager/switch_controller",
                "joint_state": "/joint_states",
            },
            "ur5e": {
                "state": "/joint_states",
                "command": "/scaled_joint_trajectory_controller/joint_trajectory",
                "enable": "/dashboard_client/play",
                "joint_state": "/joint_states",
            },
        }

        logger.info(
            "HardwareInterface: robot=%s, rate=%d Hz",
            robot_type,
            control_rate_hz,
        )

    def enable(self) -> bool:
        """
        Enable robot for motion.

        In production, this calls the robot's enable service and waits
        for confirmation. For dVRK, this homes the robot and enables
        servo mode.
        """
        logger.info("Enabling robot hardware: %s", self.robot_type)
        self._is_enabled = True
        return True

    def disable(self):
        """Disable robot (safe state, no motion)."""
        logger.info("Disabling robot hardware")
        self._is_enabled = False

    def send_cartesian_command(
        self,
        position_m: np.ndarray,
        orientation_quat: np.ndarray,
        velocity_m_s: np.ndarray | None = None,
    ) -> bool:
        """
        Send Cartesian pose command to robot.

        Args:
            position_m: Target [x, y, z] in robot base frame.
            orientation_quat: Target [w, x, y, z] quaternion.
            velocity_m_s: Optional velocity feed-forward.

        Returns:
            True if command accepted, False if rejected (not enabled, etc.).

        In production (dVRK):
            msg = TransformStamped()
            msg.transform.translation.x = position_m[0]
            msg.transform.translation.y = position_m[1]
            msg.transform.translation.z = position_m[2]
            msg.transform.rotation.w = orientation_quat[0]
            # ... etc
            self.servo_cp_pub.publish(msg)
        """
        if not self._is_enabled:
            return False

        self._current_position_m = position_m.copy()
        self._current_orientation_quat = orientation_quat.copy()
        return True

    def get_state(self) -> dict:
        """
        Read current robot state from hardware.

        Returns:
            Dictionary with position, orientation, joints, velocities.
        """
        return {
            "position_m": self._current_position_m.copy(),
            "orientation_quat": self._current_orientation_quat.copy(),
            "joint_positions_rad": self._current_joint_positions.copy(),
            "is_enabled": self._is_enabled,
        }


# =============================================================================
# SECTION 4: CONTROL LOOP
# =============================================================================
# The main real-time control loop that ties everything together.
# Runs at the control rate (typically 100 Hz for surgical robots).
#
# INSTRUCTIONS:
# - This loop must run in a real-time thread (SCHED_FIFO priority 90+).
# - Use mlockall() to prevent page faults.
# - Do NOT allocate memory inside the loop.
# - Do NOT perform I/O (file writes, network) inside the loop.
# - Log data to a lock-free ring buffer, drain in a separate thread.
# =============================================================================


class SurgicalControlLoop:
    """
    Main real-time control loop for surgical robot deployment.

    This is the core of the robot control system. It runs at a fixed
    rate and coordinates policy inference, safety checking, and
    hardware commanding.

    REAL-TIME SETUP INSTRUCTIONS:
    -----------------------------
    1. Set PREEMPT_RT kernel:
         sudo apt install linux-image-rt-amd64
    2. Set real-time priority for control process:
         sudo chrt -f 90 python3 control_node.py
    3. Lock memory:
         import ctypes
         libc = ctypes.CDLL('libc.so.6')
         libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
    4. Set CPU affinity to isolate control loop:
         import os
         os.sched_setaffinity(0, {2})  # Pin to CPU core 2

    Example:
        >>> loop = SurgicalControlLoop(
        ...     robot=hardware_interface,
        ...     policy=policy_engine,
        ...     state_machine=procedure_sm,
        ...     control_rate_hz=100,
        ... )
        >>> loop.run()  # Blocks until procedure completes
    """

    def __init__(
        self,
        robot: RobotHardwareInterface,
        policy: PolicyInferenceEngine,
        state_machine: ProcedureStateMachine,
        control_rate_hz: int = 100,
    ):
        self.robot = robot
        self.policy = policy
        self.state_machine = state_machine
        self.control_rate_hz = control_rate_hz
        self._dt_s = 1.0 / control_rate_hz

        # Telemetry
        self._cycle_count = 0
        self._max_jitter_us = 0.0
        self._total_jitter_us = 0.0
        self._loop_running = False

        # Data recording buffer (lock-free ring buffer in production)
        self._recorded_data: list[dict] = []

        logger.info(
            "SurgicalControlLoop: rate=%d Hz, dt=%.4f s",
            control_rate_hz,
            self._dt_s,
        )

    def run(self, max_cycles: int = 0) -> dict:
        """
        Run the control loop.

        Args:
            max_cycles: Maximum number of cycles (0 = run until COMPLETE/EMERGENCY).

        Returns:
            Summary statistics of the run.
        """
        logger.info("Starting control loop at %d Hz", self.control_rate_hz)
        self._loop_running = True

        try:
            while self._loop_running:
                cycle_start = time.monotonic()
                self._cycle_count += 1

                # --- Read sensors ---
                robot_state = self.robot.get_state()

                # --- Build observation ---
                observation = self._build_observation(robot_state)

                # --- Policy inference (only if autonomy enabled) ---
                phase_config = self.state_machine.current_config
                if phase_config.autonomy_enabled:
                    action = self.policy.infer(observation)
                    target_position, target_orientation = self._action_to_command(
                        action, robot_state
                    )
                else:
                    # Hold current position when autonomy disabled
                    target_position = robot_state["position_m"]
                    target_orientation = robot_state["orientation_quat"]

                # --- Apply velocity limit from current phase ---
                target_position = self._apply_velocity_limit(
                    robot_state["position_m"],
                    target_position,
                    phase_config.max_velocity_m_s,
                )

                # --- Send command to hardware ---
                self.robot.send_cartesian_command(
                    position_m=target_position,
                    orientation_quat=target_orientation,
                )

                # --- Record data ---
                self._record_cycle(
                    robot_state, target_position, phase_config
                )

                # --- Check termination ---
                if max_cycles > 0 and self._cycle_count >= max_cycles:
                    break
                if self.state_machine.current_phase in (
                    ProcedurePhase.COMPLETE,
                    ProcedurePhase.EMERGENCY,
                ):
                    if self._cycle_count > 10:
                        break

                # --- Sleep for remaining cycle time ---
                elapsed = time.monotonic() - cycle_start
                sleep_time = self._dt_s - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # --- Track jitter ---
                actual_dt = time.monotonic() - cycle_start
                jitter_us = abs(actual_dt - self._dt_s) * 1e6
                self._max_jitter_us = max(self._max_jitter_us, jitter_us)
                self._total_jitter_us += jitter_us

        finally:
            self._loop_running = False
            self.robot.disable()

        return self._get_run_summary()

    def stop(self):
        """Signal the control loop to stop."""
        self._loop_running = False

    def _build_observation(self, robot_state: dict) -> np.ndarray:
        """
        Build observation vector from sensor data.

        The observation must match the training observation space exactly.
        See examples/ 01 OncologySurgicalEnv._get_observation_dim() for
        the expected format.

        Typical observation for surgical manipulation:
        [0:3]   End-effector position (x, y, z)
        [3:7]   End-effector orientation (quaternion)
        [7:14]  Joint positions (7 DOF)
        [14:17] Goal position
        [17:21] Goal orientation
        [21:27] Force-torque (Fx, Fy, Fz, Tx, Ty, Tz)
        [27:35] Task-specific (needle pose, tissue state, etc.)
        """
        obs = np.zeros(self.policy.obs_dim, dtype=np.float32)

        # End-effector pose
        obs[0:3] = robot_state["position_m"]
        obs[3:7] = robot_state["orientation_quat"]

        # Joint positions
        n_joints = min(7, len(robot_state["joint_positions_rad"]))
        obs[7 : 7 + n_joints] = robot_state["joint_positions_rad"][:n_joints]

        # Goal (in production, from surgical plan)
        obs[14:17] = np.array([0.0, 0.0, -0.15])  # Target position

        return obs

    def _action_to_command(
        self, action: np.ndarray, robot_state: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert policy action to robot command.

        Actions are relative displacements in [-1, 1], scaled by
        action_scale to produce absolute positions.

        INSTRUCTIONS:
        - action_scale must match the training environment's action space.
        - For dVRK PSM: action_scale = 0.001 m (1 mm per max action step).
        - Actions are in robot base frame (same as training).
        """
        action_scale = 0.001  # 1 mm per action step

        current_pos = robot_state["position_m"]
        delta_position = action[:3] * action_scale
        target_position = current_pos + delta_position

        # Orientation: keep current for now
        target_orientation = robot_state["orientation_quat"]

        return target_position, target_orientation

    def _apply_velocity_limit(
        self,
        current_m: np.ndarray,
        target_m: np.ndarray,
        max_vel_m_s: float,
    ) -> np.ndarray:
        """
        Limit position command to respect velocity constraint.

        The maximum displacement per cycle is:
            max_step = max_velocity * dt
        """
        if max_vel_m_s <= 0:
            return current_m

        max_step = max_vel_m_s * self._dt_s
        delta = target_m - current_m
        dist = float(np.linalg.norm(delta))

        if dist > max_step and dist > 0:
            direction = delta / dist
            return current_m + direction * max_step

        return target_m

    def _record_cycle(
        self,
        robot_state: dict,
        target_position: np.ndarray,
        phase_config: PhaseConfig,
    ):
        """
        Record control cycle data for post-operative analysis.

        In production, write to a lock-free ring buffer that is
        drained by a separate non-RT thread to disk (ROS 2 bag or CSV).
        """
        if self._cycle_count % 10 == 0:  # Record every 10th cycle to limit data
            self._recorded_data.append(
                {
                    "cycle": self._cycle_count,
                    "phase": self.state_machine.current_phase.name,
                    "position_m": robot_state["position_m"].tolist(),
                    "target_m": target_position.tolist(),
                }
            )

    def _get_run_summary(self) -> dict:
        """Get control loop run summary."""
        avg_jitter = (
            self._total_jitter_us / self._cycle_count
            if self._cycle_count > 0
            else 0
        )
        return {
            "total_cycles": self._cycle_count,
            "max_jitter_us": self._max_jitter_us,
            "avg_jitter_us": avg_jitter,
            "final_phase": self.state_machine.current_phase.name,
            "recorded_samples": len(self._recorded_data),
        }


# =============================================================================
# SECTION 5: LAUNCH CONFIGURATION
# =============================================================================
# ROS 2 launch file structure for the surgical robot control stack.
# This would be a .py launch file in production.
# =============================================================================


def generate_launch_config(robot_type: str = "dvrk_psm") -> dict:
    """
    Generate ROS 2 launch configuration for surgical deployment.

    In production, this would be a ROS 2 launch file:
        ros2 launch surgical_robot surgical_bringup.launch.py \\
            robot_type:=dvrk_psm \\
            policy_path:=policies/needle_insertion.onnx \\
            control_rate:=100

    Returns:
        Dictionary describing the launch configuration.
    """
    config = {
        "robot_type": robot_type,
        "nodes": [
            {
                "name": "hardware_interface",
                "package": "surgical_robot",
                "executable": "hardware_interface_node",
                "parameters": {
                    "robot_type": robot_type,
                    "control_rate_hz": 100,
                },
                "remappings": [],
            },
            {
                "name": "policy_inference",
                "package": "surgical_robot",
                "executable": "policy_inference_node",
                "parameters": {
                    "model_path": "policies/needle_insertion.onnx",
                    "obs_dim": 35,
                    "act_dim": 8,
                },
            },
            {
                "name": "safety_monitor",
                "package": "surgical_robot",
                "executable": "safety_monitor_node",
                "parameters": {
                    "force_max_n": 5.0,
                    "workspace_radius_m": 0.12,
                    "watchdog_timeout_ms": 12.0,
                },
            },
            {
                "name": "procedure_sm",
                "package": "surgical_robot",
                "executable": "procedure_state_machine_node",
                "parameters": {},
            },
            {
                "name": "perception",
                "package": "surgical_robot",
                "executable": "perception_node",
                "parameters": {
                    "camera_topic": "/endoscope/image_raw",
                    "depth_topic": "/endoscope/depth",
                },
            },
            {
                "name": "recorder",
                "package": "surgical_robot",
                "executable": "recorder_node",
                "parameters": {
                    "output_dir": "/data/procedure_logs/",
                    "topics": [
                        "/joint_states",
                        "/ft_sensor/wrench",
                        "/procedure/phase",
                        "/policy/action",
                    ],
                },
            },
        ],
        "composable_container": {
            "name": "surgical_container",
            "package": "rclcpp_components",
            "executable": "component_container_mt",
            "components": [
                "surgical_robot::HardwareInterfaceComponent",
                "surgical_robot::PolicyInferenceComponent",
                "surgical_robot::SafetyMonitorComponent",
            ],
        },
    }
    return config


# =============================================================================
# SECTION 6: MAIN DEMONSTRATION
# =============================================================================


def run_ros2_deployment_demo():
    """
    Demonstrate the ROS 2 surgical deployment architecture.

    Simulates a procedure from IDLE through OPERATE to COMPLETE,
    showing state transitions and control loop operation.
    """
    logger.info("=" * 70)
    logger.info("ROS 2 SURGICAL ROBOT DEPLOYMENT ARCHITECTURE")
    logger.info("=" * 70)

    # --- Initialize components ---
    robot = RobotHardwareInterface(robot_type="dvrk_psm", control_rate_hz=100)
    policy = PolicyInferenceEngine(obs_dim=35, act_dim=8)
    state_machine = ProcedureStateMachine()

    # --- Create control loop ---
    control_loop = SurgicalControlLoop(
        robot=robot,
        policy=policy,
        state_machine=state_machine,
        control_rate_hz=100,
    )

    # --- Simulate procedure phase transitions ---
    logger.info("\n--- Simulating procedure phase transitions ---")
    transitions = [
        (ProcedurePhase.HOMING, False),
        (ProcedurePhase.REGISTRATION, False),
        (ProcedurePhase.APPROACH, True),
        (ProcedurePhase.OPERATE, False),
    ]

    for target, needs_confirm in transitions:
        success = state_machine.request_transition(
            target, operator_confirmed=needs_confirm
        )
        if not success:
            logger.warning("Transition to %s failed", target.name)

    # Enable robot
    robot.enable()

    # Run control loop for 50 cycles (0.5 seconds at 100 Hz)
    logger.info("\n--- Running control loop for 50 cycles ---")
    summary = control_loop.run(max_cycles=50)

    # Transition to COMPLETE
    state_machine.request_transition(ProcedurePhase.RETRACT)
    state_machine.request_transition(ProcedurePhase.COMPLETE)

    # --- Print launch configuration ---
    launch_config = generate_launch_config("dvrk_psm")

    # --- Print results ---
    print("\n" + "=" * 60)
    print("ROS 2 DEPLOYMENT RESULTS")
    print("=" * 60)
    print(f"Total control cycles: {summary['total_cycles']}")
    print(f"Max jitter:           {summary['max_jitter_us']:.0f} us")
    print(f"Avg jitter:           {summary['avg_jitter_us']:.0f} us")
    print(f"Final phase:          {summary['final_phase']}")
    print(f"Recorded samples:     {summary['recorded_samples']}")
    print(f"\nROS 2 nodes required: {len(launch_config['nodes'])}")
    for node in launch_config["nodes"]:
        print(f"  - {node['name']} ({node['package']})")

    return summary


if __name__ == "__main__":
    run_ros2_deployment_demo()
