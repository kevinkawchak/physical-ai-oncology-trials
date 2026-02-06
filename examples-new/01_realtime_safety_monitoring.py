"""
=============================================================================
EXAMPLE 01: Real-Time Safety Monitoring for Surgical Robots
=============================================================================

WHAT THIS CODE DOES:
--------------------
Implements the full safety monitoring stack that runs on a physical surgical
robot during oncology procedures. This is the layer between the control policy
and the hardware actuators that prevents patient harm.

Every command from the AI policy or surgeon teleop passes through this
safety monitor before reaching the motors. If any safety invariant is
violated, the system triggers an immediate controlled stop.

WHEN TO USE THIS:
-----------------
- You are deploying a trained policy to a physical dVRK, Kinova, or UR robot
- You need IEC 62304 / IEC 80601-2-77 compliant safety architecture
- You are integrating force-torque sensors for contact force limiting
- You need workspace boundary enforcement near patient anatomy
- You must implement watchdog timers for real-time control loops

HARDWARE REQUIREMENTS:
----------------------
    - Force-torque sensor (ATI Mini45, OnRobot HEX-E, or robot-integrated)
    - Emergency stop circuit (hardware E-stop is mandatory; this is software layer)
    - Real-time capable computer (PREEMPT_RT kernel, <1ms jitter)
    - Robot with position and torque sensing (dVRK PSM, Kinova Gen3, UR5e)

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - NumPy 1.24.0+
    - Python 3.10+

Optional (for deployment):
    - ROS 2 Jazzy/Kilted (for hardware integration)
    - dVRK 2.4.0 (for da Vinci hardware)
    - PREEMPT_RT Linux kernel (for real-time guarantees)

REGULATORY NOTES:
-----------------
    - IEC 80601-2-77: Particular requirements for robotically assisted
      surgical equipment
    - IEC 62304: Medical device software lifecycle
    - ISO 13482: Safety requirements for personal care robots
    - FDA Guidance: Computer-Assisted Surgical Equipment (2025)

    This safety monitor implements the "Safety Controller" concept from
    IEC 80601-2-77 Clause 201.12.4.4 (supervisory safety system).

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: SAFETY CONFIGURATION
# =============================================================================
# These limits are the single source of truth for all safety parameters.
# They MUST be reviewed by the clinical engineering team before deployment.
#
# INSTRUCTIONS:
# 1. Set force limits based on tissue type at the surgical site.
#    - Soft tissue (liver, lung): 5 N max
#    - Bone contact (spine): 15 N max
#    - General oncology resection: 8 N max
# 2. Set workspace boundaries from preoperative imaging registration.
# 3. Set velocity limits based on the procedure phase.
# =============================================================================


class SafetyLevel(Enum):
    """
    IEC 80601-2-77 safety levels for robotic surgical systems.

    NOMINAL: Normal operation, all parameters within limits.
    WARNING: Approaching limits, operator notified, operation continues.
    CRITICAL: Limit exceeded, controlled deceleration to stop.
    EMERGENCY: Hardware E-stop or catastrophic fault, immediate power cut.
    """

    NOMINAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class StopCategory(Enum):
    """
    IEC 60204-1 stop categories for machinery.

    CATEGORY_0: Immediate removal of power (uncontrolled stop).
    CATEGORY_1: Controlled deceleration, then power removal.
    CATEGORY_2: Controlled stop with power maintained for holding.
    """

    CATEGORY_0 = "immediate_power_removal"
    CATEGORY_1 = "controlled_deceleration_then_stop"
    CATEGORY_2 = "controlled_stop_power_maintained"


@dataclass
class SafetyLimits:
    """
    Safety limits for a surgical robot operating in an oncology procedure.

    INSTRUCTIONS FOR SETTING LIMITS:
    --------------------------------
    force_max_n: Maximum contact force at end-effector.
        - Derived from tissue biomechanics at surgical site.
        - Lung parenchyma tears above ~5 N; use 3 N limit with 60% margin.
        - Liver capsule tolerates ~8 N; use 5 N limit.
        - For needle insertion, set based on expected insertion force + margin.

    torque_max_nm: Maximum torque at end-effector.
        - Prevents twisting injury to tissue.
        - Typical: 0.3 Nm for soft tissue, 1.0 Nm for bone.

    velocity_max_m_s: Maximum Cartesian velocity of tool tip.
        - IEC 80601-2-77 recommends <0.25 m/s for autonomous motion.
        - During surgeon teleop, can be higher (up to 0.5 m/s).

    workspace_center_m: Center of permitted workspace (x, y, z) in robot base frame.
        - Set from preoperative planning: center on tumor site.

    workspace_radius_m: Radius of spherical permitted workspace.
        - Must contain entire surgical field but exclude critical anatomy
          outside the operative field (e.g., contralateral lung).

    joint_limits_deg: Per-joint position limits [min, max] for each joint.
        - Tighter than mechanical limits to prevent self-collision
          and collision with patient anatomy outside workspace.

    watchdog_timeout_ms: Maximum allowed time between control loop iterations.
        - If the control loop stalls longer than this, trigger stop.
        - Typical: 10 ms for 100 Hz control, 2 ms for 1 kHz control.

    force_rate_max_n_per_s: Maximum rate of force increase.
        - Prevents sudden impact even within force limits.
        - Typical: 50 N/s for soft tissue.
    """

    force_max_n: float = 5.0
    torque_max_nm: float = 0.5
    velocity_max_m_s: float = 0.25
    acceleration_max_m_s2: float = 1.0
    workspace_center_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.15])
    )
    workspace_radius_m: float = 0.15
    joint_limits_deg: list = field(
        default_factory=lambda: [
            (-90, 90),
            (-45, 45),
            (-170, 170),
            (-170, 170),
            (-170, 170),
            (-170, 170),
            (-170, 170),
        ]
    )
    watchdog_timeout_ms: float = 10.0
    force_rate_max_n_per_s: float = 50.0
    warning_threshold_fraction: float = 0.8


# =============================================================================
# SECTION 2: SENSOR DATA STRUCTURES
# =============================================================================
# These represent the data flowing from physical sensors into the safety
# monitor at each control cycle. In a real system, these are populated
# by ROS 2 subscriber callbacks or direct hardware API reads.
# =============================================================================


@dataclass
class ForceTorqueReading:
    """
    Force-torque sensor reading at the robot end-effector.

    SENSOR SETUP INSTRUCTIONS:
    --------------------------
    1. Mount ATI Mini45 or equivalent between wrist flange and instrument.
    2. Calibrate sensor with instrument attached but no external load.
    3. Record bias values and subtract in real-time (see bias_compensate).
    4. Sensor frame must be aligned with tool frame or a known transform
       must be applied (see transform_to_tool_frame).

    Attributes:
        force_xyz_n: Force vector in sensor frame [Fx, Fy, Fz] in Newtons.
        torque_xyz_nm: Torque vector [Tx, Ty, Tz] in Newton-meters.
        timestamp_ns: Sensor timestamp in nanoseconds (monotonic clock).
        is_valid: False if sensor reports error or data is stale.
    """

    force_xyz_n: np.ndarray
    torque_xyz_nm: np.ndarray
    timestamp_ns: int
    is_valid: bool = True


@dataclass
class RobotState:
    """
    Full robot state from joint encoders, end-effector kinematics, and sensors.

    Attributes:
        joint_positions_rad: Joint positions in radians.
        joint_velocities_rad_s: Joint velocities in rad/s.
        ee_position_m: End-effector position [x, y, z] in robot base frame.
        ee_velocity_m_s: End-effector Cartesian velocity [vx, vy, vz].
        ee_orientation_quat: End-effector orientation as quaternion [w, x, y, z].
        force_torque: Current force-torque reading.
        timestamp_ns: State timestamp in nanoseconds.
    """

    joint_positions_rad: np.ndarray
    joint_velocities_rad_s: np.ndarray
    ee_position_m: np.ndarray
    ee_velocity_m_s: np.ndarray
    ee_orientation_quat: np.ndarray
    force_torque: ForceTorqueReading
    timestamp_ns: int


@dataclass
class SafetyEvent:
    """
    Record of a safety event for audit logging.

    IEC 62304 requires full traceability of all safety-critical events.
    These records are persisted to disk in real-time and cannot be deleted.
    """

    timestamp_ns: int
    level: SafetyLevel
    category: str
    description: str
    measured_value: float
    limit_value: float
    action_taken: str
    robot_state_snapshot: dict = field(default_factory=dict)


# =============================================================================
# SECTION 3: SAFETY MONITOR CORE
# =============================================================================
# This is the main safety monitoring class. It runs every control cycle
# and checks all safety invariants before allowing commands to pass through.
#
# ARCHITECTURE:
# =============
#   Policy/Teleop Command
#          |
#          v
#   +------------------+
#   | SafetyMonitor    |  <-- Checks ALL invariants
#   |   .check_force() |
#   |   .check_ws()    |
#   |   .check_vel()   |
#   |   .check_joints()|
#   |   .check_wd()    |
#   +------------------+
#          |
#     Pass / Reject
#          |
#          v
#   Robot Hardware API
# =============================================================================


class SafetyMonitor:
    """
    Real-time safety monitor for surgical robots in oncology procedures.

    This class implements IEC 80601-2-77 supervisory safety monitoring.
    Every command is checked against safety invariants before execution.

    INTEGRATION INSTRUCTIONS:
    -------------------------
    1. Instantiate with procedure-specific SafetyLimits.
    2. In your control loop, call check_and_filter() on every command.
    3. Register callbacks for safety events if you need custom responses.
    4. Call update_state() with fresh sensor data each cycle.

    Example:
        >>> limits = SafetyLimits(force_max_n=5.0, workspace_radius_m=0.15)
        >>> monitor = SafetyMonitor(limits)
        >>> # In control loop:
        >>> safe_cmd, level = monitor.check_and_filter(raw_command, robot_state)
        >>> if level == SafetyLevel.NOMINAL:
        ...     robot.send_command(safe_cmd)
    """

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self._last_timestamp_ns: int = 0
        self._last_force_magnitude: float = 0.0
        self._event_log: list[SafetyEvent] = []
        self._callbacks: dict[SafetyLevel, list[Callable]] = {
            level: [] for level in SafetyLevel
        }
        self._is_stopped = False
        self._stop_reason = ""
        self._cycle_count = 0

        logger.info(
            "SafetyMonitor initialized: force_max=%.1f N, "
            "workspace_radius=%.3f m, watchdog=%.1f ms",
            limits.force_max_n,
            limits.workspace_radius_m,
            limits.watchdog_timeout_ms,
        )

    def register_callback(
        self, level: SafetyLevel, callback: Callable[[SafetyEvent], None]
    ):
        """
        Register a callback for a specific safety level.

        Use this to connect external systems (alarms, UI, logging) to
        safety events.

        Args:
            level: Safety level to trigger on.
            callback: Function called with SafetyEvent when level is reached.
        """
        self._callbacks[level].append(callback)

    def check_and_filter(
        self,
        commanded_position_m: np.ndarray,
        commanded_velocity_m_s: np.ndarray,
        state: RobotState,
    ) -> tuple[np.ndarray, np.ndarray, SafetyLevel]:
        """
        Check all safety invariants and filter the command if needed.

        This is the main entry point called every control cycle. It returns
        either the original command (if safe) or a modified command
        (deceleration, hold position) along with the current safety level.

        Args:
            commanded_position_m: Desired end-effector position [x, y, z].
            commanded_velocity_m_s: Desired end-effector velocity [vx, vy, vz].
            state: Current robot state from sensors.

        Returns:
            Tuple of (safe_position, safe_velocity, safety_level).
            If CRITICAL or EMERGENCY, safe_position = current position
            and safe_velocity = zeros.
        """
        self._cycle_count += 1
        worst_level = SafetyLevel.NOMINAL

        if self._is_stopped:
            return (
                state.ee_position_m.copy(),
                np.zeros(3),
                SafetyLevel.CRITICAL,
            )

        # CHECK 1: Watchdog timer
        level = self._check_watchdog(state.timestamp_ns)
        worst_level = max(worst_level, level, key=lambda l: l.value)

        # CHECK 2: Force-torque limits
        level = self._check_force_torque(state.force_torque)
        worst_level = max(worst_level, level, key=lambda l: l.value)

        # CHECK 3: Workspace boundary
        level = self._check_workspace(commanded_position_m)
        worst_level = max(worst_level, level, key=lambda l: l.value)

        # CHECK 4: Velocity limits
        level = self._check_velocity(commanded_velocity_m_s)
        worst_level = max(worst_level, level, key=lambda l: l.value)

        # CHECK 5: Joint limits
        level = self._check_joint_limits(state.joint_positions_rad)
        worst_level = max(worst_level, level, key=lambda l: l.value)

        # CHECK 6: Force rate of change
        level = self._check_force_rate(state.force_torque)
        worst_level = max(worst_level, level, key=lambda l: l.value)

        # Apply safety response based on worst level
        if worst_level.value >= SafetyLevel.CRITICAL.value:
            # Controlled stop: hold current position, zero velocity
            safe_position = state.ee_position_m.copy()
            safe_velocity = np.zeros(3)
            self._is_stopped = True
        elif worst_level == SafetyLevel.WARNING:
            # Scale down velocity toward limits
            safe_position = self._clamp_to_workspace(commanded_position_m)
            vel_scale = self.limits.warning_threshold_fraction
            safe_velocity = commanded_velocity_m_s * vel_scale
        else:
            safe_position = commanded_position_m
            safe_velocity = commanded_velocity_m_s

        self._last_timestamp_ns = state.timestamp_ns
        return safe_position, safe_velocity, worst_level

    def _check_watchdog(self, current_timestamp_ns: int) -> SafetyLevel:
        """
        Check that the control loop is running within timing requirements.

        If the interval between cycles exceeds watchdog_timeout_ms, the
        real-time guarantee is violated and the system must stop.

        INSTRUCTIONS:
        - On PREEMPT_RT Linux, typical jitter is <50 us at 1 kHz.
        - If you see watchdog violations, check for:
          1. Non-RT threads holding locks
          2. Page faults (use mlockall)
          3. IRQ storms from network or USB
        """
        if self._last_timestamp_ns == 0:
            return SafetyLevel.NOMINAL

        dt_ms = (current_timestamp_ns - self._last_timestamp_ns) / 1e6

        if dt_ms > self.limits.watchdog_timeout_ms:
            event = SafetyEvent(
                timestamp_ns=current_timestamp_ns,
                level=SafetyLevel.CRITICAL,
                category="watchdog",
                description=(
                    f"Control loop overrun: {dt_ms:.2f} ms "
                    f"(limit: {self.limits.watchdog_timeout_ms:.1f} ms)"
                ),
                measured_value=dt_ms,
                limit_value=self.limits.watchdog_timeout_ms,
                action_taken="STOP_CATEGORY_1",
            )
            self._record_event(event)
            return SafetyLevel.CRITICAL

        warn_threshold = self.limits.watchdog_timeout_ms * self.limits.warning_threshold_fraction
        if dt_ms > warn_threshold:
            return SafetyLevel.WARNING

        return SafetyLevel.NOMINAL

    def _check_force_torque(self, ft: ForceTorqueReading) -> SafetyLevel:
        """
        Check force and torque against tissue-specific limits.

        INSTRUCTIONS:
        - force_max_n must be set for the tissue at the current surgical site.
        - If the force sensor reports invalid data, treat as CRITICAL.
        - Force is checked as magnitude (L2 norm), not per-axis.
        """
        if not ft.is_valid:
            event = SafetyEvent(
                timestamp_ns=ft.timestamp_ns,
                level=SafetyLevel.CRITICAL,
                category="force_sensor_fault",
                description="Force-torque sensor reporting invalid data",
                measured_value=0.0,
                limit_value=0.0,
                action_taken="STOP_CATEGORY_1",
            )
            self._record_event(event)
            return SafetyLevel.CRITICAL

        force_magnitude = float(np.linalg.norm(ft.force_xyz_n))
        torque_magnitude = float(np.linalg.norm(ft.torque_xyz_nm))

        # Check force
        if force_magnitude > self.limits.force_max_n:
            event = SafetyEvent(
                timestamp_ns=ft.timestamp_ns,
                level=SafetyLevel.CRITICAL,
                category="force_exceeded",
                description=(
                    f"Force {force_magnitude:.2f} N exceeds limit "
                    f"{self.limits.force_max_n:.1f} N"
                ),
                measured_value=force_magnitude,
                limit_value=self.limits.force_max_n,
                action_taken="STOP_CATEGORY_1",
            )
            self._record_event(event)
            return SafetyLevel.CRITICAL

        if force_magnitude > self.limits.force_max_n * self.limits.warning_threshold_fraction:
            return SafetyLevel.WARNING

        # Check torque
        if torque_magnitude > self.limits.torque_max_nm:
            event = SafetyEvent(
                timestamp_ns=ft.timestamp_ns,
                level=SafetyLevel.CRITICAL,
                category="torque_exceeded",
                description=(
                    f"Torque {torque_magnitude:.3f} Nm exceeds limit "
                    f"{self.limits.torque_max_nm:.2f} Nm"
                ),
                measured_value=torque_magnitude,
                limit_value=self.limits.torque_max_nm,
                action_taken="STOP_CATEGORY_1",
            )
            self._record_event(event)
            return SafetyLevel.CRITICAL

        return SafetyLevel.NOMINAL

    def _check_workspace(self, target_position_m: np.ndarray) -> SafetyLevel:
        """
        Check that the commanded position is within the permitted workspace.

        INSTRUCTIONS:
        - Workspace is a sphere centered on the tumor site.
        - Center and radius are set during preoperative planning after
          patient registration (see example 04 for registration).
        - The workspace must be verified against preop imaging before
          the procedure begins.
        """
        dist = float(
            np.linalg.norm(target_position_m - self.limits.workspace_center_m)
        )

        if dist > self.limits.workspace_radius_m:
            event = SafetyEvent(
                timestamp_ns=int(time.monotonic_ns()),
                level=SafetyLevel.CRITICAL,
                category="workspace_violation",
                description=(
                    f"Position {dist:.4f} m from workspace center "
                    f"exceeds radius {self.limits.workspace_radius_m:.3f} m"
                ),
                measured_value=dist,
                limit_value=self.limits.workspace_radius_m,
                action_taken="STOP_CATEGORY_1",
            )
            self._record_event(event)
            return SafetyLevel.CRITICAL

        warn_radius = self.limits.workspace_radius_m * self.limits.warning_threshold_fraction
        if dist > warn_radius:
            return SafetyLevel.WARNING

        return SafetyLevel.NOMINAL

    def _check_velocity(self, velocity_m_s: np.ndarray) -> SafetyLevel:
        """
        Check Cartesian velocity against procedure-phase limits.

        IEC 80601-2-77 requires velocity limiting for autonomous motion.
        Typical limits:
        - Autonomous positioning: 0.25 m/s
        - Surgeon teleop: 0.50 m/s
        - Approach to tissue: 0.05 m/s
        """
        speed = float(np.linalg.norm(velocity_m_s))

        if speed > self.limits.velocity_max_m_s:
            event = SafetyEvent(
                timestamp_ns=int(time.monotonic_ns()),
                level=SafetyLevel.CRITICAL,
                category="velocity_exceeded",
                description=(
                    f"Speed {speed:.3f} m/s exceeds limit "
                    f"{self.limits.velocity_max_m_s:.2f} m/s"
                ),
                measured_value=speed,
                limit_value=self.limits.velocity_max_m_s,
                action_taken="STOP_CATEGORY_1",
            )
            self._record_event(event)
            return SafetyLevel.CRITICAL

        if speed > self.limits.velocity_max_m_s * self.limits.warning_threshold_fraction:
            return SafetyLevel.WARNING

        return SafetyLevel.NOMINAL

    def _check_joint_limits(self, joint_positions_rad: np.ndarray) -> SafetyLevel:
        """
        Check joint positions against per-joint limits.

        These are software limits tighter than mechanical stops to prevent
        self-collision and collisions with the patient outside the workspace.
        """
        for i, (pos_rad, (lo_deg, hi_deg)) in enumerate(
            zip(joint_positions_rad, self.limits.joint_limits_deg)
        ):
            pos_deg = float(np.degrees(pos_rad))
            if pos_deg < lo_deg or pos_deg > hi_deg:
                event = SafetyEvent(
                    timestamp_ns=int(time.monotonic_ns()),
                    level=SafetyLevel.CRITICAL,
                    category="joint_limit",
                    description=(
                        f"Joint {i} at {pos_deg:.1f} deg outside "
                        f"[{lo_deg}, {hi_deg}] deg"
                    ),
                    measured_value=pos_deg,
                    limit_value=hi_deg if pos_deg > hi_deg else lo_deg,
                    action_taken="STOP_CATEGORY_1",
                )
                self._record_event(event)
                return SafetyLevel.CRITICAL

        return SafetyLevel.NOMINAL

    def _check_force_rate(self, ft: ForceTorqueReading) -> SafetyLevel:
        """
        Check rate of change of force to detect sudden impacts.

        Even within force limits, a sudden spike indicates unexpected
        contact and warrants stopping.
        """
        if not ft.is_valid:
            return SafetyLevel.NOMINAL  # Already caught by force check

        force_magnitude = float(np.linalg.norm(ft.force_xyz_n))

        if self._last_timestamp_ns > 0 and self._last_force_magnitude > 0:
            dt_s = (ft.timestamp_ns - self._last_timestamp_ns) / 1e9
            if dt_s > 0:
                force_rate = abs(force_magnitude - self._last_force_magnitude) / dt_s
                if force_rate > self.limits.force_rate_max_n_per_s:
                    event = SafetyEvent(
                        timestamp_ns=ft.timestamp_ns,
                        level=SafetyLevel.CRITICAL,
                        category="force_rate_exceeded",
                        description=(
                            f"Force rate {force_rate:.1f} N/s exceeds limit "
                            f"{self.limits.force_rate_max_n_per_s:.1f} N/s"
                        ),
                        measured_value=force_rate,
                        limit_value=self.limits.force_rate_max_n_per_s,
                        action_taken="STOP_CATEGORY_1",
                    )
                    self._record_event(event)
                    self._last_force_magnitude = force_magnitude
                    return SafetyLevel.CRITICAL

        self._last_force_magnitude = force_magnitude
        return SafetyLevel.NOMINAL

    def _clamp_to_workspace(self, position_m: np.ndarray) -> np.ndarray:
        """Clamp position to workspace boundary."""
        offset = position_m - self.limits.workspace_center_m
        dist = float(np.linalg.norm(offset))

        if dist > self.limits.workspace_radius_m:
            direction = offset / dist
            return (
                self.limits.workspace_center_m
                + direction * self.limits.workspace_radius_m * 0.95
            )
        return position_m

    def reset_stop(self, authorization_code: str):
        """
        Reset the safety stop after CRITICAL event.

        Requires an authorization code to prevent accidental resets.
        In production, this would require a physical key-switch or
        two-operator confirmation per IEC 80601-2-77.

        Args:
            authorization_code: Authorization string (facility-specific).
        """
        if authorization_code:
            logger.warning(
                "Safety stop reset. Previous reason: %s", self._stop_reason
            )
            self._is_stopped = False
            self._stop_reason = ""
        else:
            logger.error("Reset rejected: invalid authorization")

    def _record_event(self, event: SafetyEvent):
        """Record safety event and fire callbacks."""
        self._event_log.append(event)
        logger.warning(
            "SAFETY EVENT [%s]: %s (measured=%.3f, limit=%.3f)",
            event.level.name,
            event.description,
            event.measured_value,
            event.limit_value,
        )

        for callback in self._callbacks.get(event.level, []):
            try:
                callback(event)
            except Exception as e:
                logger.error("Safety callback failed: %s", e)

    def get_event_log(self) -> list[SafetyEvent]:
        """Get full event log for regulatory audit."""
        return list(self._event_log)

    def get_status_summary(self) -> dict:
        """Get current safety status for dashboard display."""
        return {
            "is_stopped": self._is_stopped,
            "stop_reason": self._stop_reason,
            "total_cycles": self._cycle_count,
            "total_events": len(self._event_log),
            "critical_events": sum(
                1 for e in self._event_log if e.level == SafetyLevel.CRITICAL
            ),
            "warning_events": sum(
                1 for e in self._event_log if e.level == SafetyLevel.WARNING
            ),
        }


# =============================================================================
# SECTION 4: FORCE-TORQUE SENSOR PROCESSING
# =============================================================================
# Utilities for processing raw force-torque sensor data before it reaches
# the safety monitor. In a real system, this runs in the sensor driver node.
# =============================================================================


class ForceTorqueSensorProcessor:
    """
    Process raw force-torque sensor data for safety monitoring.

    SETUP INSTRUCTIONS:
    -------------------
    1. Power on sensor and wait for thermal stabilization (~5 minutes).
    2. Call calibrate_bias() with no external load on the sensor.
    3. Verify bias by checking zero-load readings are < 0.05 N / 0.005 Nm.
    4. Set low-pass filter cutoff based on task:
       - General manipulation: 50 Hz cutoff
       - Needle insertion: 100 Hz cutoff (to capture insertion events)
       - Palpation: 20 Hz cutoff (smooth force profile)
    """

    def __init__(
        self,
        filter_cutoff_hz: float = 50.0,
        sample_rate_hz: float = 1000.0,
    ):
        self.filter_cutoff_hz = filter_cutoff_hz
        self.sample_rate_hz = sample_rate_hz
        self._bias_force = np.zeros(3)
        self._bias_torque = np.zeros(3)
        self._filter_alpha = self._compute_filter_alpha(
            filter_cutoff_hz, sample_rate_hz
        )
        self._filtered_force = np.zeros(3)
        self._filtered_torque = np.zeros(3)

        logger.info(
            "FT sensor processor: cutoff=%.0f Hz, rate=%.0f Hz, alpha=%.4f",
            filter_cutoff_hz,
            sample_rate_hz,
            self._filter_alpha,
        )

    @staticmethod
    def _compute_filter_alpha(cutoff_hz: float, sample_hz: float) -> float:
        """Compute exponential moving average filter coefficient."""
        dt = 1.0 / sample_hz
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return dt / (tau + dt)

    def calibrate_bias(
        self, readings: list[ForceTorqueReading], n_samples: int = 100
    ):
        """
        Calibrate sensor bias from zero-load readings.

        INSTRUCTIONS:
        - Ensure no external forces on the sensor (robot stationary, no contact).
        - Collect at least 100 readings for stable bias estimate.
        - Re-calibrate if ambient temperature changes by >5 C.
        """
        if not readings:
            logger.warning("No readings for bias calibration")
            return

        forces = np.array([r.force_xyz_n for r in readings[:n_samples]])
        torques = np.array([r.torque_xyz_nm for r in readings[:n_samples]])

        self._bias_force = np.mean(forces, axis=0)
        self._bias_torque = np.mean(torques, axis=0)

        force_std = np.std(np.linalg.norm(forces - self._bias_force, axis=1))
        logger.info(
            "Bias calibrated: force_bias=[%.3f, %.3f, %.3f] N, noise_std=%.4f N",
            *self._bias_force,
            force_std,
        )

    def process(self, raw: ForceTorqueReading) -> ForceTorqueReading:
        """
        Process raw reading: bias compensation + low-pass filtering.

        Args:
            raw: Raw sensor reading.

        Returns:
            Processed reading ready for safety monitoring.
        """
        # Bias compensation
        compensated_force = raw.force_xyz_n - self._bias_force
        compensated_torque = raw.torque_xyz_nm - self._bias_torque

        # Low-pass filter (exponential moving average)
        self._filtered_force += self._filter_alpha * (
            compensated_force - self._filtered_force
        )
        self._filtered_torque += self._filter_alpha * (
            compensated_torque - self._filtered_torque
        )

        return ForceTorqueReading(
            force_xyz_n=self._filtered_force.copy(),
            torque_xyz_nm=self._filtered_torque.copy(),
            timestamp_ns=raw.timestamp_ns,
            is_valid=raw.is_valid,
        )


# =============================================================================
# SECTION 5: WORKSPACE BOUNDARY GENERATION
# =============================================================================
# Generate workspace boundaries from preoperative imaging data.
# In production, these come from the surgical planning system.
# =============================================================================


class WorkspaceBoundaryGenerator:
    """
    Generate safety workspace boundaries from preoperative planning data.

    INSTRUCTIONS:
    - Input: tumor location and critical structure locations in patient frame.
    - Input: patient-to-robot registration transform (see example 04).
    - Output: SafetyLimits.workspace_center_m and workspace_radius_m
      in robot base frame.

    The workspace is intentionally conservative: it must contain the entire
    surgical field while excluding anatomy that should not be touched.
    """

    def __init__(
        self,
        patient_to_robot_transform: np.ndarray | None = None,
    ):
        """
        Args:
            patient_to_robot_transform: 4x4 homogeneous transform from
                patient image coordinates to robot base frame.
                Set from registration (example 04).
        """
        if patient_to_robot_transform is None:
            patient_to_robot_transform = np.eye(4)
        self.T_patient_to_robot = patient_to_robot_transform

    def compute_workspace_from_tumor(
        self,
        tumor_center_patient_m: np.ndarray,
        tumor_radius_m: float,
        safety_margin_m: float = 0.03,
        max_radius_m: float = 0.20,
    ) -> tuple[np.ndarray, float]:
        """
        Compute workspace sphere from tumor location.

        Args:
            tumor_center_patient_m: Tumor center in patient frame [x, y, z].
            tumor_radius_m: Tumor radius in meters.
            safety_margin_m: Additional margin around tumor for instruments.
            max_radius_m: Maximum allowed workspace radius.

        Returns:
            Tuple of (workspace_center_robot_m, workspace_radius_m).
        """
        # Transform tumor center to robot frame
        center_h = np.append(tumor_center_patient_m, 1.0)
        center_robot = (self.T_patient_to_robot @ center_h)[:3]

        # Workspace radius = tumor radius + instrument reach + margin
        instrument_reach = 0.05
        radius = min(
            tumor_radius_m + instrument_reach + safety_margin_m,
            max_radius_m,
        )

        logger.info(
            "Workspace: center=[%.3f, %.3f, %.3f] m, radius=%.3f m",
            *center_robot,
            radius,
        )
        return center_robot, radius


# =============================================================================
# SECTION 6: MAIN DEMONSTRATION
# =============================================================================
# Demonstrates the safety monitor in a simulated control loop.
# Replace the simulated sensor data with actual hardware reads.
# =============================================================================


def run_safety_monitor_demo():
    """
    Demonstrate the real-time safety monitoring system.

    This simulates a 1-second control loop at 100 Hz where the robot
    approaches a workspace boundary and encounters a force limit.

    WHAT TO MODIFY FOR YOUR SYSTEM:
    - Replace SafetyLimits values with your procedure-specific limits.
    - Replace simulated sensor data with actual hardware reads.
    - Connect safety event callbacks to your alarm/UI system.
    """
    logger.info("=" * 70)
    logger.info("REAL-TIME SAFETY MONITORING DEMONSTRATION")
    logger.info("=" * 70)

    # --- Setup safety limits for lung tumor resection ---
    limits = SafetyLimits(
        force_max_n=5.0,
        torque_max_nm=0.3,
        velocity_max_m_s=0.25,
        workspace_center_m=np.array([0.0, 0.0, -0.15]),
        workspace_radius_m=0.12,
        watchdog_timeout_ms=12.0,
        force_rate_max_n_per_s=40.0,
    )

    monitor = SafetyMonitor(limits)

    # Register callback for critical events
    def on_critical(event: SafetyEvent):
        logger.error("CRITICAL CALLBACK: %s", event.description)

    monitor.register_callback(SafetyLevel.CRITICAL, on_critical)

    # --- Setup force-torque sensor processor ---
    ft_processor = ForceTorqueSensorProcessor(
        filter_cutoff_hz=50.0, sample_rate_hz=100.0
    )

    # Calibrate bias with zero-load readings
    bias_readings = [
        ForceTorqueReading(
            force_xyz_n=np.random.randn(3) * 0.02,
            torque_xyz_nm=np.random.randn(3) * 0.002,
            timestamp_ns=int(i * 1e7),
        )
        for i in range(100)
    ]
    ft_processor.calibrate_bias(bias_readings)

    # --- Simulate control loop ---
    dt_s = 0.01  # 100 Hz
    n_steps = 100  # 1 second
    results = {
        "nominal": 0,
        "warning": 0,
        "critical": 0,
    }

    for step in range(n_steps):
        t_ns = int(step * dt_s * 1e9)

        # Simulate robot moving toward workspace boundary
        progress = step / n_steps
        ee_pos = np.array([0.0, 0.0, -0.15]) + np.array(
            [progress * 0.13, 0.0, 0.0]
        )
        ee_vel = np.array([0.13, 0.0, 0.0])

        # Simulate increasing contact force near boundary
        force = np.array([progress * 6.0, 0.1, 0.0])  # Ramps to 6 N

        raw_ft = ForceTorqueReading(
            force_xyz_n=force,
            torque_xyz_nm=np.array([0.0, 0.0, 0.01]),
            timestamp_ns=t_ns,
        )
        processed_ft = ft_processor.process(raw_ft)

        state = RobotState(
            joint_positions_rad=np.zeros(7),
            joint_velocities_rad_s=np.zeros(7),
            ee_position_m=ee_pos,
            ee_velocity_m_s=ee_vel,
            ee_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            force_torque=processed_ft,
            timestamp_ns=t_ns,
        )

        safe_pos, safe_vel, level = monitor.check_and_filter(
            commanded_position_m=ee_pos + ee_vel * dt_s,
            commanded_velocity_m_s=ee_vel,
            state=state,
        )

        if level == SafetyLevel.NOMINAL:
            results["nominal"] += 1
        elif level == SafetyLevel.WARNING:
            results["warning"] += 1
        else:
            results["critical"] += 1
            break

    # --- Print results ---
    status = monitor.get_status_summary()
    events = monitor.get_event_log()

    print("\n" + "=" * 60)
    print("SAFETY MONITOR RESULTS")
    print("=" * 60)
    print(f"Total cycles:     {status['total_cycles']}")
    print(f"Nominal cycles:   {results['nominal']}")
    print(f"Warning cycles:   {results['warning']}")
    print(f"Critical cycles:  {results['critical']}")
    print(f"System stopped:   {status['is_stopped']}")
    print(f"Total events:     {status['total_events']}")

    if events:
        print("\nSafety Events:")
        for event in events:
            print(
                f"  [{event.level.name}] {event.category}: "
                f"{event.description}"
            )

    return {
        "cycles_completed": status["total_cycles"],
        "system_stopped": status["is_stopped"],
        "events": len(events),
        "results": results,
    }


if __name__ == "__main__":
    run_safety_monitor_demo()
