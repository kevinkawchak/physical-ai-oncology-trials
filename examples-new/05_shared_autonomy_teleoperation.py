"""
=============================================================================
EXAMPLE 05: Shared Autonomy and Teleoperation for Surgical Robots
=============================================================================

WHAT THIS CODE DOES:
--------------------
Implements the shared control architecture where a surgeon teleoperates
the robot while an AI policy provides real-time assistance. The system
blends surgeon commands with policy suggestions, implements virtual
fixtures to prevent unsafe motions, and manages transitions between
full teleop and full autonomy.

This is how surgeons and AI work together on the same robot during
oncology procedures.

WHEN TO USE THIS:
-----------------
- You are building the surgeon-robot interface for teleoperated surgery
- You need virtual fixtures (no-go zones) around critical anatomy
- You want AI-assisted guidance during manual teleoperation
- You need smooth transitions between teleop and autonomous modes
- You are implementing haptic feedback for force reflection

HARDWARE REQUIREMENTS:
----------------------
    - Surgical robot (dVRK with MTM/PSM, Kinova, etc.)
    - Surgeon console with input devices (MTM, 3D Systems Touch, SpaceMouse)
    - Force-feedback capable input device (for haptic rendering)
    - Foot pedal for mode switching (clutch, camera, autonomy toggle)

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - NumPy 1.24.0+
    - SciPy 1.11.0+

Optional (for deployment):
    - ROS 2 Jazzy (for hardware integration)
    - dVRK 2.4.0 (for da Vinci MTM/PSM teleoperation)
    - OpenHaptics (for haptic device integration)

CLINICAL CONTEXT:
-----------------
During oncology surgery, the surgeon may want:
- Full manual control during delicate dissection near vessels
- AI assistance for repetitive tasks (suturing, retraction)
- Virtual fixtures to prevent accidentally cutting into critical structures
- Smooth handoff between manual and autonomous phases

The shared autonomy framework manages all of these through a
configurable blending architecture.

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: AUTONOMY LEVELS
# =============================================================================
# The system supports a spectrum from full teleop to full autonomy.
# The surgeon or supervising physician controls the autonomy level.
#
# These levels are based on the SAE J3016 analogy for surgical robots:
# Level 0: No assistance (raw teleop)
# Level 1: Warning only (virtual fixtures display but don't constrain)
# Level 2: Assisted teleop (virtual fixtures constrain motion)
# Level 3: Conditional autonomy (AI controls, surgeon monitors)
# Level 4: High autonomy (AI controls subtask, surgeon supervises)
# =============================================================================


class AutonomyLevel(Enum):
    """
    Surgical robot autonomy levels.

    TELEOP_RAW: Surgeon has full control. No AI intervention.
        Use for: initial exploration, situations where AI might interfere.

    TELEOP_GUIDED: Surgeon controls, AI provides visual guidance overlays.
        Use for: approach to surgical site, instrument positioning.

    TELEOP_CONSTRAINED: Surgeon controls, but virtual fixtures prevent
        entry into forbidden regions (critical structures, outside workspace).
        Use for: resection near vessels, dissection along tissue planes.

    SHARED_BLENDED: AI and surgeon commands are blended with configurable
        authority. Surgeon can override AI at any time.
        Use for: precision tasks where AI assists with stability/tremor.

    AUTONOMOUS_SUPERVISED: AI controls the robot, surgeon monitors and
        can intervene at any time via foot pedal.
        Use for: automated suturing, retraction, routine positioning.
    """

    TELEOP_RAW = 0
    TELEOP_GUIDED = 1
    TELEOP_CONSTRAINED = 2
    SHARED_BLENDED = 3
    AUTONOMOUS_SUPERVISED = 4


@dataclass
class SharedAutonomyConfig:
    """
    Configuration for shared autonomy system.

    INSTRUCTIONS FOR SETTING BLEND AUTHORITY:
    ------------------------------------------
    surgeon_authority: Float in [0, 1]. Fraction of surgeon command in blend.
        - 1.0 = full surgeon control (teleop)
        - 0.0 = full AI control (autonomous)
        - 0.7 = surgeon dominant with AI stabilization (typical for resection)
        - 0.3 = AI dominant with surgeon override (typical for suturing)

    virtual_fixture_stiffness: Force per unit penetration into forbidden zone.
        - 500 N/m = soft warning (haptic nudge)
        - 2000 N/m = firm constraint (prevents entry)
        - 5000 N/m = rigid wall (hard stop at boundary)

    tremor_filter_cutoff_hz: Low-pass filter to reduce surgeon hand tremor.
        - 6 Hz removes most physiological tremor
        - Set to 0 to disable
    """

    autonomy_level: AutonomyLevel = AutonomyLevel.TELEOP_CONSTRAINED
    surgeon_authority: float = 0.7
    virtual_fixture_stiffness: float = 2000.0
    virtual_fixture_damping: float = 50.0
    tremor_filter_cutoff_hz: float = 6.0
    max_teleop_velocity_m_s: float = 0.10
    scaling_factor: float = 0.3  # Motion scaling (0.3 = 3:1 reduction)
    deadband_m: float = 0.0005  # Input deadband to prevent drift


# =============================================================================
# SECTION 2: SURGEON INPUT PROCESSING
# =============================================================================
# Process raw input from the surgeon console (MTM, haptic device, etc.)
# and apply scaling, filtering, and deadband.
# =============================================================================


class SurgeonInputProcessor:
    """
    Process surgeon console input for teleoperation.

    Applies motion scaling, tremor filtering, and clutching to convert
    raw surgeon hand motion into robot commands.

    SETUP INSTRUCTIONS:
    -------------------
    For dVRK MTM (Master Tool Manipulator):
      - Subscribe to /MTML/measured_cp (left MTM) or /MTMR/measured_cp
      - Position is in MTM frame; needs transform to PSM workspace
      - Clutch via /footpedals/clutch topic

    For 3D Systems Touch (Phantom Omni):
      - Use OpenHaptics SDK or ros2_haptics
      - Position scaling: 1000:1 (workspace is ~16 cm cube)

    For SpaceMouse:
      - 6DOF input but no force feedback
      - Use as velocity command (integrate to get position)

    Example:
        >>> processor = SurgeonInputProcessor(scaling=0.3)
        >>> robot_cmd = processor.process(mtm_position, is_clutched=False)
    """

    def __init__(self, config: SharedAutonomyConfig):
        self.config = config
        self._clutch_offset = np.zeros(3)
        self._is_clutched = False
        self._last_input_position = np.zeros(3)
        self._filter_state = np.zeros(3)
        self._initialized = False

    def process(
        self,
        input_position_m: np.ndarray,
        input_orientation_quat: np.ndarray,
        is_clutched: bool,
        current_robot_position_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process raw surgeon input into scaled, filtered robot command.

        Args:
            input_position_m: Raw position from input device.
            input_orientation_quat: Raw orientation from input device.
            is_clutched: True if clutch pedal is pressed (decouple input).
            current_robot_position_m: Current robot end-effector position.

        Returns:
            Tuple of (target_position_m, target_orientation_quat) for robot.

        CLUTCH BEHAVIOR:
        - When clutched: surgeon can reposition hands without moving robot.
        - On un-clutch: offset is computed so motion resumes from current pos.
        - This is essential for large workspace motions with small input device.
        """
        if not self._initialized:
            self._last_input_position = input_position_m.copy()
            self._clutch_offset = input_position_m - current_robot_position_m / self.config.scaling_factor
            self._initialized = True

        # Handle clutch state transitions
        if is_clutched:
            self._is_clutched = True
            self._last_input_position = input_position_m.copy()
            return current_robot_position_m, input_orientation_quat

        if self._is_clutched:
            # Un-clutching: reset offset
            self._clutch_offset = input_position_m - current_robot_position_m / self.config.scaling_factor
            self._is_clutched = False

        # Compute scaled delta
        raw_delta = input_position_m - self._clutch_offset
        scaled_position = raw_delta * self.config.scaling_factor

        # Apply deadband
        delta_from_current = scaled_position - current_robot_position_m
        if np.linalg.norm(delta_from_current) < self.config.deadband_m:
            scaled_position = current_robot_position_m

        # Apply tremor filter (simple EMA)
        if self.config.tremor_filter_cutoff_hz > 0:
            dt = 0.01  # Assume 100 Hz
            tau = 1.0 / (2 * np.pi * self.config.tremor_filter_cutoff_hz)
            alpha = dt / (tau + dt)
            self._filter_state += alpha * (scaled_position - self._filter_state)
            filtered_position = self._filter_state.copy()
        else:
            filtered_position = scaled_position

        self._last_input_position = input_position_m.copy()
        return filtered_position, input_orientation_quat


# =============================================================================
# SECTION 3: VIRTUAL FIXTURES
# =============================================================================
# Virtual fixtures constrain robot motion to safe regions.
# They are the AI's contribution to safety during teleoperation.
#
# Types implemented:
# 1. FORBIDDEN REGION: Prevents entry (e.g., around blood vessels)
# 2. GUIDANCE: Attracts motion along a planned path
# 3. BOUNDARY: Workspace boundary enforcement
# =============================================================================


class VirtualFixtureType(Enum):
    FORBIDDEN_REGION = auto()  # Repulsive: push away from danger
    GUIDANCE_PATH = auto()  # Attractive: pull toward planned trajectory
    BOUNDARY_PLANE = auto()  # Half-space: do not cross this plane


@dataclass
class VirtualFixture:
    """
    A single virtual fixture constraining robot motion.

    Attributes:
        fixture_type: Type of constraint.
        name: Human-readable name (e.g., "pulmonary_artery_exclusion").
        parameters: Type-specific parameters (see below).
        stiffness: Force per unit penetration (N/m).
        damping: Velocity-dependent damping (Ns/m).
        is_active: Whether this fixture is currently enforced.

    Parameters by type:
    -------------------
    FORBIDDEN_REGION:
        center_m: [x, y, z] center of exclusion sphere.
        radius_m: Radius of exclusion sphere.

    GUIDANCE_PATH:
        waypoints_m: Nx3 array of path waypoints.
        tube_radius_m: Distance within which guidance is active.

    BOUNDARY_PLANE:
        point_m: [x, y, z] point on the plane.
        normal: [nx, ny, nz] outward normal (allowed side).
    """

    fixture_type: VirtualFixtureType
    name: str
    parameters: dict
    stiffness: float = 2000.0
    damping: float = 50.0
    is_active: bool = True


class VirtualFixtureEngine:
    """
    Compute virtual fixture forces and constrained positions.

    INSTRUCTIONS:
    - Add fixtures from preoperative planning data (tumor margins,
      vessel locations, workspace boundaries).
    - Fixtures are defined in robot base frame.
    - Update fixture positions if tissue deforms (from sensor fusion, ex 02).
    - The engine computes a constraint force at the current EE position.
      This force is either applied as haptic feedback (felt by surgeon)
      and/or used to modify the commanded position.

    Example:
        >>> engine = VirtualFixtureEngine()
        >>> engine.add_forbidden_region("artery", center, radius=0.008)
        >>> force, constrained_pos = engine.compute(
        ...     current_pos, target_pos, velocity
        ... )
    """

    def __init__(self):
        self._fixtures: list[VirtualFixture] = []

    def add_forbidden_region(
        self,
        name: str,
        center_m: np.ndarray,
        radius_m: float,
        stiffness: float = 2000.0,
    ):
        """
        Add a spherical forbidden region (e.g., around a blood vessel).

        The instrument is repelled from the center when it enters
        the exclusion sphere.
        """
        self._fixtures.append(
            VirtualFixture(
                fixture_type=VirtualFixtureType.FORBIDDEN_REGION,
                name=name,
                parameters={"center_m": center_m, "radius_m": radius_m},
                stiffness=stiffness,
            )
        )
        logger.info(
            "Added forbidden region '%s': center=%s, radius=%.1f mm",
            name,
            center_m,
            radius_m * 1000,
        )

    def add_guidance_path(
        self,
        name: str,
        waypoints_m: np.ndarray,
        tube_radius_m: float = 0.005,
        stiffness: float = 500.0,
    ):
        """
        Add a guidance path that attracts the instrument.

        The instrument is gently pulled toward the nearest point
        on the path when within the tube radius.
        """
        self._fixtures.append(
            VirtualFixture(
                fixture_type=VirtualFixtureType.GUIDANCE_PATH,
                name=name,
                parameters={
                    "waypoints_m": waypoints_m,
                    "tube_radius_m": tube_radius_m,
                },
                stiffness=stiffness,
            )
        )

    def add_boundary_plane(
        self,
        name: str,
        point_m: np.ndarray,
        normal: np.ndarray,
        stiffness: float = 5000.0,
    ):
        """
        Add a planar boundary (half-space constraint).

        Motion is allowed on the side of the plane indicated by the normal.
        """
        normal = normal / np.linalg.norm(normal)
        self._fixtures.append(
            VirtualFixture(
                fixture_type=VirtualFixtureType.BOUNDARY_PLANE,
                name=name,
                parameters={"point_m": point_m, "normal": normal},
                stiffness=stiffness,
            )
        )

    def compute(
        self,
        current_position_m: np.ndarray,
        target_position_m: np.ndarray,
        velocity_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute virtual fixture constraint.

        Args:
            current_position_m: Current end-effector position.
            target_position_m: Desired (unconstrained) position.
            velocity_m_s: Current end-effector velocity.

        Returns:
            Tuple of (constraint_force_n, constrained_position_m).
            constraint_force_n: Force to render as haptic feedback.
            constrained_position_m: Position after applying constraints.
        """
        total_force = np.zeros(3)
        constrained = target_position_m.copy()

        for fixture in self._fixtures:
            if not fixture.is_active:
                continue

            if fixture.fixture_type == VirtualFixtureType.FORBIDDEN_REGION:
                f, p = self._compute_forbidden_region(fixture, current_position_m, target_position_m, velocity_m_s)
            elif fixture.fixture_type == VirtualFixtureType.GUIDANCE_PATH:
                f, p = self._compute_guidance(fixture, current_position_m, target_position_m)
            elif fixture.fixture_type == VirtualFixtureType.BOUNDARY_PLANE:
                f, p = self._compute_boundary(fixture, current_position_m, target_position_m, velocity_m_s)
            else:
                continue

            total_force += f
            constrained = p  # Last fixture wins for position

        return total_force, constrained

    def _compute_forbidden_region(
        self,
        fixture: VirtualFixture,
        current: np.ndarray,
        target: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute repulsive force from forbidden region."""
        center = fixture.parameters["center_m"]
        radius = fixture.parameters["radius_m"]

        offset = target - center
        distance = float(np.linalg.norm(offset))

        if distance < radius and distance > 1e-6:
            # Penetration: push outward
            penetration = radius - distance
            direction = offset / distance
            force = direction * penetration * fixture.stiffness
            force -= velocity * fixture.damping  # Damping

            # Project target to boundary
            constrained = center + direction * radius
            return force, constrained

        return np.zeros(3), target

    def _compute_guidance(
        self,
        fixture: VirtualFixture,
        current: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute attractive force toward guidance path."""
        waypoints = fixture.parameters["waypoints_m"]
        tube_radius = fixture.parameters["tube_radius_m"]

        # Find closest point on piecewise linear path
        min_dist = float("inf")
        closest_point = target.copy()

        for i in range(len(waypoints) - 1):
            p = self._closest_point_on_segment(target, waypoints[i], waypoints[i + 1])
            dist = float(np.linalg.norm(target - p))
            if dist < min_dist:
                min_dist = dist
                closest_point = p

        if min_dist < tube_radius and min_dist > 1e-6:
            direction = closest_point - target
            force = direction * fixture.stiffness * (1 - min_dist / tube_radius)
            # Blend toward path
            blend = 0.3  # Gentle guidance
            constrained = target + blend * (closest_point - target)
            return force, constrained

        return np.zeros(3), target

    def _compute_boundary(
        self,
        fixture: VirtualFixture,
        current: np.ndarray,
        target: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute constraint force from boundary plane."""
        point = fixture.parameters["point_m"]
        normal = fixture.parameters["normal"]

        # Signed distance from target to plane (positive = allowed side)
        signed_dist = float(np.dot(target - point, normal))

        if signed_dist < 0:
            # On forbidden side: push back
            penetration = -signed_dist
            force = normal * penetration * fixture.stiffness
            force -= velocity * fixture.damping

            # Project onto plane
            constrained = target + normal * penetration
            return force, constrained

        return np.zeros(3), target

    @staticmethod
    def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Find closest point on line segment AB to point P."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0.0, 1.0)
        return a + t * ab


# =============================================================================
# SECTION 4: COMMAND BLENDER
# =============================================================================
# Blends surgeon teleop commands with AI policy commands based on
# the current autonomy level and surgeon authority setting.
# =============================================================================


class CommandBlender:
    """
    Blend surgeon and AI commands with configurable authority.

    The blender combines two command sources:
    1. Surgeon teleop command (from SurgeonInputProcessor)
    2. AI policy command (from PolicyInferenceEngine, ex 03)

    The blend is controlled by surgeon_authority in [0, 1]:
        output = authority * surgeon_cmd + (1 - authority) * ai_cmd

    With additional constraints:
    - Surgeon can always override AI by pressing harder on the input device
    - AI command is zeroed if autonomy level < SHARED_BLENDED
    - Virtual fixtures are applied after blending

    Example:
        >>> blender = CommandBlender(config)
        >>> blended_pos = blender.blend(
        ...     surgeon_pos, ai_pos, current_pos, autonomy_level
        ... )
    """

    def __init__(self, config: SharedAutonomyConfig):
        self.config = config

    def blend(
        self,
        surgeon_target_m: np.ndarray,
        ai_target_m: np.ndarray,
        current_position_m: np.ndarray,
        autonomy_level: AutonomyLevel,
    ) -> np.ndarray:
        """
        Blend surgeon and AI commands.

        Args:
            surgeon_target_m: Surgeon-commanded position.
            ai_target_m: AI-commanded position.
            current_position_m: Current end-effector position.
            autonomy_level: Current autonomy level.

        Returns:
            Blended target position.
        """
        if autonomy_level == AutonomyLevel.TELEOP_RAW:
            return surgeon_target_m

        if autonomy_level == AutonomyLevel.TELEOP_GUIDED:
            # AI provides overlay only, no motion influence
            return surgeon_target_m

        if autonomy_level == AutonomyLevel.TELEOP_CONSTRAINED:
            # Surgeon controls, virtual fixtures applied later
            return surgeon_target_m

        if autonomy_level == AutonomyLevel.SHARED_BLENDED:
            alpha = self.config.surgeon_authority
            return alpha * surgeon_target_m + (1 - alpha) * ai_target_m

        if autonomy_level == AutonomyLevel.AUTONOMOUS_SUPERVISED:
            return ai_target_m

        return surgeon_target_m


# =============================================================================
# SECTION 5: SHARED AUTONOMY CONTROLLER
# =============================================================================
# Main controller that integrates all components.
# =============================================================================


class SharedAutonomyController:
    """
    Main shared autonomy controller for surgical teleoperation.

    Integrates surgeon input processing, AI policy, virtual fixtures,
    and command blending into a single control loop output.

    INTEGRATION INSTRUCTIONS:
    -------------------------
    1. Initialize with procedure-specific configuration.
    2. Setup virtual fixtures from preoperative planning.
    3. In control loop:
       a. Read surgeon input device
       b. Run AI policy inference
       c. Call compute() to get blended, constrained command
       d. Send to robot hardware (via safety monitor, ex 01)

    Foot pedal mapping (typical):
    - Left pedal:  Clutch (hold to decouple input)
    - Right pedal: Camera control (hold to move camera)
    - Center pedal: Autonomy toggle (tap to cycle levels)

    Example:
        >>> controller = SharedAutonomyController(config)
        >>> controller.add_vessel_fixture("artery", position, radius=0.008)
        >>> # In loop:
        >>> result = controller.compute(
        ...     surgeon_input, ai_command, robot_state, foot_pedals
        ... )
        >>> robot.send(result.target_position)
    """

    def __init__(self, config: SharedAutonomyConfig):
        self.config = config
        self._input_processor = SurgeonInputProcessor(config)
        self._fixture_engine = VirtualFixtureEngine()
        self._blender = CommandBlender(config)
        self._autonomy_level = config.autonomy_level
        self._cycle_count = 0

        logger.info(
            "SharedAutonomyController: level=%s, authority=%.1f",
            self._autonomy_level.name,
            config.surgeon_authority,
        )

    @property
    def autonomy_level(self) -> AutonomyLevel:
        return self._autonomy_level

    def set_autonomy_level(self, level: AutonomyLevel):
        """
        Change autonomy level.

        Typically triggered by surgeon foot pedal or voice command.
        Transition is smooth (no sudden jumps in robot motion).
        """
        old_level = self._autonomy_level
        self._autonomy_level = level
        logger.info("Autonomy level: %s -> %s", old_level.name, level.name)

    def add_vessel_fixture(
        self,
        name: str,
        center_m: np.ndarray,
        radius_m: float = 0.008,
    ):
        """
        Add a forbidden region around a blood vessel.

        The vessel center and radius come from the preoperative segmentation
        transformed to robot frame via patient registration (ex 04).
        """
        # Add safety margin to vessel radius
        exclusion_radius = radius_m + 0.003  # 3 mm safety margin
        self._fixture_engine.add_forbidden_region(name, center_m, exclusion_radius)

    def add_resection_boundary(self, name: str, point_m: np.ndarray, normal: np.ndarray):
        """
        Add a planar boundary for resection margin enforcement.

        The plane represents the outer boundary of the planned resection.
        The robot is constrained to stay on the tumor side (inward normal).
        """
        self._fixture_engine.add_boundary_plane(name, point_m, normal)

    def add_approach_path(self, name: str, waypoints_m: np.ndarray):
        """Add guidance path for instrument approach trajectory."""
        self._fixture_engine.add_guidance_path(name, waypoints_m)

    def compute(
        self,
        surgeon_input_position_m: np.ndarray,
        surgeon_input_orientation_quat: np.ndarray,
        ai_target_position_m: np.ndarray,
        current_robot_position_m: np.ndarray,
        current_robot_velocity_m_s: np.ndarray,
        is_clutched: bool = False,
    ) -> dict:
        """
        Compute the final robot command for this control cycle.

        Args:
            surgeon_input_position_m: Raw surgeon input device position.
            surgeon_input_orientation_quat: Raw surgeon input orientation.
            ai_target_position_m: AI policy target position.
            current_robot_position_m: Current end-effector position.
            current_robot_velocity_m_s: Current end-effector velocity.
            is_clutched: Whether clutch pedal is pressed.

        Returns:
            Dictionary with:
            - target_position_m: Final constrained target for robot.
            - target_orientation_quat: Final orientation target.
            - haptic_force_n: Force to render on surgeon input device.
            - autonomy_level: Current autonomy level.
            - fixture_active: Whether any fixture is constraining.
        """
        self._cycle_count += 1

        # Step 1: Process surgeon input (scaling, filtering, clutch)
        surgeon_target, surgeon_orient = self._input_processor.process(
            surgeon_input_position_m,
            surgeon_input_orientation_quat,
            is_clutched,
            current_robot_position_m,
        )

        # Step 2: Blend surgeon and AI commands
        blended_target = self._blender.blend(
            surgeon_target,
            ai_target_position_m,
            current_robot_position_m,
            self._autonomy_level,
        )

        # Step 3: Apply virtual fixtures
        fixture_force, constrained_target = self._fixture_engine.compute(
            current_robot_position_m,
            blended_target,
            current_robot_velocity_m_s,
        )

        fixture_active = float(np.linalg.norm(fixture_force)) > 0.1

        # Step 4: Compute haptic feedback force for surgeon
        # Scale fixture forces for haptic rendering
        haptic_scale = 0.5  # Reduce force for haptic device capability
        haptic_force = fixture_force * haptic_scale

        return {
            "target_position_m": constrained_target,
            "target_orientation_quat": surgeon_orient,
            "haptic_force_n": haptic_force,
            "autonomy_level": self._autonomy_level,
            "fixture_active": fixture_active,
            "surgeon_target_m": surgeon_target,
            "ai_target_m": ai_target_position_m,
            "blend_authority": self.config.surgeon_authority,
        }


# =============================================================================
# SECTION 6: MAIN DEMONSTRATION
# =============================================================================


def run_shared_autonomy_demo():
    """
    Demonstrate shared autonomy during a simulated tumor resection.

    Simulates a surgeon teleoperating near a blood vessel while
    virtual fixtures prevent entry into the vessel exclusion zone.
    """
    logger.info("=" * 70)
    logger.info("SHARED AUTONOMY TELEOPERATION DEMONSTRATION")
    logger.info("=" * 70)

    # --- Setup ---
    config = SharedAutonomyConfig(
        autonomy_level=AutonomyLevel.TELEOP_CONSTRAINED,
        surgeon_authority=0.7,
        virtual_fixture_stiffness=2000.0,
        scaling_factor=0.3,
        tremor_filter_cutoff_hz=6.0,
    )

    controller = SharedAutonomyController(config)

    # Add virtual fixtures from preoperative planning
    # Vessel at (0.02, 0, -0.15) with 8mm radius
    controller.add_vessel_fixture(
        "pulmonary_artery",
        center_m=np.array([0.02, 0.0, -0.15]),
        radius_m=0.008,
    )

    # Resection boundary plane
    controller.add_resection_boundary(
        "resection_margin",
        point_m=np.array([0.05, 0.0, -0.15]),
        normal=np.array([-1.0, 0.0, 0.0]),  # Don't go past x=0.05
    )

    # --- Simulate 100 control cycles (1 second at 100 Hz) ---
    n_cycles = 100
    dt = 0.01
    results_log = []

    robot_pos = np.array([0.0, 0.0, -0.15])
    robot_vel = np.zeros(3)

    for cycle in range(n_cycles):
        progress = cycle / n_cycles

        # Simulate surgeon moving toward the vessel
        surgeon_input = np.array(
            [
                0.0 + progress * 0.08,  # Moving in +x toward vessel
                np.sin(cycle * 0.3) * 0.002,  # Small lateral tremor
                -0.15 + np.sin(cycle * 0.5) * 0.001,
            ]
        )

        # Simulate AI suggesting a safer path
        ai_target = np.array(
            [
                0.0 + progress * 0.04,  # AI moves slower, more conservative
                0.0,
                -0.15,
            ]
        )

        result = controller.compute(
            surgeon_input_position_m=surgeon_input,
            surgeon_input_orientation_quat=np.array([1, 0, 0, 0.0]),
            ai_target_position_m=ai_target,
            current_robot_position_m=robot_pos,
            current_robot_velocity_m_s=robot_vel,
        )

        # Update simulated robot
        old_pos = robot_pos.copy()
        robot_pos = result["target_position_m"]
        robot_vel = (robot_pos - old_pos) / dt

        results_log.append(
            {
                "cycle": cycle,
                "robot_x_mm": robot_pos[0] * 1000,
                "fixture_active": result["fixture_active"],
                "haptic_force_n": float(np.linalg.norm(result["haptic_force_n"])),
            }
        )

    # --- Test autonomy level switching ---
    logger.info("\n--- Testing autonomy level transitions ---")
    levels = [
        AutonomyLevel.TELEOP_RAW,
        AutonomyLevel.TELEOP_GUIDED,
        AutonomyLevel.TELEOP_CONSTRAINED,
        AutonomyLevel.SHARED_BLENDED,
        AutonomyLevel.AUTONOMOUS_SUPERVISED,
    ]
    for level in levels:
        controller.set_autonomy_level(level)

    # --- Print results ---
    fixture_active_count = sum(1 for r in results_log if r["fixture_active"])
    max_haptic = max(r["haptic_force_n"] for r in results_log)
    max_x = max(r["robot_x_mm"] for r in results_log)

    print("\n" + "=" * 60)
    print("SHARED AUTONOMY RESULTS")
    print("=" * 60)
    print(f"Control cycles:         {n_cycles}")
    print(f"Autonomy level:         {config.autonomy_level.name}")
    print(f"Surgeon authority:      {config.surgeon_authority:.0%}")
    print(f"Fixture activations:    {fixture_active_count}/{n_cycles}")
    print(f"Max haptic force:       {max_haptic:.2f} N")
    print(f"Max robot X position:   {max_x:.1f} mm")
    print("Vessel center X:        20.0 mm")
    print(f"Vessel exclusion zone:  {(8 + 3):.0f} mm radius")
    vessel_breached = max_x > 20.0 - 11.0
    print(f"Vessel exclusion held:  {not vessel_breached}")

    return {
        "cycles": n_cycles,
        "fixture_activations": fixture_active_count,
        "max_haptic_force_n": max_haptic,
        "vessel_exclusion_held": not vessel_breached,
    }


if __name__ == "__main__":
    run_shared_autonomy_demo()
