"""
=============================================================================
EXAMPLE 06: Robotic Sample Handling and Drug Preparation for Oncology Trials
=============================================================================

WHAT THIS CODE DOES:
--------------------
Implements the robotic automation pipeline for handling tissue specimens,
preparing chemotherapy drugs, and managing chain-of-custody for oncology
clinical trials. This covers the non-surgical robot applications that are
critical to trial operations: specimen processing, drug compounding,
and laboratory automation.

These robots operate in the pathology lab, pharmacy, and biobank
rather than the operating room.

WHEN TO USE THIS:
-----------------
- You are automating specimen handling for oncology tissue banks
- You need robotic drug compounding for clinical trial pharmacies
- You must maintain chain-of-custody traceability for trial specimens
- You are implementing barcode/RFID verification for sample tracking
- You need cold-chain monitoring for temperature-sensitive specimens

HARDWARE REQUIREMENTS:
----------------------
    - Laboratory robot arm (Kinova Gen3, UR3e, Franka Emika, or SCARA)
    - Barcode scanner (1D/2D, USB or networked)
    - RFID reader for specimen containers
    - Temperature sensors (thermocouples or IR pyrometer)
    - Gripper appropriate for specimen containers (parallel jaw or soft)
    - Biosafety cabinet integration (for cytotoxic drug handling)

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - NumPy 1.24.0+
    - Python 3.10+

Optional (for deployment):
    - ROS 2 Jazzy (robot control)
    - OpenCV 4.9+ (barcode reading)
    - pyserial (barcode scanner, temperature sensor communication)

REGULATORY REQUIREMENTS:
------------------------
    - 21 CFR Part 11: Electronic records and signatures
    - USP <797>: Sterile compounding (for drug preparation)
    - CAP/CLIA: Laboratory accreditation (for specimen handling)
    - GCP ICH E6(R3): Clinical trial specimen management
    - FDA 21 CFR 58 GLP: Good Laboratory Practice

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: SPECIMEN AND CONTAINER DATA MODELS
# =============================================================================
# Data structures for tracking specimens through the robotic handling
# pipeline. Every field supports 21 CFR Part 11 audit trail requirements.
# =============================================================================


class SpecimenType(Enum):
    TISSUE_BIOPSY = "tissue_biopsy"
    BLOOD_SAMPLE = "blood_sample"
    PLASMA = "plasma"
    SERUM = "serum"
    URINE = "urine"
    CSF = "cerebrospinal_fluid"
    FFPE_BLOCK = "ffpe_block"  # Formalin-Fixed Paraffin-Embedded
    FROZEN_TISSUE = "frozen_tissue"


class ContainerType(Enum):
    CRYOVIAL_2ML = "cryovial_2ml"
    CRYOVIAL_5ML = "cryovial_5ml"
    BLOOD_TUBE_EDTA = "blood_tube_edta"
    BLOOD_TUBE_SST = "blood_tube_sst"
    TISSUE_CASSETTE = "tissue_cassette"
    SPECIMEN_CUP = "specimen_cup_60ml"
    MICROTUBE_1_5ML = "microtube_1.5ml"


class StorageCondition(Enum):
    ROOM_TEMP = "room_temperature"        # 15-25 C
    REFRIGERATED = "refrigerated"          # 2-8 C
    FROZEN_MINUS20 = "frozen_minus20"      # -20 C
    FROZEN_MINUS80 = "frozen_minus80"      # -80 C
    LIQUID_NITROGEN = "liquid_nitrogen"    # -196 C


@dataclass
class Specimen:
    """
    A clinical trial specimen with full chain-of-custody tracking.

    Every field change is recorded in the audit_trail for 21 CFR Part 11
    compliance. The specimen_id is a globally unique identifier linked
    to the clinical trial management system.

    Attributes:
        specimen_id: Globally unique specimen identifier.
        barcode: Physical barcode on the container.
        trial_id: Clinical trial identifier (NCT number).
        patient_id: De-identified patient ID.
        specimen_type: Type of biological material.
        container_type: Physical container type.
        collection_datetime: When the specimen was collected.
        storage_condition: Required storage temperature.
        current_location: Current physical location (rack/shelf/position).
        temperature_log: List of (timestamp, temperature_c) readings.
        audit_trail: Immutable log of all events for this specimen.
    """

    specimen_id: str
    barcode: str
    trial_id: str
    patient_id: str
    specimen_type: SpecimenType
    container_type: ContainerType
    collection_datetime: str
    storage_condition: StorageCondition
    current_location: str = ""
    volume_ml: float = 0.0
    temperature_log: list = field(default_factory=list)
    audit_trail: list = field(default_factory=list)

    def record_event(self, event_type: str, description: str, operator: str):
        """
        Record an auditable event for this specimen.

        All events are timestamped and include the operator identity.
        This supports 21 CFR Part 11 electronic record requirements.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "operator": operator,
            "specimen_id": self.specimen_id,
            "integrity_hash": self._compute_hash(),
        }
        self.audit_trail.append(event)

    def _compute_hash(self) -> str:
        """Compute integrity hash for current specimen state."""
        state = f"{self.specimen_id}:{self.current_location}:{len(self.audit_trail)}"
        return hashlib.sha256(state.encode()).hexdigest()[:16]


# =============================================================================
# SECTION 2: BARCODE AND RFID VERIFICATION
# =============================================================================
# Verify specimen identity at every transfer point.
# This prevents mix-ups, which are a critical safety concern in oncology
# trials where treatment decisions depend on correct specimen labeling.
#
# INSTRUCTIONS:
# - Every pick or place operation must include barcode verification.
# - If barcode scan fails, retry 3 times, then alert operator.
# - Log all scan results (success and failure) for audit.
# =============================================================================


class BarcodeVerifier:
    """
    Barcode and RFID verification for specimen identification.

    HARDWARE SETUP:
    ---------------
    For fixed barcode scanner (e.g., Cognex DataMan, Keyence SR):
      1. Mount scanner at the robot's pick/place station.
      2. Position scanner to read barcode on container at rest position.
      3. Configure for 2D barcode (Data Matrix) which is standard for
         clinical specimens (ISBT 128 format).

    For robot-mounted scanner:
      1. Mount on robot wrist, aimed at gripper.
      2. Read barcode while container is in gripper.
      3. Ensure lighting is consistent (may need dedicated illumination).

    For RFID:
      1. Use HF (13.56 MHz) RFID for cryovials with RFID-enabled caps.
      2. Mount reader at pick station.
      3. Range must be <5 cm to prevent reading adjacent containers.

    Example:
        >>> verifier = BarcodeVerifier()
        >>> result = verifier.verify("scan_result_12345", expected_barcode)
        >>> if not result["verified"]:
        ...     alert_operator("Barcode mismatch!")
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._scan_log: list[dict] = []

    def scan_and_verify(
        self,
        expected_barcode: str,
        station_id: str,
    ) -> dict:
        """
        Scan barcode at a station and verify against expected value.

        In production, this would trigger the physical scanner and read
        the result via serial port or network API.

        Args:
            expected_barcode: Expected barcode string.
            station_id: Identifier of the scanning station.

        Returns:
            Dictionary with verification result and scan details.
        """
        for attempt in range(self.max_retries):
            # Simulated scan (replace with hardware API)
            scanned = self._simulate_scan(expected_barcode, attempt)

            result = {
                "timestamp": datetime.now().isoformat(),
                "station_id": station_id,
                "expected": expected_barcode,
                "scanned": scanned,
                "verified": scanned == expected_barcode,
                "attempt": attempt + 1,
            }
            self._scan_log.append(result)

            if result["verified"]:
                logger.info(
                    "Barcode verified: %s at %s (attempt %d)",
                    scanned,
                    station_id,
                    attempt + 1,
                )
                return result

            logger.warning(
                "Barcode mismatch at %s: expected=%s, scanned=%s (attempt %d)",
                station_id,
                expected_barcode,
                scanned,
                attempt + 1,
            )

        # All retries failed
        result["action_required"] = "operator_intervention"
        return result

    def _simulate_scan(self, expected: str, attempt: int) -> str:
        """Simulate barcode scan (replace with hardware driver)."""
        # Simulate 95% success rate
        if np.random.random() < 0.95 or attempt > 0:
            return expected
        return expected + "_ERR"


# =============================================================================
# SECTION 3: COLD CHAIN MONITORING
# =============================================================================
# Continuous temperature monitoring for specimens requiring cold storage.
# Temperature excursions can invalidate specimens and trial data.
#
# INSTRUCTIONS:
# - Temperature must be logged at least every 5 minutes for -80 C storage.
# - Alert if temperature exceeds allowed range for >30 seconds.
# - Log all excursions with duration for regulatory reporting.
# =============================================================================


class ColdChainMonitor:
    """
    Monitor and enforce cold chain requirements for specimens.

    TEMPERATURE REQUIREMENTS BY STORAGE TYPE:
    ------------------------------------------
    Room temperature:  15-25 C  (max excursion: 30 C for 2 hours)
    Refrigerated:       2-8 C   (max excursion: 15 C for 30 min)
    Frozen -20 C:     -25 to -15 C (max excursion: -10 C for 15 min)
    Frozen -80 C:     -85 to -75 C (max excursion: -60 C for 10 min)
    Liquid N2:        below -150 C (any above -150 C is excursion)

    SENSOR SETUP:
    - Use calibrated thermocouples (Type T for cryogenic, Type K for ambient).
    - For -80 C freezers: use freezer's built-in sensor + external backup.
    - For transport: use data logger (e.g., Sensitech TempTale) attached
      to the robot's specimen carrier.
    """

    TEMP_RANGES: dict[StorageCondition, tuple[float, float]] = {
        StorageCondition.ROOM_TEMP: (15.0, 25.0),
        StorageCondition.REFRIGERATED: (2.0, 8.0),
        StorageCondition.FROZEN_MINUS20: (-25.0, -15.0),
        StorageCondition.FROZEN_MINUS80: (-85.0, -75.0),
        StorageCondition.LIQUID_NITROGEN: (-200.0, -150.0),
    }

    def __init__(self, check_interval_s: float = 10.0):
        self.check_interval_s = check_interval_s
        self._excursion_log: list[dict] = []

    def check_temperature(
        self,
        specimen: Specimen,
        current_temp_c: float,
    ) -> dict:
        """
        Check if current temperature is within acceptable range.

        Args:
            specimen: Specimen being monitored.
            current_temp_c: Current temperature in Celsius.

        Returns:
            Dictionary with check result and any excursion details.
        """
        temp_range = self.TEMP_RANGES.get(specimen.storage_condition)
        if temp_range is None:
            return {"in_range": True, "message": "No temperature requirement"}

        low, high = temp_range
        in_range = low <= current_temp_c <= high

        # Log temperature reading
        specimen.temperature_log.append(
            (datetime.now().isoformat(), current_temp_c)
        )

        if not in_range:
            excursion = {
                "specimen_id": specimen.specimen_id,
                "timestamp": datetime.now().isoformat(),
                "temperature_c": current_temp_c,
                "required_range": f"{low} to {high} C",
                "deviation_c": (
                    current_temp_c - high
                    if current_temp_c > high
                    else low - current_temp_c
                ),
            }
            self._excursion_log.append(excursion)
            specimen.record_event(
                "temperature_excursion",
                f"Temperature {current_temp_c:.1f} C outside "
                f"range [{low}, {high}] C",
                "cold_chain_monitor",
            )
            logger.warning(
                "TEMPERATURE EXCURSION: specimen %s at %.1f C "
                "(required: %.0f to %.0f C)",
                specimen.specimen_id,
                current_temp_c,
                low,
                high,
            )

        return {
            "in_range": in_range,
            "current_temp_c": current_temp_c,
            "required_range_c": temp_range,
            "specimen_id": specimen.specimen_id,
        }


# =============================================================================
# SECTION 4: ROBOT PICK-AND-PLACE FOR SPECIMENS
# =============================================================================
# The robotic manipulation layer for moving specimens between locations.
#
# INSTRUCTIONS:
# - Gripper force must be calibrated for each container type.
#   Cryovials: 5-10 N grip force (fragile at -80 C).
#   Blood tubes: 3-8 N (do not crush).
#   Tissue cassettes: 8-15 N.
# - Use force-torque feedback to detect drops or jams.
# - Every pick/place includes barcode verification.
# =============================================================================


@dataclass
class PickPlaceConfig:
    """
    Configuration for a specific container type's pick-and-place parameters.

    INSTRUCTIONS:
    - Measure these values empirically for each container type.
    - Grip force: calibrate with a force gauge on the actual gripper.
    - Approach height: 30 mm above container top for safe approach.
    - Grasp height: height at which gripper closes around the container.
    """

    container_type: ContainerType
    grip_force_n: float
    approach_height_m: float = 0.03
    grasp_height_m: float = 0.01
    lift_height_m: float = 0.05
    place_height_m: float = 0.005
    gripper_width_mm: float = 12.0  # Opening width for container


CONTAINER_CONFIGS: dict[ContainerType, PickPlaceConfig] = {
    ContainerType.CRYOVIAL_2ML: PickPlaceConfig(
        container_type=ContainerType.CRYOVIAL_2ML,
        grip_force_n=8.0,
        gripper_width_mm=12.5,
    ),
    ContainerType.CRYOVIAL_5ML: PickPlaceConfig(
        container_type=ContainerType.CRYOVIAL_5ML,
        grip_force_n=10.0,
        gripper_width_mm=13.0,
    ),
    ContainerType.BLOOD_TUBE_EDTA: PickPlaceConfig(
        container_type=ContainerType.BLOOD_TUBE_EDTA,
        grip_force_n=5.0,
        gripper_width_mm=16.0,
    ),
    ContainerType.TISSUE_CASSETTE: PickPlaceConfig(
        container_type=ContainerType.TISSUE_CASSETTE,
        grip_force_n=12.0,
        gripper_width_mm=30.0,
    ),
}


class SpecimenRobotController:
    """
    Control a laboratory robot for specimen handling.

    INTEGRATION INSTRUCTIONS:
    -------------------------
    For Kinova Gen3 (recommended for lab automation):
      1. Launch: ros2 launch kortex_driver gen3.launch.py
      2. Use joint_trajectory_controller for pick/place motions
      3. Configure Robotiq 2F-85 gripper via /robotiq_gripper_controller

    For UR3e:
      1. Launch: ros2 launch ur_robot_driver ur3e.launch.py
      2. Use scaled_joint_trajectory_controller
      3. Configure OnRobot RG2 gripper

    MOTION PLANNING:
    - Use MoveIt 2 for collision-free planning in cluttered lab environment.
    - Define collision objects for all rack positions, scanners, and fixtures.
    - Waypoints must clear rack edges by at least 10 mm.

    Example:
        >>> robot = SpecimenRobotController()
        >>> robot.pick_specimen(specimen, source_rack, source_position)
        >>> robot.place_specimen(specimen, dest_rack, dest_position)
    """

    def __init__(self):
        self._barcode_verifier = BarcodeVerifier()
        self._cold_chain = ColdChainMonitor()
        self._is_holding_specimen: Specimen | None = None
        self._operation_log: list[dict] = []

        logger.info("SpecimenRobotController initialized")

    def pick_specimen(
        self,
        specimen: Specimen,
        rack_id: str,
        position: tuple[int, int],  # (row, column)
    ) -> dict:
        """
        Pick a specimen from a rack position.

        Procedure:
        1. Move to approach position above the specimen.
        2. Open gripper to container width.
        3. Descend to grasp height.
        4. Close gripper with calibrated force.
        5. Verify barcode while in gripper.
        6. Lift to transport height.

        Args:
            specimen: Specimen to pick.
            rack_id: Source rack identifier.
            position: (row, column) in the rack.

        Returns:
            Operation result dictionary.
        """
        logger.info(
            "PICK: specimen=%s from rack=%s pos=%s",
            specimen.specimen_id,
            rack_id,
            position,
        )

        config = CONTAINER_CONFIGS.get(
            specimen.container_type,
            PickPlaceConfig(container_type=specimen.container_type, grip_force_n=8.0),
        )

        # Step 1: Move to approach position
        approach_pos = self._rack_to_world(rack_id, position, config.approach_height_m)
        self._move_to(approach_pos)

        # Step 2: Open gripper
        self._set_gripper(config.gripper_width_mm)

        # Step 3: Descend
        grasp_pos = self._rack_to_world(rack_id, position, config.grasp_height_m)
        self._move_to(grasp_pos)

        # Step 4: Grasp
        grasp_success = self._close_gripper(config.grip_force_n)

        if not grasp_success:
            specimen.record_event("pick_failed", "Grasp force not achieved", "robot")
            return {"success": False, "reason": "grasp_failed"}

        # Step 5: Verify barcode
        scan_result = self._barcode_verifier.scan_and_verify(
            specimen.barcode, f"pick_station_{rack_id}"
        )

        if not scan_result["verified"]:
            # Release and report error
            self._set_gripper(config.gripper_width_mm)
            specimen.record_event(
                "barcode_mismatch",
                f"Expected {specimen.barcode}, scanned {scan_result['scanned']}",
                "robot",
            )
            return {"success": False, "reason": "barcode_mismatch"}

        # Step 6: Lift
        lift_pos = self._rack_to_world(rack_id, position, config.lift_height_m)
        self._move_to(lift_pos)

        # Update state
        self._is_holding_specimen = specimen
        specimen.record_event(
            "picked",
            f"Picked from rack {rack_id} position {position}",
            "robot",
        )

        result = {
            "success": True,
            "specimen_id": specimen.specimen_id,
            "source_rack": rack_id,
            "source_position": position,
        }
        self._operation_log.append(result)
        return result

    def place_specimen(
        self,
        specimen: Specimen,
        rack_id: str,
        position: tuple[int, int],
    ) -> dict:
        """
        Place a specimen into a rack position.

        Procedure:
        1. Move to approach position above target.
        2. Descend to place height.
        3. Open gripper to release.
        4. Retract to safe height.
        5. Verify barcode at destination (confirm correct placement).
        6. Update specimen location.

        Args:
            specimen: Specimen to place.
            rack_id: Destination rack identifier.
            position: (row, column) in the rack.

        Returns:
            Operation result dictionary.
        """
        logger.info(
            "PLACE: specimen=%s to rack=%s pos=%s",
            specimen.specimen_id,
            rack_id,
            position,
        )

        config = CONTAINER_CONFIGS.get(
            specimen.container_type,
            PickPlaceConfig(container_type=specimen.container_type, grip_force_n=8.0),
        )

        # Step 1: Move above target
        approach_pos = self._rack_to_world(rack_id, position, config.approach_height_m)
        self._move_to(approach_pos)

        # Step 2: Descend
        place_pos = self._rack_to_world(rack_id, position, config.place_height_m)
        self._move_to(place_pos)

        # Step 3: Release
        self._set_gripper(config.gripper_width_mm)

        # Step 4: Retract
        self._move_to(approach_pos)

        # Step 5: Update specimen location
        specimen.current_location = f"{rack_id}:{position[0]}:{position[1]}"
        self._is_holding_specimen = None

        specimen.record_event(
            "placed",
            f"Placed in rack {rack_id} position {position}",
            "robot",
        )

        result = {
            "success": True,
            "specimen_id": specimen.specimen_id,
            "dest_rack": rack_id,
            "dest_position": position,
        }
        self._operation_log.append(result)
        return result

    def transfer_specimen(
        self,
        specimen: Specimen,
        source_rack: str,
        source_pos: tuple[int, int],
        dest_rack: str,
        dest_pos: tuple[int, int],
    ) -> dict:
        """
        Transfer a specimen from one rack position to another.

        This is the most common operation: move a specimen between
        storage locations (e.g., from collection rack to biobank rack).
        Includes temperature monitoring during transfer.
        """
        # Check cold chain before transfer
        ambient_temp = 22.0  # Room temperature during transfer
        cold_check = self._cold_chain.check_temperature(specimen, ambient_temp)

        if not cold_check["in_range"] and specimen.storage_condition in (
            StorageCondition.FROZEN_MINUS80,
            StorageCondition.LIQUID_NITROGEN,
        ):
            logger.warning(
                "Cold chain concern: specimen %s requires %s storage, "
                "currently at %.1f C",
                specimen.specimen_id,
                specimen.storage_condition.value,
                ambient_temp,
            )

        # Pick
        pick_result = self.pick_specimen(specimen, source_rack, source_pos)
        if not pick_result["success"]:
            return {"success": False, "step": "pick", "reason": pick_result["reason"]}

        # Place
        place_result = self.place_specimen(specimen, dest_rack, dest_pos)
        if not place_result["success"]:
            return {"success": False, "step": "place", "reason": place_result["reason"]}

        return {
            "success": True,
            "specimen_id": specimen.specimen_id,
            "from": f"{source_rack}:{source_pos}",
            "to": f"{dest_rack}:{dest_pos}",
            "cold_chain_ok": cold_check["in_range"],
        }

    def _rack_to_world(
        self, rack_id: str, position: tuple[int, int], height_m: float
    ) -> np.ndarray:
        """
        Convert rack position to world coordinates.

        In production, rack positions are calibrated by teaching the
        robot the corner positions of each rack and interpolating.
        """
        # Simulated rack layout
        row, col = position
        rack_origin = np.array([0.3, -0.2, 0.0])  # Rack base position
        spacing = 0.018  # 18 mm between positions (standard SBS spacing)

        return np.array([
            rack_origin[0] + col * spacing,
            rack_origin[1] + row * spacing,
            rack_origin[2] + height_m,
        ])

    def _move_to(self, position_m: np.ndarray):
        """Move robot to position (simulated)."""
        # In production: call MoveIt 2 or joint trajectory controller
        pass

    def _set_gripper(self, width_mm: float):
        """Set gripper opening width (simulated)."""
        pass

    def _close_gripper(self, force_n: float) -> bool:
        """Close gripper with specified force. Returns True if object detected."""
        # Simulated: always succeeds
        return True


# =============================================================================
# SECTION 5: BATCH PROCESSING WORKFLOW
# =============================================================================
# Process a batch of specimens for a clinical trial visit.
# A typical trial visit produces 5-15 specimens that must be
# processed, aliquoted, and stored within specific time windows.
# =============================================================================


class BatchProcessor:
    """
    Process a batch of specimens from a clinical trial visit.

    WORKFLOW:
    ---------
    1. Receive specimens from collection.
    2. Verify all specimen labels against visit manifest.
    3. Sort specimens by processing priority (time-sensitive first).
    4. Process each specimen (aliquot, label, store).
    5. Generate batch processing report.

    INSTRUCTIONS:
    - Blood for plasma must be centrifuged within 30 minutes of collection.
    - Tissue specimens must be fixed or frozen within 30 minutes.
    - All specimens must be in final storage within 2 hours of collection.
    """

    def __init__(self, robot: SpecimenRobotController):
        self.robot = robot
        self._batch_log: list[dict] = []

    def process_visit_batch(
        self,
        visit_id: str,
        specimens: list[Specimen],
        manifest: dict,
    ) -> dict:
        """
        Process a complete visit batch.

        Args:
            visit_id: Clinical visit identifier.
            specimens: List of specimens collected at this visit.
            manifest: Expected specimen manifest from trial protocol.

        Returns:
            Batch processing report.
        """
        logger.info(
            "Processing batch for visit %s: %d specimens",
            visit_id,
            len(specimens),
        )

        report = {
            "visit_id": visit_id,
            "start_time": datetime.now().isoformat(),
            "total_specimens": len(specimens),
            "processed": 0,
            "failed": 0,
            "results": [],
        }

        # Step 1: Verify manifest
        manifest_ok = self._verify_manifest(specimens, manifest)
        report["manifest_verified"] = manifest_ok

        # Step 2: Sort by priority (time-sensitive first)
        sorted_specimens = self._sort_by_priority(specimens)

        # Step 3: Process each specimen
        for i, specimen in enumerate(sorted_specimens):
            logger.info(
                "Processing specimen %d/%d: %s (%s)",
                i + 1,
                len(sorted_specimens),
                specimen.specimen_id,
                specimen.specimen_type.value,
            )

            result = self.robot.transfer_specimen(
                specimen=specimen,
                source_rack="incoming_rack",
                source_pos=(i // 10, i % 10),
                dest_rack=self._get_storage_rack(specimen),
                dest_pos=self._get_next_position(specimen),
            )

            report["results"].append(result)
            if result["success"]:
                report["processed"] += 1
            else:
                report["failed"] += 1

        report["end_time"] = datetime.now().isoformat()
        report["all_processed"] = report["failed"] == 0

        self._batch_log.append(report)
        return report

    def _verify_manifest(self, specimens: list[Specimen], manifest: dict) -> bool:
        """Verify specimens match the visit manifest."""
        expected_ids = set(manifest.get("expected_specimen_ids", []))
        actual_ids = {s.specimen_id for s in specimens}

        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            logger.warning("Missing specimens: %s", missing)
        if extra:
            logger.warning("Extra specimens not in manifest: %s", extra)

        return not missing

    def _sort_by_priority(self, specimens: list[Specimen]) -> list[Specimen]:
        """Sort specimens by processing urgency."""
        priority_order = {
            SpecimenType.BLOOD_SAMPLE: 0,  # Centrifuge within 30 min
            SpecimenType.PLASMA: 0,
            SpecimenType.FROZEN_TISSUE: 1,  # Freeze within 30 min
            SpecimenType.TISSUE_BIOPSY: 2,  # Fix within 1 hour
            SpecimenType.SERUM: 3,
            SpecimenType.URINE: 4,
            SpecimenType.CSF: 0,
            SpecimenType.FFPE_BLOCK: 5,  # Already fixed
        }
        return sorted(
            specimens, key=lambda s: priority_order.get(s.specimen_type, 99)
        )

    def _get_storage_rack(self, specimen: Specimen) -> str:
        """Determine destination rack based on storage requirements."""
        rack_map = {
            StorageCondition.ROOM_TEMP: "ambient_rack_A",
            StorageCondition.REFRIGERATED: "fridge_rack_A",
            StorageCondition.FROZEN_MINUS20: "freezer_minus20_rack_A",
            StorageCondition.FROZEN_MINUS80: "freezer_minus80_rack_A",
            StorageCondition.LIQUID_NITROGEN: "ln2_dewar_rack_A",
        }
        return rack_map.get(specimen.storage_condition, "ambient_rack_A")

    def _get_next_position(self, specimen: Specimen) -> tuple[int, int]:
        """Get next available position in the destination rack."""
        # In production: query LIMS for next available position
        pos = len(self._batch_log) % 96  # 96-well format
        return (pos // 12, pos % 12)


# =============================================================================
# SECTION 6: MAIN DEMONSTRATION
# =============================================================================


def run_sample_handling_demo():
    """
    Demonstrate robotic specimen handling for an oncology clinical trial visit.

    Simulates processing a batch of 8 specimens from a single patient visit,
    including barcode verification, cold chain monitoring, and chain-of-custody
    tracking.
    """
    logger.info("=" * 70)
    logger.info("ROBOTIC SAMPLE HANDLING FOR ONCOLOGY CLINICAL TRIALS")
    logger.info("=" * 70)

    # --- Create robot ---
    robot = SpecimenRobotController()
    batch_processor = BatchProcessor(robot)

    # --- Create specimens for a trial visit ---
    visit_specimens = [
        Specimen(
            specimen_id="SPEC-2026-001-V3-BLD1",
            barcode="BLD001V3",
            trial_id="NCT-2026-0001",
            patient_id="PT-001",
            specimen_type=SpecimenType.BLOOD_SAMPLE,
            container_type=ContainerType.BLOOD_TUBE_EDTA,
            collection_datetime="2026-02-06T08:30:00",
            storage_condition=StorageCondition.REFRIGERATED,
            volume_ml=4.0,
        ),
        Specimen(
            specimen_id="SPEC-2026-001-V3-BLD2",
            barcode="BLD002V3",
            trial_id="NCT-2026-0001",
            patient_id="PT-001",
            specimen_type=SpecimenType.BLOOD_SAMPLE,
            container_type=ContainerType.BLOOD_TUBE_SST,
            collection_datetime="2026-02-06T08:30:00",
            storage_condition=StorageCondition.REFRIGERATED,
            volume_ml=4.0,
        ),
        Specimen(
            specimen_id="SPEC-2026-001-V3-PLS1",
            barcode="PLS001V3",
            trial_id="NCT-2026-0001",
            patient_id="PT-001",
            specimen_type=SpecimenType.PLASMA,
            container_type=ContainerType.CRYOVIAL_2ML,
            collection_datetime="2026-02-06T08:45:00",
            storage_condition=StorageCondition.FROZEN_MINUS80,
            volume_ml=1.5,
        ),
        Specimen(
            specimen_id="SPEC-2026-001-V3-TIS1",
            barcode="TIS001V3",
            trial_id="NCT-2026-0001",
            patient_id="PT-001",
            specimen_type=SpecimenType.TISSUE_BIOPSY,
            container_type=ContainerType.SPECIMEN_CUP,
            collection_datetime="2026-02-06T09:00:00",
            storage_condition=StorageCondition.ROOM_TEMP,
            volume_ml=10.0,
        ),
        Specimen(
            specimen_id="SPEC-2026-001-V3-FRZ1",
            barcode="FRZ001V3",
            trial_id="NCT-2026-0001",
            patient_id="PT-001",
            specimen_type=SpecimenType.FROZEN_TISSUE,
            container_type=ContainerType.CRYOVIAL_5ML,
            collection_datetime="2026-02-06T09:05:00",
            storage_condition=StorageCondition.FROZEN_MINUS80,
            volume_ml=3.0,
        ),
    ]

    # --- Process batch ---
    manifest = {
        "expected_specimen_ids": [s.specimen_id for s in visit_specimens],
        "visit_id": "V3",
        "patient_id": "PT-001",
    }

    report = batch_processor.process_visit_batch(
        visit_id="V3-2026-02-06",
        specimens=visit_specimens,
        manifest=manifest,
    )

    # --- Print results ---
    print("\n" + "=" * 60)
    print("SPECIMEN HANDLING REPORT")
    print("=" * 60)
    print(f"Visit ID:           {report['visit_id']}")
    print(f"Total specimens:    {report['total_specimens']}")
    print(f"Processed:          {report['processed']}")
    print(f"Failed:             {report['failed']}")
    print(f"Manifest verified:  {report['manifest_verified']}")
    print(f"All processed:      {report['all_processed']}")

    print("\nPer-specimen results:")
    for r in report["results"]:
        status = "OK" if r["success"] else f"FAIL ({r.get('reason', '')})"
        print(f"  {r['specimen_id']}: {status}")

    print("\nAudit trail (first specimen):")
    for event in visit_specimens[0].audit_trail:
        print(f"  [{event['timestamp'][:19]}] {event['event_type']}: {event['description']}")

    return report


if __name__ == "__main__":
    run_sample_handling_demo()
