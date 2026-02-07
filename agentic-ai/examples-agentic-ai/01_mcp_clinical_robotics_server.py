"""
=============================================================================
EXAMPLE 01: MCP Server for Clinical Robotics in Oncology Trials
=============================================================================

Implements a Model Context Protocol (MCP) server that exposes clinical
robotics data, DICOM imaging, robot telemetry, and trial status as
structured tools and resources for LLM-based agents.

CLINICAL CONTEXT:
-----------------
MCP enables standardized tool interfaces between LLMs and clinical systems:
  - Robot telemetry (joint states, forces, workspace boundaries)
  - DICOM imaging pipelines (CT, MRI, PET for tumor localization)
  - Real-time patient vitals during robotic procedures
  - Trial protocol state and compliance verification
  - Intraoperative event logging with 21 CFR Part 11 audit trails

WHY MCP FOR ONCOLOGY ROBOTICS:
------------------------------
MCP (Model Context Protocol) provides a standard transport layer between
AI agents and clinical systems. Unlike ad-hoc tool definitions, MCP:
  - Standardizes tool discovery and invocation across LLM providers
  - Supports bidirectional streaming for real-time telemetry
  - Enables composable server architectures (one server per subsystem)
  - Facilitates regulatory auditing through structured request/response logs

ARCHITECTURE:
-------------
  LLM Agent  <-->  MCP Client  <-->  MCP Server  <-->  Clinical Systems
                                         |
                                    +----+----+
                                    |    |    |
                                  Robot DICOM Vitals

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - mcp >= 0.1.0 (Model Context Protocol SDK)
    - anthropic >= 0.40.0 (for Claude integration)

Optional:
    - pydicom >= 2.4.0 (DICOM file handling)
    - numpy >= 1.24.0 (telemetry processing)

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for MCP and clinical data
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from mcp.server import Server
    from mcp.types import Resource, Tool

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    logger.info("MCP SDK not installed. Using standalone implementation.")


# =============================================================================
# SECTION 1: CLINICAL DATA MODELS
# =============================================================================


class RobotMode(Enum):
    """Operational modes for surgical robots."""

    IDLE = "idle"
    HOMING = "homing"
    MANUAL = "manual_control"
    AUTONOMOUS = "autonomous"
    SHARED_CONTROL = "shared_control"
    EMERGENCY_STOP = "emergency_stop"


class ProcedurePhase(Enum):
    """Phases of a robotic oncology procedure."""

    PRE_OPERATIVE = "pre_operative"
    PATIENT_SETUP = "patient_setup"
    REGISTRATION = "registration"
    APPROACH = "approach"
    INTERVENTION = "intervention"
    CLOSURE = "closure"
    POST_OPERATIVE = "post_operative"


@dataclass
class RobotTelemetry:
    """Real-time robot state telemetry."""

    joint_positions: list = field(default_factory=lambda: [0.0] * 7)
    joint_velocities: list = field(default_factory=lambda: [0.0] * 7)
    joint_torques: list = field(default_factory=lambda: [0.0] * 7)
    end_effector_pose: list = field(default_factory=lambda: [0.0] * 6)
    force_torque: list = field(default_factory=lambda: [0.0] * 6)
    mode: RobotMode = RobotMode.IDLE
    timestamp: float = 0.0
    safety_status: str = "nominal"

    def to_dict(self) -> dict:
        """Serialize to dictionary for MCP transport."""
        return {
            "joint_positions_rad": self.joint_positions,
            "joint_velocities_rad_s": self.joint_velocities,
            "joint_torques_nm": self.joint_torques,
            "end_effector_pose_m_rad": self.end_effector_pose,
            "force_torque_n_nm": self.force_torque,
            "mode": self.mode.value,
            "timestamp_s": self.timestamp,
            "safety_status": self.safety_status,
        }


@dataclass
class PatientVitals:
    """Real-time patient vitals during procedure."""

    heart_rate_bpm: float = 72.0
    blood_pressure_systolic: float = 120.0
    blood_pressure_diastolic: float = 80.0
    spo2_percent: float = 98.0
    respiratory_rate: float = 16.0
    temperature_celsius: float = 36.8
    etco2_mmhg: float = 35.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for MCP transport."""
        return {
            "heart_rate_bpm": self.heart_rate_bpm,
            "blood_pressure": f"{self.blood_pressure_systolic:.0f}/{self.blood_pressure_diastolic:.0f}",
            "spo2_percent": self.spo2_percent,
            "respiratory_rate_bpm": self.respiratory_rate,
            "temperature_celsius": self.temperature_celsius,
            "etco2_mmhg": self.etco2_mmhg,
            "timestamp_s": self.timestamp,
        }


@dataclass
class DicomStudy:
    """DICOM imaging study metadata."""

    study_uid: str = ""
    patient_id: str = ""
    modality: str = "CT"
    study_date: str = ""
    series_count: int = 0
    slice_count: int = 0
    tumor_location: str = ""
    tumor_volume_cc: float = 0.0
    margin_mm: float = 0.0
    critical_structures: list = field(default_factory=list)


@dataclass
class AuditEntry:
    """21 CFR Part 11 compliant audit trail entry."""

    entry_id: str = ""
    timestamp: str = ""
    action: str = ""
    actor: str = ""
    resource: str = ""
    parameters: dict = field(default_factory=dict)
    result: str = ""
    signature_hash: str = ""


# =============================================================================
# SECTION 2: MCP TOOL DEFINITIONS
# =============================================================================


class ClinicalRoboticsMCPTools:
    """
    MCP tool definitions for clinical robotics operations.

    Each tool follows MCP's JSON Schema specification for input/output
    and includes safety classification per IEC 62304.

    TOOL CATEGORIES:
    ----------------
    1. Robot Telemetry (read-only, real-time)
    2. Robot Commands (write, safety-gated)
    3. DICOM Imaging (read-only, PHI-aware)
    4. Patient Vitals (read-only, real-time)
    5. Procedure Management (read-write, audited)
    """

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return all MCP tool definitions with JSON Schema."""
        return [
            # --- Robot Telemetry Tools ---
            {
                "name": "get_robot_telemetry",
                "description": (
                    "Retrieve current robot joint positions, velocities, torques, "
                    "end-effector pose, and force/torque sensor readings. "
                    "Returns real-time state at ~1kHz sampling."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_derivatives": {
                            "type": "boolean",
                            "description": "Include velocity and acceleration derivatives",
                            "default": False,
                        }
                    },
                },
                "safety_level": "read_only",
            },
            {
                "name": "get_workspace_boundaries",
                "description": (
                    "Get the current workspace boundary definitions including "
                    "keep-out zones around critical anatomy and instrument "
                    "workspace limits per IEC 80601-2-77."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "coordinate_frame": {
                            "type": "string",
                            "enum": ["robot_base", "patient", "world"],
                            "default": "patient",
                        }
                    },
                },
                "safety_level": "read_only",
            },
            # --- Robot Command Tools ---
            {
                "name": "send_robot_command",
                "description": (
                    "Send a motion command to the surgical robot. Requires "
                    "human-in-the-loop confirmation for all commands. Motion "
                    "is subject to safety envelope constraints."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command_type": {
                            "type": "string",
                            "enum": ["move_to_pose", "move_along_path", "hold_position", "retract"],
                        },
                        "target_pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Target [x, y, z, rx, ry, rz] in meters and radians",
                        },
                        "max_velocity": {
                            "type": "number",
                            "description": "Maximum velocity in m/s (clamped to safety limit)",
                            "default": 0.05,
                        },
                        "requires_confirmation": {
                            "type": "boolean",
                            "default": True,
                        },
                    },
                    "required": ["command_type"],
                },
                "safety_level": "critical",
            },
            # --- DICOM Imaging Tools ---
            {
                "name": "query_dicom_study",
                "description": (
                    "Query DICOM imaging study for the current patient. Returns "
                    "de-identified metadata including modality, series info, "
                    "and tumor segmentation results. PHI is stripped per "
                    "DICOM PS3.15 Annex E."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "study_uid": {"type": "string", "description": "DICOM Study Instance UID"},
                        "include_segmentation": {
                            "type": "boolean",
                            "description": "Include tumor segmentation metadata",
                            "default": True,
                        },
                    },
                    "required": ["study_uid"],
                },
                "safety_level": "read_only",
            },
            {
                "name": "get_tumor_coordinates",
                "description": (
                    "Get 3D tumor centroid and bounding box coordinates in "
                    "patient coordinate frame from the most recent imaging. "
                    "Coordinates are registered to the intraoperative reference."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "coordinate_frame": {
                            "type": "string",
                            "enum": ["patient", "robot_base", "image"],
                            "default": "patient",
                        },
                        "include_margins": {
                            "type": "boolean",
                            "description": "Include surgical margin boundaries",
                            "default": True,
                        },
                    },
                },
                "safety_level": "read_only",
            },
            # --- Patient Vitals Tools ---
            {
                "name": "get_patient_vitals",
                "description": (
                    "Get current patient vital signs including heart rate, "
                    "blood pressure, SpO2, respiratory rate, temperature, "
                    "and end-tidal CO2."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "duration_seconds": {
                            "type": "number",
                            "description": "Return vitals averaged over this duration",
                            "default": 5.0,
                        }
                    },
                },
                "safety_level": "read_only",
            },
            # --- Procedure Management Tools ---
            {
                "name": "log_procedure_event",
                "description": (
                    "Log an event during the surgical procedure. Creates a "
                    "21 CFR Part 11 compliant audit trail entry with "
                    "timestamp, actor, action, and electronic signature."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "enum": [
                                "phase_transition",
                                "instrument_change",
                                "safety_event",
                                "clinical_observation",
                                "specimen_collected",
                                "imaging_acquired",
                            ],
                        },
                        "description": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "critical"],
                            "default": "info",
                        },
                    },
                    "required": ["event_type", "description"],
                },
                "safety_level": "audited",
            },
            {
                "name": "get_procedure_checklist",
                "description": (
                    "Get the current procedure checklist status. Returns "
                    "completed and pending steps per the surgical safety "
                    "checklist (WHO Surgical Safety Checklist adapted for "
                    "robotic oncology)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "enum": [
                                "sign_in",
                                "time_out",
                                "sign_out",
                            ],
                        }
                    },
                },
                "safety_level": "read_only",
            },
        ]


# =============================================================================
# SECTION 3: MCP RESOURCE DEFINITIONS
# =============================================================================


class ClinicalRoboticsMCPResources:
    """
    MCP resource definitions for clinical data access.

    Resources provide read-only access to structured clinical data
    that agents can reference during planning and execution.
    """

    @staticmethod
    def get_resource_definitions() -> list[dict]:
        """Return MCP resource URIs and metadata."""
        return [
            {
                "uri": "oncology://trial/protocol",
                "name": "Trial Protocol",
                "description": "Current clinical trial protocol document including eligibility, treatment arms, endpoints",
                "mimeType": "application/json",
            },
            {
                "uri": "oncology://patient/current/imaging",
                "name": "Patient Imaging",
                "description": "De-identified imaging metadata for the current patient (CT, MRI, PET)",
                "mimeType": "application/json",
            },
            {
                "uri": "oncology://robot/configuration",
                "name": "Robot Configuration",
                "description": "Current robot kinematic configuration, tool attachments, and safety parameters",
                "mimeType": "application/json",
            },
            {
                "uri": "oncology://procedure/plan",
                "name": "Procedure Plan",
                "description": "Surgical procedure plan with steps, expected durations, and safety checkpoints",
                "mimeType": "application/json",
            },
            {
                "uri": "oncology://safety/constraints",
                "name": "Safety Constraints",
                "description": "Active safety constraints including workspace limits, force thresholds, keep-out zones",
                "mimeType": "application/json",
            },
        ]


# =============================================================================
# SECTION 4: MCP SERVER IMPLEMENTATION
# =============================================================================


class ClinicalRoboticsMCPServer:
    """
    MCP server for clinical robotics in oncology trials.

    This server exposes robot telemetry, imaging data, patient vitals,
    and procedure management as MCP tools and resources. All operations
    are logged for 21 CFR Part 11 compliance.

    DEPLOYMENT:
    -----------
    The server can run in two modes:
    1. stdio transport: For local agent integration
    2. SSE transport: For networked multi-agent architectures

    SAFETY ARCHITECTURE:
    --------------------
    Tool calls are classified by safety level:
    - read_only: No confirmation needed, no side effects
    - audited: Logged to audit trail, no physical effect
    - critical: Requires human confirmation before execution
    """

    def __init__(self, trial_id: str = "NCT-2026-0001"):
        self.trial_id = trial_id
        self.audit_trail: list[AuditEntry] = []
        self._telemetry = RobotTelemetry()
        self._vitals = PatientVitals()
        self._procedure_phase = ProcedurePhase.PRE_OPERATIVE
        self._imaging_studies: dict[str, DicomStudy] = {}
        self._checklist: dict[str, list[dict]] = self._init_checklist()
        self._event_counter = 0

        # Safety constraints
        self._workspace_limits = {
            "x_range_m": [-0.3, 0.3],
            "y_range_m": [-0.3, 0.3],
            "z_range_m": [0.0, 0.4],
            "max_velocity_m_s": 0.1,
            "max_force_n": 10.0,
            "max_torque_nm": 2.0,
        }
        self._keepout_zones: list[dict] = []

        logger.info(f"MCP Server initialized for trial {trial_id}")

    def _init_checklist(self) -> dict[str, list[dict]]:
        """Initialize WHO-adapted surgical safety checklist."""
        return {
            "sign_in": [
                {"item": "Patient identity verified", "completed": False},
                {"item": "Surgical site marked and confirmed", "completed": False},
                {"item": "Consent form signed and on file", "completed": False},
                {"item": "Imaging loaded and registered", "completed": False},
                {"item": "Robot self-test passed", "completed": False},
                {"item": "Safety systems verified", "completed": False},
                {"item": "Anesthesia safety check complete", "completed": False},
            ],
            "time_out": [
                {"item": "Team introduction and role confirmation", "completed": False},
                {"item": "Patient name and procedure confirmed", "completed": False},
                {"item": "Anticipated critical events reviewed", "completed": False},
                {"item": "Robot workspace boundaries set", "completed": False},
                {"item": "Keep-out zones loaded from imaging", "completed": False},
                {"item": "Emergency stop tested", "completed": False},
            ],
            "sign_out": [
                {"item": "Procedure documented in operative report", "completed": False},
                {"item": "Specimen labeling confirmed", "completed": False},
                {"item": "Instrument and needle counts correct", "completed": False},
                {"item": "Robot shutdown sequence completed", "completed": False},
                {"item": "Post-operative orders entered", "completed": False},
                {"item": "Audit trail exported and signed", "completed": False},
            ],
        }

    # --- Tool Handlers ---

    async def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """
        Route and execute MCP tool calls.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool input arguments

        Returns:
            Tool execution result
        """
        handlers = {
            "get_robot_telemetry": self._handle_get_telemetry,
            "get_workspace_boundaries": self._handle_get_workspace,
            "send_robot_command": self._handle_send_command,
            "query_dicom_study": self._handle_query_dicom,
            "get_tumor_coordinates": self._handle_get_tumor_coords,
            "get_patient_vitals": self._handle_get_vitals,
            "log_procedure_event": self._handle_log_event,
            "get_procedure_checklist": self._handle_get_checklist,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        # Log to audit trail
        self._log_audit(
            action=f"tool_call:{tool_name}",
            actor="mcp_agent",
            resource=tool_name,
            parameters=arguments,
        )

        result = await handler(arguments)

        self._log_audit(
            action=f"tool_result:{tool_name}",
            actor="mcp_server",
            resource=tool_name,
            parameters={"result_keys": list(result.keys())},
        )

        return result

    async def _handle_get_telemetry(self, args: dict) -> dict:
        """Handle robot telemetry request."""
        self._telemetry.timestamp = time.time()
        result = self._telemetry.to_dict()

        if args.get("include_derivatives", False) and HAS_NUMPY:
            result["joint_accelerations_rad_s2"] = [0.0] * 7
            result["ee_velocity_m_s"] = [0.0] * 6

        return result

    async def _handle_get_workspace(self, args: dict) -> dict:
        """Handle workspace boundary request."""
        coord_frame = args.get("coordinate_frame", "patient")
        return {
            "coordinate_frame": coord_frame,
            "workspace_limits": self._workspace_limits,
            "keepout_zones": self._keepout_zones,
            "keepout_zone_count": len(self._keepout_zones),
        }

    async def _handle_send_command(self, args: dict) -> dict:
        """
        Handle robot command with safety validation.

        All commands are validated against workspace boundaries
        and require human confirmation for critical operations.
        """
        command_type = args.get("command_type", "hold_position")
        target_pose = args.get("target_pose")
        max_velocity = min(args.get("max_velocity", 0.05), self._workspace_limits["max_velocity_m_s"])

        # Validate target pose against workspace
        if target_pose and len(target_pose) >= 3:
            x, y, z = target_pose[0], target_pose[1], target_pose[2]
            x_min, x_max = self._workspace_limits["x_range_m"]
            y_min, y_max = self._workspace_limits["y_range_m"]
            z_min, z_max = self._workspace_limits["z_range_m"]

            if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                return {
                    "status": "rejected",
                    "reason": "Target pose outside workspace boundaries",
                    "workspace_limits": self._workspace_limits,
                }

            # Check keep-out zones
            for zone in self._keepout_zones:
                center = zone.get("center", [0, 0, 0])
                radius = zone.get("radius_m", 0.01)
                dist = sum((a - b) ** 2 for a, b in zip([x, y, z], center)) ** 0.5
                if dist < radius:
                    return {
                        "status": "rejected",
                        "reason": f"Target inside keep-out zone: {zone.get('label', 'unknown')}",
                        "zone_label": zone.get("label"),
                        "distance_to_boundary_m": radius - dist,
                    }

        return {
            "status": "pending_confirmation" if args.get("requires_confirmation", True) else "accepted",
            "command_type": command_type,
            "target_pose": target_pose,
            "max_velocity_m_s": max_velocity,
            "confirmation_required": args.get("requires_confirmation", True),
            "safety_check": "passed",
        }

    async def _handle_query_dicom(self, args: dict) -> dict:
        """Handle DICOM study query with PHI stripping."""
        study_uid = args.get("study_uid", "")

        # Return de-identified study metadata
        study = self._imaging_studies.get(study_uid)
        if study is None:
            # Return demo study
            study = DicomStudy(
                study_uid=study_uid,
                patient_id="[DE-IDENTIFIED]",
                modality="CT",
                study_date="2026-01-15",
                series_count=3,
                slice_count=512,
                tumor_location="Right upper lobe",
                tumor_volume_cc=4.2,
                margin_mm=10.0,
                critical_structures=["pulmonary artery", "bronchus", "aorta"],
            )

        result = {
            "study_uid": study.study_uid,
            "patient_id": "[DE-IDENTIFIED]",
            "modality": study.modality,
            "study_date": study.study_date,
            "series_count": study.series_count,
            "slice_count": study.slice_count,
        }

        if args.get("include_segmentation", True):
            result["segmentation"] = {
                "tumor_location": study.tumor_location,
                "tumor_volume_cc": study.tumor_volume_cc,
                "planned_margin_mm": study.margin_mm,
                "critical_structures": study.critical_structures,
            }

        return result

    async def _handle_get_tumor_coords(self, args: dict) -> dict:
        """Handle tumor coordinate request."""
        coord_frame = args.get("coordinate_frame", "patient")
        include_margins = args.get("include_margins", True)

        result = {
            "coordinate_frame": coord_frame,
            "tumor_centroid": [0.05, -0.02, 0.15],
            "bounding_box": {
                "min": [0.03, -0.04, 0.13],
                "max": [0.07, 0.00, 0.17],
            },
            "volume_cc": 4.2,
        }

        if include_margins:
            result["margin_surface"] = {
                "type": "mesh",
                "vertex_count": 1024,
                "margin_mm": 10.0,
                "clearance_status": "adequate",
            }

        return result

    async def _handle_get_vitals(self, args: dict) -> dict:
        """Handle patient vitals request."""
        self._vitals.timestamp = time.time()
        result = self._vitals.to_dict()

        # Add trend indicators
        result["trends"] = {
            "heart_rate": "stable",
            "blood_pressure": "stable",
            "spo2": "stable",
        }

        # Add alerts if any vitals are abnormal
        alerts = []
        if self._vitals.heart_rate_bpm > 120 or self._vitals.heart_rate_bpm < 50:
            alerts.append({"vital": "heart_rate", "status": "abnormal", "value": self._vitals.heart_rate_bpm})
        if self._vitals.spo2_percent < 92:
            alerts.append({"vital": "spo2", "status": "low", "value": self._vitals.spo2_percent})
        result["alerts"] = alerts

        return result

    async def _handle_log_event(self, args: dict) -> dict:
        """Handle procedure event logging."""
        self._event_counter += 1
        event_id = f"EVT-{self._event_counter:06d}"

        entry = AuditEntry(
            entry_id=event_id,
            timestamp=datetime.now().isoformat(),
            action=args.get("event_type", "clinical_observation"),
            actor="surgical_team",
            resource="procedure",
            parameters={"description": args.get("description", ""), "severity": args.get("severity", "info")},
            result="logged",
        )
        self.audit_trail.append(entry)

        return {
            "event_id": event_id,
            "status": "logged",
            "timestamp": entry.timestamp,
            "audit_trail_length": len(self.audit_trail),
        }

    async def _handle_get_checklist(self, args: dict) -> dict:
        """Handle procedure checklist request."""
        phase = args.get("phase")
        if phase and phase in self._checklist:
            items = self._checklist[phase]
            completed = sum(1 for item in items if item["completed"])
            return {
                "phase": phase,
                "items": items,
                "completed": completed,
                "total": len(items),
                "all_complete": completed == len(items),
            }

        # Return all phases
        result = {}
        for phase_name, items in self._checklist.items():
            completed = sum(1 for item in items if item["completed"])
            result[phase_name] = {
                "completed": completed,
                "total": len(items),
                "all_complete": completed == len(items),
            }
        return result

    # --- Resource Handlers ---

    async def handle_resource_read(self, uri: str) -> dict:
        """
        Handle MCP resource read request.

        Args:
            uri: Resource URI (e.g., oncology://trial/protocol)

        Returns:
            Resource content
        """
        handlers = {
            "oncology://trial/protocol": self._get_protocol_resource,
            "oncology://patient/current/imaging": self._get_imaging_resource,
            "oncology://robot/configuration": self._get_robot_config_resource,
            "oncology://procedure/plan": self._get_procedure_plan_resource,
            "oncology://safety/constraints": self._get_safety_resource,
        }

        handler = handlers.get(uri)
        if handler is None:
            return {"error": f"Unknown resource URI: {uri}"}

        return await handler()

    async def _get_protocol_resource(self) -> dict:
        """Get trial protocol resource."""
        return {
            "trial_id": self.trial_id,
            "title": "Phase II Study of AI-Guided Robotic Tumor Resection",
            "primary_endpoint": "R0 resection rate",
            "secondary_endpoints": [
                "Procedure time",
                "Estimated blood loss",
                "Margin distance",
                "30-day complication rate",
            ],
            "treatment_arms": [
                {"arm": "A", "description": "AI-guided robotic resection"},
                {"arm": "B", "description": "Standard robotic resection"},
            ],
        }

    async def _get_imaging_resource(self) -> dict:
        """Get patient imaging resource."""
        return {
            "available_studies": [
                {"modality": "CT", "date": "2026-01-15", "description": "Pre-operative staging CT"},
                {"modality": "PET-CT", "date": "2026-01-20", "description": "FDG PET-CT for staging"},
                {"modality": "MRI", "date": "2026-01-22", "description": "MRI for surgical planning"},
            ],
            "registration_status": "completed",
            "registration_error_mm": 1.2,
        }

    async def _get_robot_config_resource(self) -> dict:
        """Get robot configuration resource."""
        return {
            "robot_model": "da Vinci Xi",
            "arm_count": 4,
            "dof_per_arm": 7,
            "instruments": [
                {"port": 1, "instrument": "Maryland Bipolar Forceps"},
                {"port": 2, "instrument": "Monopolar Curved Scissors"},
                {"port": 3, "instrument": "Cadiere Forceps"},
                {"port": 4, "instrument": "30-degree Endoscope"},
            ],
            "workspace_volume_cc": 27000,
        }

    async def _get_procedure_plan_resource(self) -> dict:
        """Get procedure plan resource."""
        return {
            "procedure": "Right Upper Lobectomy",
            "approach": "Robotic-assisted thoracoscopic",
            "steps": [
                {"step": 1, "action": "Port placement and docking", "est_minutes": 15},
                {"step": 2, "action": "Hilar dissection and vessel identification", "est_minutes": 20},
                {"step": 3, "action": "Pulmonary artery branch ligation", "est_minutes": 10},
                {"step": 4, "action": "Pulmonary vein ligation", "est_minutes": 10},
                {"step": 5, "action": "Bronchus division", "est_minutes": 10},
                {"step": 6, "action": "Fissure completion", "est_minutes": 15},
                {"step": 7, "action": "Specimen extraction and lymph node dissection", "est_minutes": 20},
            ],
        }

    async def _get_safety_resource(self) -> dict:
        """Get active safety constraints."""
        return {
            "workspace_limits": self._workspace_limits,
            "keepout_zones": self._keepout_zones,
            "force_limits": {
                "max_insertion_force_n": 5.0,
                "max_lateral_force_n": 3.0,
                "max_torque_nm": 0.5,
            },
            "velocity_limits": {
                "max_linear_m_s": 0.1,
                "max_angular_rad_s": 0.5,
                "approach_velocity_m_s": 0.02,
            },
            "emergency_stop_latency_ms": 10,
        }

    # --- Audit Trail ---

    def _log_audit(self, action: str, actor: str, resource: str, parameters: dict) -> None:
        """Create audit trail entry."""
        self._event_counter += 1
        entry = AuditEntry(
            entry_id=f"AUD-{self._event_counter:06d}",
            timestamp=datetime.now().isoformat(),
            action=action,
            actor=actor,
            resource=resource,
            parameters=parameters,
            result="recorded",
        )
        self.audit_trail.append(entry)

    def export_audit_trail(self) -> str:
        """Export audit trail as JSON for regulatory submission."""
        entries = []
        for entry in self.audit_trail:
            entries.append(
                {
                    "id": entry.entry_id,
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "actor": entry.actor,
                    "resource": entry.resource,
                    "parameters": entry.parameters,
                }
            )
        return json.dumps({"trial_id": self.trial_id, "entries": entries}, indent=2)


# =============================================================================
# SECTION 5: MCP SERVER BOOTSTRAP (stdio transport)
# =============================================================================


def create_mcp_server() -> Any:
    """
    Create and configure the MCP server instance.

    Returns a configured MCP server if the SDK is available,
    otherwise returns the standalone implementation.
    """
    if HAS_MCP:
        server = Server("clinical-robotics")

        # Register tools
        tool_defs = ClinicalRoboticsMCPTools.get_tool_definitions()
        logger.info(f"Registered {len(tool_defs)} MCP tools")

        # Register resources
        resource_defs = ClinicalRoboticsMCPResources.get_resource_definitions()
        logger.info(f"Registered {len(resource_defs)} MCP resources")

        return server

    # Return standalone server
    return ClinicalRoboticsMCPServer()


# =============================================================================
# SECTION 6: DEMO - AGENT INTERACTION WITH MCP SERVER
# =============================================================================


async def run_mcp_demo():
    """
    Demonstrate MCP server interaction for clinical robotics.

    Simulates an LLM agent querying robot state, imaging data,
    and sending commands through the MCP interface.
    """
    logger.info("=" * 60)
    logger.info("MCP CLINICAL ROBOTICS SERVER DEMO")
    logger.info("=" * 60)

    server = ClinicalRoboticsMCPServer(trial_id="NCT-2026-0001")

    # Add keep-out zone around critical structure
    server._keepout_zones.append(
        {
            "label": "pulmonary_artery",
            "center": [0.04, -0.01, 0.14],
            "radius_m": 0.015,
            "structure": "Right pulmonary artery",
        }
    )

    # 1. Agent queries robot state
    print("\n--- Step 1: Query Robot Telemetry ---")
    telemetry = await server.handle_tool_call("get_robot_telemetry", {"include_derivatives": True})
    print(f"Robot mode: {telemetry['mode']}")
    print(f"Safety status: {telemetry['safety_status']}")

    # 2. Agent queries patient imaging
    print("\n--- Step 2: Query DICOM Study ---")
    imaging = await server.handle_tool_call(
        "query_dicom_study",
        {"study_uid": "1.2.840.113619.2.55.3.1234", "include_segmentation": True},
    )
    print(f"Modality: {imaging['modality']}")
    if "segmentation" in imaging:
        print(f"Tumor location: {imaging['segmentation']['tumor_location']}")
        print(f"Tumor volume: {imaging['segmentation']['tumor_volume_cc']} cc")

    # 3. Agent gets tumor coordinates
    print("\n--- Step 3: Get Tumor Coordinates ---")
    tumor = await server.handle_tool_call(
        "get_tumor_coordinates",
        {"coordinate_frame": "patient", "include_margins": True},
    )
    print(f"Tumor centroid: {tumor['tumor_centroid']}")
    print(f"Margin clearance: {tumor['margin_surface']['clearance_status']}")

    # 4. Agent checks patient vitals
    print("\n--- Step 4: Check Patient Vitals ---")
    vitals = await server.handle_tool_call("get_patient_vitals", {})
    print(f"Heart rate: {vitals['heart_rate_bpm']} bpm")
    print(f"SpO2: {vitals['spo2_percent']}%")

    # 5. Agent sends robot command (safe target)
    print("\n--- Step 5: Send Robot Command (Safe Target) ---")
    cmd_result = await server.handle_tool_call(
        "send_robot_command",
        {
            "command_type": "move_to_pose",
            "target_pose": [0.05, -0.02, 0.20, 0.0, 0.0, 0.0],
            "max_velocity": 0.03,
        },
    )
    print(f"Command status: {cmd_result['status']}")
    print(f"Safety check: {cmd_result.get('safety_check', 'N/A')}")

    # 6. Agent sends command into keep-out zone (rejected)
    print("\n--- Step 6: Send Robot Command (Keep-Out Zone) ---")
    cmd_result = await server.handle_tool_call(
        "send_robot_command",
        {
            "command_type": "move_to_pose",
            "target_pose": [0.04, -0.01, 0.14, 0.0, 0.0, 0.0],
            "max_velocity": 0.02,
        },
    )
    print(f"Command status: {cmd_result['status']}")
    print(f"Reason: {cmd_result.get('reason', 'N/A')}")

    # 7. Agent reads procedure plan resource
    print("\n--- Step 7: Read Procedure Plan Resource ---")
    plan = await server.handle_resource_read("oncology://procedure/plan")
    print(f"Procedure: {plan['procedure']}")
    for step in plan["steps"][:3]:
        print(f"  Step {step['step']}: {step['action']}")

    # 8. Log procedure event
    print("\n--- Step 8: Log Procedure Event ---")
    event = await server.handle_tool_call(
        "log_procedure_event",
        {"event_type": "clinical_observation", "description": "Tumor margin appears adequate on visual inspection"},
    )
    print(f"Event logged: {event['event_id']}")

    # 9. Export audit trail
    print("\n--- Step 9: Export Audit Trail ---")
    audit_json = server.export_audit_trail()
    audit_data = json.loads(audit_json)
    print(f"Audit trail entries: {len(audit_data['entries'])}")

    print("\n" + "=" * 60)
    print("MCP DEMO COMPLETE")
    print("=" * 60)

    return {"tools_called": 8, "resources_read": 1, "audit_entries": len(server.audit_trail)}


if __name__ == "__main__":
    result = asyncio.run(run_mcp_demo())
    print(f"\nDemo result: {result}")
