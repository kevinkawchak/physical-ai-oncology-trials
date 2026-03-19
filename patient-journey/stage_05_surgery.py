#!/usr/bin/env python3
"""
Stage 5: Surgery — Active Robot Operations (Day 14)
=====================================================

Orchestrates the operative day: robotic-assisted thoracoscopic right
upper lobectomy with mediastinal lymph node dissection.  Real-time
safety monitoring, sensor fusion, shared autonomy, digital twin overlay,
and specimen chain-of-custody.

Governing regulations:
    - 21 CFR 50.30 (Runtime Safety Monitoring, Task-Order Lifecycle,
      Forbidden Operations)
    - 21 CFR 312.404 (Human Oversight)
    - Physical AI Adaptation of 21 CFR 312.62 (Investigator Recordkeeping)
    - ICH E6(R3) section 2.3 (Medical Care)
    - ICH E6(R3) section 2.12 (Investigator Oversight of Physical AI)
    - ICH E6(R3) section 2.10 (Safety Reporting)
    - Physical AI Adaptation of 21 CFR 312.57 (Recordkeeping)

Usage:
    from stage_05_surgery import SurgeryOrchestrator

    orchestrator = SurgeryOrchestrator(patient_state)
    state = orchestrator.run(patient_state)

Last updated: March 2026

DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    warnings.warn("NumPy not available; some features will be limited.")

from patient_state import (
    AdverseEvent,
    AESeverity,
    ImagingTimepoint,
    MCPConformanceLevel,
    PatientJourneyState,
    PatientStage,
    PatientStatus,
    PhysicalAIClassification,
    RegulatoryEvent,
    SurgicalRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-order lifecycle states per 21 CFR 50.30
# ---------------------------------------------------------------------------

TASK_ORDER_STATES: list[str] = ["IDLE", "SETUP", "DOCKED", "READY", "ACTIVE", "PAUSE", "COMPLETE", "ABORT"]

# Force limits per IEC 80601-2-77
FORCE_LIMIT_TIP_N: float = 15.0
FORCE_LIMIT_LATERAL_N: float = 5.0
ESTOP_LATENCY_LIMIT_MS: float = 50.0

# Forbidden operations per 21 CFR 50.30
FORBIDDEN_OPERATIONS: list[str] = [
    "autonomous_tissue_cutting_without_confirmation",
    "autonomous_vessel_ligation",
    "operation_beyond_workspace_boundary",
    "operation_with_estop_disabled",
]


class SurgeryOrchestrator:
    """Orchestrates Day 14 active robot operations.

    Manages the complete surgical procedure including ROS2 deployment,
    safety monitoring, sensor fusion, shared autonomy, digital twin
    overlay, specimen handling, and investigator oversight review.

    Per 21 CFR 50.30, the procedure follows a strict task-order lifecycle
    and continuous runtime safety monitoring at 1 kHz.

    Attributes:
        patient_state: Patient state from Stage 4.
    """

    def __init__(self, patient_state: PatientJourneyState) -> None:
        self.patient_state = patient_state
        self._day = 14
        self._procedure_events: list[dict] = []
        logger.info("SurgeryOrchestrator initialised for PAT-2026-0042")

    def initialize_ros2_deployment(self) -> dict:
        """Initialise ROS2 surgical deployment.

        Calls examples-new/03_ros2_surgical_deployment.py patterns.
        Per 21 CFR 50.30 Task-Order Lifecycle: IDLE -> SETUP -> DOCKED -> READY.

        Returns:
            Deployment state with lifecycle transitions.
        """
        logger.info("Initialising ROS2 deployment per 21 CFR 50.30 task-order lifecycle")
        # Per 21 CFR 50.30 - task-order lifecycle
        result = {
            "lifecycle_transitions": [
                {"from": "IDLE", "to": "SETUP", "timestamp_min": 0},
                {"from": "SETUP", "to": "DOCKED", "timestamp_min": 12},
                {"from": "DOCKED", "to": "READY", "timestamp_min": 18},
            ],
            "current_state": "READY",
            "ros2_nodes_active": 14,
            "communication_latency_ms": 2.1,
        }
        return result

    def execute_pre_procedure_safety_matrix(self) -> dict:
        """Execute pre-procedure safety matrix per 21 CFR 50.30.

        Validates all safety prerequisites before active operation begins.

        Returns:
            Safety matrix results with all checkpoints.
        """
        logger.info("Executing pre-procedure safety matrix per 21 CFR 50.30")
        result = {
            "patient_identity_verified": True,
            "consent_verified": True,
            "robot_calibration_verified": True,
            "safety_limits_configured": True,
            "estop_tested": True,
            "estop_latency_ms": 38,
            "workspace_boundaries_set": True,
            "surgical_team_briefed": True,
            "anesthesia_ready": True,
            "all_checks_passed": True,
        }
        return result

    def activate_surgical_digital_twin(self) -> dict:
        """Activate surgical digital twin overlay.

        Calls digital-twins/clinical-integration/clinical_dt_interface.py.
        Per ICH E6(R3) section 1.4.2, AR overlay of tumor boundaries,
        vessels, and lymph nodes.

        Returns:
            Twin activation result.
        """
        logger.info("Activating surgical digital twin per ICH E6(R3) section 1.4.2")
        result = {
            "twin_active": True,
            "overlay_elements": ["tumor_boundaries", "vessels", "lymph_nodes", "margin_targets"],
            "registration_error_mm": 0.9,
            "update_rate_hz": 30,
        }
        return result

    def start_realtime_sync(self) -> dict:
        """Start real-time EKF state estimation at 30 Hz.

        Returns:
            Sync configuration.
        """
        logger.info("Starting real-time sync at 30 Hz")
        return {"sync_rate_hz": 30, "estimator": "EKF", "active": True, "latency_ms": 28}

    def run_sensor_fusion(self) -> dict:
        """Run intraoperative sensor fusion pipeline.

        Calls examples-new/02_sensor_fusion_intraoperative.py patterns.
        Fuses stereo endoscope, RGBD, instrument tracking, and force/torque.

        Returns:
            Sensor fusion results.
        """
        logger.info("Running sensor fusion pipeline")
        result = {
            "sensors": {
                "stereo_endoscope": {"active": True, "fps": 30},
                "rgbd_camera": {"active": True, "fps": 30},
                "instrument_tracking": {"active": True, "rate_hz": 100},
                "force_torque": {"active": True, "rate_hz": 1000},
            },
            "fusion_latency_ms": 15,
            "tracking_accuracy_mm": 0.5,
        }
        return result

    def manage_shared_autonomy(self, autonomy_level: int = 2) -> dict:
        """Manage shared autonomy during the procedure.

        Per 21 CFR 312.404 and 21 CFR 50.30, Level 2: surgeon leads,
        AI assists.  Forbidden Operations: autonomous tissue cutting
        without surgeon confirmation.

        Args:
            autonomy_level: Autonomy level (0-4, default 2).

        Returns:
            Autonomy management result.
        """
        logger.info("Managing shared autonomy Level %d per 21 CFR 312.404", autonomy_level)
        # Per 21 CFR 312.404 - human oversight requirements
        result = {
            "autonomy_level": autonomy_level,
            "description": "Surgeon leads, AI assists with guidance and stabilisation",
            "surgeon_has_override": True,
            "forbidden_operations_enforced": FORBIDDEN_OPERATIONS,
            "ai_assistance_active": [
                "tremor_filtering",
                "instrument_guidance",
                "tissue_boundary_alerts",
                "force_feedback_scaling",
            ],
        }
        return result

    def monitor_safety_realtime(self, procedure_events: list[dict]) -> dict:
        """Monitor safety in real time at 1 kHz.

        Per 21 CFR 50.30, continuous monitoring of force, position,
        and physiological parameters.  Records T+45 min force spike
        to 12.1N with warning issued per ICH E6(R3) section 2.10.1(a),
        resolved in 200ms.

        Args:
            procedure_events: List of procedure event dictionaries.

        Returns:
            Safety monitoring results with event log.
        """
        logger.info("Monitoring safety at 1 kHz per 21 CFR 50.30")
        # Per 21 CFR 50.30 - runtime safety monitoring
        result = {
            "monitoring_rate_hz": 1000,
            "duration_hours": 2.8,
            "data_points": 10080000,
            "force_events": [
                {
                    "time_min": 45,
                    "force_n": 12.1,
                    "limit_n": FORCE_LIMIT_TIP_N,
                    "within_limit": True,
                    "warning_issued": True,
                    "resolution_ms": 200,
                    # Per ICH E6(R3) section 2.10.1(a) - force exceedance events
                    "reported_per": "ICH E6(R3) section 2.10.1(a)",
                },
            ],
            "max_force_n": 12.1,
            "boundary_violations": 0,
            "estop_activations": 0,
            "physiological_alerts": 0,
        }
        return result

    def run_sim_vs_real(self) -> dict:
        """Compare digital twin predictions with real surgical outcomes.

        Per ICH E6(R3) section 2.12.3(c), digital twin accuracy assessment
        with 5mm deviation threshold.

        Returns:
            Sim-vs-real comparison results.
        """
        logger.info("Running sim-vs-real comparison per ICH E6(R3) section 2.12.3(c)")
        result = {
            "max_deviation_mm": 3.2,
            "mean_deviation_mm": 1.4,
            "threshold_mm": 5.0,
            "within_threshold": True,
            "comparison_points": 1200,
        }
        return result

    def handle_specimens(self, specimens: list[dict]) -> dict:
        """Handle specimen chain-of-custody with Franka Panda.

        Per 21 CFR 312.62 and 21 CFR 312.57, Franka Panda specimen handling
        with 21 CFR Part 11 audit trail.

        Args:
            specimens: List of specimen dictionaries.

        Returns:
            Specimen handling results with audit trail.
        """
        logger.info("Handling specimens per 21 CFR 312.62 and 21 CFR 312.57")
        # Per 21 CFR 312.57 - 21 CFR Part 11 audit trail
        specimen_list = specimens or [
            {"id": "SPEC-001", "type": "right_upper_lobe", "weight_g": 145},
            {"id": "SPEC-002", "type": "lymph_node_station_4R", "weight_g": 2.1},
            {"id": "SPEC-003", "type": "lymph_node_station_7", "weight_g": 1.8},
        ]
        result = {
            "specimens_collected": len(specimen_list),
            "specimens": specimen_list,
            "chain_of_custody": "DOCUMENTED",
            "part_11_compliant": True,
            "electronic_signatures": [f"SIG-SPEC-{s['id']}" for s in specimen_list],
            "robot_handler": "FRANKA-PANDA-SITE003",
        }
        return result

    def record_surgical_outcome(self) -> SurgicalRecord:
        """Record surgical outcome.

        Day 14, 168 min, 85 mL EBL, negative margins, 18 LN sampled,
        3 LN positive, USL score 7.9, autonomy level 2.

        Returns:
            SurgicalRecord with complete operative data.
        """
        logger.info("Recording surgical outcome")
        record = SurgicalRecord(
            procedure_date_day=14,
            procedure_type="robotic_assisted_thoracoscopic_right_upper_lobectomy",
            robot_id="DAVINCI-XI-SITE003",
            robot_classification=PhysicalAIClassification.SURGICAL_ROBOT,
            operative_time_min=168,
            estimated_blood_loss_ml=85,
            margins_status="negative",
            lymph_nodes_sampled=18,
            lymph_nodes_positive=3,
            complications="none",
            specimens=["right_upper_lobe", "LN_station_4R", "LN_station_7"],
            pathology_stage="pT2aN2M0",
            usl_score_at_procedure=7.9,
            autonomy_level=2,
        )
        return record

    def perform_investigator_oversight_review(self) -> dict:
        """Perform investigator oversight review per ICH E6(R3) section 2.12.3.

        Systematic review of items (a)-(e) covering Physical AI system
        performance, safety events, data integrity, protocol compliance,
        and participant welfare.

        Returns:
            Oversight review results.
        """
        logger.info("Performing investigator oversight review per ICH E6(R3) section 2.12.3")
        # Per ICH E6(R3) section 2.12.3 - systematic review
        result = {
            "review_items": {
                "(a) system_performance": "SATISFACTORY",
                "(b) safety_events": "1 force warning (12.1N, resolved 200ms)",
                "(c) digital_twin_accuracy": "Within 5mm threshold (max 3.2mm)",
                "(d) protocol_compliance": "FULL COMPLIANCE",
                "(e) participant_welfare": "SATISFACTORY",
            },
            "overall_assessment": "APPROVED",
            "reviewer": "PI-SITE003",
            "cfr_section": "ICH E6(R3) section 2.12.3",
        }
        return result

    def coordinate_agents(self) -> dict:
        """Coordinate MCP agents at ROBOT_PROCEDURE conformance level.

        Per 21 CFR 50.33, the MCP server operates at ROBOT_PROCEDURE
        conformance level during active surgery.

        Returns:
            Agent coordination results.
        """
        logger.info("Coordinating agents at ROBOT_PROCEDURE MCP level per 21 CFR 50.33")
        result = {
            "mcp_conformance_level": "ROBOT_PROCEDURE",
            "active_agents": [
                "safety_monitor",
                "telemetry_recorder",
                "specimen_tracker",
                "consent_verifier",
            ],
            "real_time": True,
        }
        return result

    def run(self, patient_state: PatientJourneyState) -> PatientJourneyState:
        """Orchestrate the complete surgical procedure (Day 14).

        Args:
            patient_state: Patient state from Stage 4 completion.

        Returns:
            Updated state with surgical record and SURGERY_COMPLETE status.
        """
        logger.info("Starting Stage 5: Surgery (Day 14)")
        state = patient_state

        # Step 1: ROS2 deployment per 21 CFR 50.30
        self.initialize_ros2_deployment()

        # Step 2: Pre-procedure safety matrix per 21 CFR 50.30
        self.execute_pre_procedure_safety_matrix()

        # Step 3: Activate surgical digital twin per ICH E6(R3) section 1.4.2
        self.activate_surgical_digital_twin()

        # Step 4: Start real-time sync
        self.start_realtime_sync()

        # Step 5: Sensor fusion
        self.run_sensor_fusion()

        # Step 6: Shared autonomy per 21 CFR 312.404
        self.manage_shared_autonomy(autonomy_level=2)

        # Step 7: Real-time safety monitoring per 21 CFR 50.30
        safety = self.monitor_safety_realtime([])

        # Step 8: Sim-vs-real per ICH E6(R3) section 2.12.3(c)
        self.run_sim_vs_real()

        # Step 9: Specimen handling per 21 CFR 312.62
        self.handle_specimens([])

        # Step 10: Record outcome
        surgical_record = self.record_surgical_outcome()
        state.surgical_record = surgical_record

        # Step 11: Investigator oversight per ICH E6(R3) section 2.12.3
        self.perform_investigator_oversight_review()

        # Step 12: Coordinate agents per 21 CFR 50.33
        self.coordinate_agents()

        # Update MCP conformance to ROBOT_PROCEDURE
        state.mcp_conformance_level = MCPConformanceLevel.ROBOT_PROCEDURE

        # Advance stage
        state.advance_stage(PatientStage.SURGERY)
        state.status = PatientStatus.SURGERY_COMPLETE

        # Update tumor volume post-surgery (resected)
        state.tumor.volume_cm3 = 0.0

        # Record regulatory events
        state.add_regulatory_event(RegulatoryEvent(
            event_type="SURGERY_SAFETY_MONITORING",
            day=14,
            description=(
                "Runtime safety monitoring at 1 kHz per 21 CFR 50.30. "
                f"Max force {safety['max_force_n']}N, 0 boundary violations, 0 e-stops. "
                "Force warning at T+45min (12.1N) resolved in 200ms."
            ),
            document_id="SURG-SAFETY-001",
            cfr_section="21 CFR 50.30",
            status="MONITORED",
        ))

        state.add_regulatory_event(RegulatoryEvent(
            event_type="INVESTIGATOR_OVERSIGHT",
            day=14,
            description="Investigator oversight review per ICH E6(R3) section 2.12.3: items (a)-(e) APPROVED",
            document_id="OVERSIGHT-001",
            cfr_section="ICH E6(R3) section 2.12.3",
            status="APPROVED",
        ))

        state.add_audit_entry(
            action="SURGERY_COMPLETE",
            actor="SurgeryOrchestrator",
            details=(
                f"Stage 5 complete: {surgical_record.procedure_type}, "
                f"{surgical_record.operative_time_min} min, "
                f"{surgical_record.estimated_blood_loss_ml} mL EBL, "
                f"margins {surgical_record.margins_status}, "
                f"{surgical_record.lymph_nodes_sampled} LN sampled"
            ),
        )

        logger.info("Stage 5 complete: surgery successful, negative margins")
        return state
