"""Sponsor activity script for Hour 007 (Day 1, 07:00 - Morning peak operations with full staffing)."""

from __future__ import annotations

import json
from pathlib import Path

HOUR = 7
DAY = 1
HOUR_OF_DAY = 7
PATIENT_COUNT = 10
PERIOD = "morning_peak"

ROBOT_FLEET = [
    {"category": "Surgical Robots (da Vinci Xi)", "instances": 3},
    {"category": "Collaborative Robots (Cobots, Franka Emika)", "instances": 4},
    {"category": "RT Positioning Robots", "instances": 3},
    {"category": "Needle Placement Systems", "instances": 2},
    {"category": "Social Companion Robots", "instances": 5},
    {"category": "Rehabilitation Exoskeletons", "instances": 2},
    {"category": "RT Motion-Tracking Systems", "instances": 3},
    {"category": "Imaging Assistance Robots", "instances": 4},
    {"category": "Transport/Logistics Robots", "instances": 2},
    {"category": "Environmental Monitoring Robots", "instances": 1},
]

AGENTS = [
    "portfolio_agent",
    "asset_lead_agent",
    "clinical_accountability_agent",
    "study_orchestrator",
    "clinops_agent",
    "safety_agent",
    "regulatory_agent",
    "quality_agent",
    "supply_agent",
    "data_biostats_agent",
    "site_gateway",
    "robot_execution_gateway",
]


def _parse_patient_arrivals() -> list[dict]:
    """Return simulated patient arrivals for this hour."""
    return [
        {
            "patient_id": "PAT-168H-0106",
            "arrival_minute": 50,
            "age": 44,
            "sex": "M",
            "cancer_type": "CML",
            "stage": "IV",
            "ecog": 2,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-0107",
            "arrival_minute": 12,
            "age": 61,
            "sex": "F",
            "cancer_type": "Melanoma",
            "stage": "IIA",
            "ecog": 3,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-0108",
            "arrival_minute": 29,
            "age": 78,
            "sex": "M",
            "cancer_type": "Sarcoma",
            "stage": "HR",
            "ecog": 0,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-0109",
            "arrival_minute": 46,
            "age": 44,
            "sex": "F",
            "cancer_type": "Bladder urothelial",
            "stage": "IIB",
            "ecog": 1,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-0110",
            "arrival_minute": 8,
            "age": 61,
            "sex": "M",
            "cancer_type": "Medulloblastoma",
            "stage": "I",
            "ecog": 2,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-0111",
            "arrival_minute": 25,
            "age": 78,
            "sex": "F",
            "cancer_type": "Gastric adenocarcinoma",
            "stage": "III",
            "ecog": 3,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-0112",
            "arrival_minute": 42,
            "age": 44,
            "sex": "M",
            "cancer_type": "Breast IDC",
            "stage": "IA",
            "ecog": 0,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-0113",
            "arrival_minute": 4,
            "age": 61,
            "sex": "F",
            "cancer_type": "Non-Hodgkin lymphoma",
            "stage": "IIIA",
            "ecog": 1,
            "robot_category": "Rehabilitation Exoskeleton",
        },
        {
            "patient_id": "PAT-168H-0114",
            "arrival_minute": 21,
            "age": 78,
            "sex": "M",
            "cancer_type": "Colorectal adenocarcinoma",
            "stage": "IB",
            "ecog": 2,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-168H-0115",
            "arrival_minute": 38,
            "age": 44,
            "sex": "F",
            "cancer_type": "Mesothelioma",
            "stage": "IIIB",
            "ecog": 3,
            "robot_category": "Environmental Monitoring",
        },
    ]


def _parse_robot_status() -> dict:
    """Return robot fleet status for this hour."""
    return {
        "total_instances": 29,
        "active": 10,
        "standby": 18,
        "maintenance": 1,
        "fleet": ROBOT_FLEET,
    }


def _parse_procedure_completions() -> int:
    """Return count of procedures completed this hour."""
    return 8


def _parse_adverse_events() -> list[dict]:
    """Return adverse events logged this hour."""
    adverse_events: list[dict] = []
    return adverse_events if isinstance(adverse_events, list) else []


def generate_sponsor_directives(hour: int = HOUR) -> dict:
    """Generate 12 sponsor decisions at 5-min intervals for hour 007.

    Parameters
    ----------
    hour : int
        The global hour (0-167) to generate directives for. Defaults to 7.

    Returns
    -------
    dict
        Serializable dictionary of sponsor directives and context.
    """
    patients = _parse_patient_arrivals()
    robot_status = _parse_robot_status()
    procedure_completions = _parse_procedure_completions()
    adverse_events = _parse_adverse_events()

    decisions: list[dict] = [
        {
            "timestamp": "2026-03-23T07:00:00Z",
            "decision_type": "INIT",
            "agent_responsible": "portfolio_agent",
            "confidence_score": 92,
            "action_taken": "sponsor_cycle_init",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Initialize hour-007 monitoring cycle for 10 patients",
        },
        {
            "timestamp": "2026-03-23T07:05:00Z",
            "decision_type": "ENROLL",
            "agent_responsible": "asset_lead_agent",
            "confidence_score": 88,
            "action_taken": "enrollment_gate_check",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Validate enrollment criteria for cohort at 07:05",
        },
        {
            "timestamp": "2026-03-23T07:10:00Z",
            "decision_type": "MONITOR",
            "agent_responsible": "clinical_accountability_agent",
            "confidence_score": 95,
            "action_taken": "telemetry_aggregation",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Vitals and robot telemetry monitoring at 07:10",
        },
        {
            "timestamp": "2026-03-23T07:15:00Z",
            "decision_type": "AUTH",
            "agent_responsible": "study_orchestrator",
            "confidence_score": 91,
            "action_taken": "procedure_authorization_g4",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Authorize procedure with gate G4 (mandatory-human-oversight)",
        },
        {
            "timestamp": "2026-03-23T07:20:00Z",
            "decision_type": "SAFETY_CHECK",
            "agent_responsible": "clinops_agent",
            "confidence_score": 87,
            "action_taken": "safety_interlock_verification",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Verify all robot interlocks and patient positioning",
        },
        {
            "timestamp": "2026-03-23T07:25:00Z",
            "decision_type": "SUPPLY",
            "agent_responsible": "safety_agent",
            "confidence_score": 94,
            "action_taken": "inventory_level_audit",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Confirm consumable levels for hour-007 procedures",
        },
        {
            "timestamp": "2026-03-23T07:30:00Z",
            "decision_type": "DATA_QUALITY",
            "agent_responsible": "regulatory_agent",
            "confidence_score": 90,
            "action_taken": "crf_validation_sweep",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Validate CRF data integrity for hour 007",
        },
        {
            "timestamp": "2026-03-23T07:35:00Z",
            "decision_type": "ESCALATION",
            "agent_responsible": "quality_agent",
            "confidence_score": 86,
            "action_taken": "anomaly_triage",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Anomaly in robot telemetry at 07:35",
        },
        {
            "timestamp": "2026-03-23T07:40:00Z",
            "decision_type": "STATUS",
            "agent_responsible": "supply_agent",
            "confidence_score": 93,
            "action_taken": "pathway_status_sync",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Status aggregation for 10 active pathways",
        },
        {
            "timestamp": "2026-03-23T07:45:00Z",
            "decision_type": "PROCEDURE",
            "agent_responsible": "data_biostats_agent",
            "confidence_score": 89,
            "action_taken": "milestone_verification",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Procedure milestone check at 07:45",
        },
        {
            "timestamp": "2026-03-23T07:50:00Z",
            "decision_type": "DISCHARGE",
            "agent_responsible": "site_gateway",
            "confidence_score": 85,
            "action_taken": "discharge_readiness_eval",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Discharge readiness for hour-007 treatment cycle",
        },
        {
            "timestamp": "2026-03-23T07:55:00Z",
            "decision_type": "REGULATORY",
            "agent_responsible": "robot_execution_gateway",
            "confidence_score": 92,
            "action_taken": "compliance_snapshot",
            "escalation_required": True,
            "safety_gate": "G4",
            "rationale": "GCP and 21 CFR Part 11 compliance snapshot",
        },
    ]

    return {
        "hour": hour,
        "day": DAY,
        "hour_of_day": HOUR_OF_DAY,
        "period": PERIOD,
        "description": "Morning peak operations with full staffing",
        "patient_count": PATIENT_COUNT,
        "patient_arrivals": patients,
        "robot_status": robot_status,
        "procedure_completions": procedure_completions,
        "adverse_events": adverse_events,
        "decisions": decisions,
        "summary": {
            "total_decisions": len(decisions),
            "escalations": sum(1 for d in decisions if d["escalation_required"]),
            "safety_checks": sum(1 for d in decisions if d["decision_type"] == "SAFETY_CHECK"),
            "avg_confidence": round(
                sum(d["confidence_score"] for d in decisions) / len(decisions),
                1,
            ),
        },
    }


if __name__ == "__main__":
    result = generate_sponsor_directives()
    print(json.dumps(result, indent=2))
