"""Sponsor activity script for Hour 033 (Day 2, 09:00 - Peak operational period with maximum throughput)."""

from __future__ import annotations

import json
from pathlib import Path

HOUR = 33
DAY = 2
HOUR_OF_DAY = 9
PATIENT_COUNT = 18
PERIOD = "peak_operations"

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
            "patient_id": "PAT-168H-0496",
            "arrival_minute": 35,
            "age": 34,
            "sex": "M",
            "cancer_type": "Renal cell carcinoma",
            "stage": "IIB",
            "ecog": 2,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-0497",
            "arrival_minute": 52,
            "age": 51,
            "sex": "F",
            "cancer_type": "Neuroblastoma",
            "stage": "I",
            "ecog": 3,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-0498",
            "arrival_minute": 14,
            "age": 68,
            "sex": "M",
            "cancer_type": "Endometrial",
            "stage": "III",
            "ecog": 0,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-0499",
            "arrival_minute": 31,
            "age": 34,
            "sex": "F",
            "cancer_type": "HCC",
            "stage": "IA",
            "ecog": 1,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-0500",
            "arrival_minute": 48,
            "age": 51,
            "sex": "M",
            "cancer_type": "Multiple myeloma",
            "stage": "IIIA",
            "ecog": 2,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-0501",
            "arrival_minute": 10,
            "age": 68,
            "sex": "F",
            "cancer_type": "Glioblastoma",
            "stage": "IB",
            "ecog": 3,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-0502",
            "arrival_minute": 27,
            "age": 34,
            "sex": "M",
            "cancer_type": "Cholangiocarcinoma",
            "stage": "IIIB",
            "ecog": 0,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-0503",
            "arrival_minute": 44,
            "age": 51,
            "sex": "F",
            "cancer_type": "Pancreatic ductal",
            "stage": "II",
            "ecog": 1,
            "robot_category": "Rehabilitation Exoskeleton",
        },
        {
            "patient_id": "PAT-168H-0504",
            "arrival_minute": 6,
            "age": 68,
            "sex": "M",
            "cancer_type": "Wilms tumor",
            "stage": "IV",
            "ecog": 2,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-168H-0505",
            "arrival_minute": 23,
            "age": 34,
            "sex": "F",
            "cancer_type": "Thyroid papillary",
            "stage": "IIA",
            "ecog": 3,
            "robot_category": "Environmental Monitoring",
        },
        {
            "patient_id": "PAT-168H-0506",
            "arrival_minute": 40,
            "age": 51,
            "sex": "M",
            "cancer_type": "NSCLC adenocarcinoma",
            "stage": "HR",
            "ecog": 0,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-0507",
            "arrival_minute": 2,
            "age": 68,
            "sex": "F",
            "cancer_type": "Cervical SCC",
            "stage": "IIB",
            "ecog": 1,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-0508",
            "arrival_minute": 19,
            "age": 34,
            "sex": "M",
            "cancer_type": "Pediatric ALL",
            "stage": "I",
            "ecog": 2,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-0509",
            "arrival_minute": 36,
            "age": 51,
            "sex": "F",
            "cancer_type": "AML",
            "stage": "III",
            "ecog": 3,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-0510",
            "arrival_minute": 53,
            "age": 68,
            "sex": "M",
            "cancer_type": "Ovarian serous",
            "stage": "IA",
            "ecog": 0,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-0511",
            "arrival_minute": 15,
            "age": 34,
            "sex": "F",
            "cancer_type": "Testicular seminoma",
            "stage": "IIIA",
            "ecog": 1,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-0512",
            "arrival_minute": 32,
            "age": 51,
            "sex": "M",
            "cancer_type": "Esophageal SCC",
            "stage": "IB",
            "ecog": 2,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-0513",
            "arrival_minute": 49,
            "age": 68,
            "sex": "F",
            "cancer_type": "Retinoblastoma",
            "stage": "IIIB",
            "ecog": 3,
            "robot_category": "Rehabilitation Exoskeleton",
        },
    ]


def _parse_robot_status() -> dict:
    """Return robot fleet status for this hour."""
    return {
        "total_instances": 29,
        "active": 18,
        "standby": 10,
        "maintenance": 1,
        "fleet": ROBOT_FLEET,
    }


def _parse_procedure_completions() -> int:
    """Return count of procedures completed this hour."""
    return 16


def _parse_adverse_events() -> list[dict]:
    """Return adverse events logged this hour."""
    adverse_events: list[dict] = []
    return adverse_events if isinstance(adverse_events, list) else []


def generate_sponsor_directives(hour: int = HOUR) -> dict:
    """Generate 12 sponsor decisions at 5-min intervals for hour 033.

    Parameters
    ----------
    hour : int
        The global hour (0-167) to generate directives for. Defaults to 33.

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
            "timestamp": "2026-03-24T09:00:00Z",
            "decision_type": "INIT",
            "agent_responsible": "portfolio_agent",
            "confidence_score": 85,
            "action_taken": "sponsor_cycle_init",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Initialize hour-033 monitoring cycle for 18 patients",
        },
        {
            "timestamp": "2026-03-24T09:05:00Z",
            "decision_type": "ENROLL",
            "agent_responsible": "asset_lead_agent",
            "confidence_score": 92,
            "action_taken": "enrollment_gate_check",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Validate enrollment criteria for cohort at 09:05",
        },
        {
            "timestamp": "2026-03-24T09:10:00Z",
            "decision_type": "MONITOR",
            "agent_responsible": "clinical_accountability_agent",
            "confidence_score": 88,
            "action_taken": "telemetry_aggregation",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Vitals and robot telemetry monitoring at 09:10",
        },
        {
            "timestamp": "2026-03-24T09:15:00Z",
            "decision_type": "AUTH",
            "agent_responsible": "study_orchestrator",
            "confidence_score": 95,
            "action_taken": "procedure_authorization_g4",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Authorize procedure with gate G4 (mandatory-human-oversight)",
        },
        {
            "timestamp": "2026-03-24T09:20:00Z",
            "decision_type": "SAFETY_CHECK",
            "agent_responsible": "clinops_agent",
            "confidence_score": 91,
            "action_taken": "safety_interlock_verification",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Verify all robot interlocks and patient positioning",
        },
        {
            "timestamp": "2026-03-24T09:25:00Z",
            "decision_type": "SUPPLY",
            "agent_responsible": "safety_agent",
            "confidence_score": 87,
            "action_taken": "inventory_level_audit",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Confirm consumable levels for hour-033 procedures",
        },
        {
            "timestamp": "2026-03-24T09:30:00Z",
            "decision_type": "DATA_QUALITY",
            "agent_responsible": "regulatory_agent",
            "confidence_score": 94,
            "action_taken": "crf_validation_sweep",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Validate CRF data integrity for hour 033",
        },
        {
            "timestamp": "2026-03-24T09:35:00Z",
            "decision_type": "ESCALATION",
            "agent_responsible": "quality_agent",
            "confidence_score": 90,
            "action_taken": "anomaly_triage",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Anomaly in robot telemetry at 09:35",
        },
        {
            "timestamp": "2026-03-24T09:40:00Z",
            "decision_type": "STATUS",
            "agent_responsible": "supply_agent",
            "confidence_score": 86,
            "action_taken": "pathway_status_sync",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Status aggregation for 18 active pathways",
        },
        {
            "timestamp": "2026-03-24T09:45:00Z",
            "decision_type": "PROCEDURE",
            "agent_responsible": "data_biostats_agent",
            "confidence_score": 93,
            "action_taken": "milestone_verification",
            "escalation_required": True,
            "safety_gate": "G2",
            "rationale": "Procedure milestone check at 09:45",
        },
        {
            "timestamp": "2026-03-24T09:50:00Z",
            "decision_type": "DISCHARGE",
            "agent_responsible": "site_gateway",
            "confidence_score": 89,
            "action_taken": "discharge_readiness_eval",
            "escalation_required": True,
            "safety_gate": "G3",
            "rationale": "Discharge readiness for hour-033 treatment cycle",
        },
        {
            "timestamp": "2026-03-24T09:55:00Z",
            "decision_type": "REGULATORY",
            "agent_responsible": "robot_execution_gateway",
            "confidence_score": 85,
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
        "description": "Peak operational period with maximum throughput",
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
