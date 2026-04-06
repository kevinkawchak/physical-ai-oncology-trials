"""Sponsor activity script for Hour 081 (Day 4, 09:00 - Peak operational period with maximum throughput)."""

from __future__ import annotations

import json
from pathlib import Path

HOUR = 81
DAY = 4
HOUR_OF_DAY = 9
PATIENT_COUNT = 21
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
            "patient_id": "PAT-168H-1216",
            "arrival_minute": 20,
            "age": 43,
            "sex": "M",
            "cancer_type": "Renal cell carcinoma",
            "stage": "IA",
            "ecog": 2,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-1217",
            "arrival_minute": 37,
            "age": 60,
            "sex": "F",
            "cancer_type": "Neuroblastoma",
            "stage": "IIIA",
            "ecog": 3,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-1218",
            "arrival_minute": 54,
            "age": 77,
            "sex": "M",
            "cancer_type": "Endometrial",
            "stage": "IB",
            "ecog": 0,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-1219",
            "arrival_minute": 16,
            "age": 43,
            "sex": "F",
            "cancer_type": "HCC",
            "stage": "IIIB",
            "ecog": 1,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-1220",
            "arrival_minute": 33,
            "age": 60,
            "sex": "M",
            "cancer_type": "Multiple myeloma",
            "stage": "II",
            "ecog": 2,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-1221",
            "arrival_minute": 50,
            "age": 77,
            "sex": "F",
            "cancer_type": "Glioblastoma",
            "stage": "IV",
            "ecog": 3,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-1222",
            "arrival_minute": 12,
            "age": 43,
            "sex": "M",
            "cancer_type": "Cholangiocarcinoma",
            "stage": "IIA",
            "ecog": 0,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-1223",
            "arrival_minute": 29,
            "age": 60,
            "sex": "F",
            "cancer_type": "Pancreatic ductal",
            "stage": "HR",
            "ecog": 1,
            "robot_category": "Rehabilitation Exoskeleton",
        },
        {
            "patient_id": "PAT-168H-1224",
            "arrival_minute": 46,
            "age": 77,
            "sex": "M",
            "cancer_type": "Wilms tumor",
            "stage": "IIB",
            "ecog": 2,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-168H-1225",
            "arrival_minute": 8,
            "age": 43,
            "sex": "F",
            "cancer_type": "Thyroid papillary",
            "stage": "I",
            "ecog": 3,
            "robot_category": "Environmental Monitoring",
        },
        {
            "patient_id": "PAT-168H-1226",
            "arrival_minute": 25,
            "age": 60,
            "sex": "M",
            "cancer_type": "NSCLC adenocarcinoma",
            "stage": "III",
            "ecog": 0,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-1227",
            "arrival_minute": 42,
            "age": 77,
            "sex": "F",
            "cancer_type": "Cervical SCC",
            "stage": "IA",
            "ecog": 1,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-1228",
            "arrival_minute": 4,
            "age": 43,
            "sex": "M",
            "cancer_type": "Pediatric ALL",
            "stage": "IIIA",
            "ecog": 2,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-1229",
            "arrival_minute": 21,
            "age": 60,
            "sex": "F",
            "cancer_type": "AML",
            "stage": "IB",
            "ecog": 3,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-1230",
            "arrival_minute": 38,
            "age": 77,
            "sex": "M",
            "cancer_type": "Ovarian serous",
            "stage": "IIIB",
            "ecog": 0,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-1231",
            "arrival_minute": 55,
            "age": 43,
            "sex": "F",
            "cancer_type": "Testicular seminoma",
            "stage": "II",
            "ecog": 1,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-1232",
            "arrival_minute": 17,
            "age": 60,
            "sex": "M",
            "cancer_type": "Esophageal SCC",
            "stage": "IV",
            "ecog": 2,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-1233",
            "arrival_minute": 34,
            "age": 77,
            "sex": "F",
            "cancer_type": "Retinoblastoma",
            "stage": "IIA",
            "ecog": 3,
            "robot_category": "Rehabilitation Exoskeleton",
        },
        {
            "patient_id": "PAT-168H-1234",
            "arrival_minute": 51,
            "age": 43,
            "sex": "M",
            "cancer_type": "Head and neck SCC",
            "stage": "HR",
            "ecog": 0,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-168H-1235",
            "arrival_minute": 13,
            "age": 60,
            "sex": "F",
            "cancer_type": "Prostate adenocarcinoma",
            "stage": "IIB",
            "ecog": 1,
            "robot_category": "Environmental Monitoring",
        },
        {
            "patient_id": "PAT-168H-1236",
            "arrival_minute": 30,
            "age": 77,
            "sex": "M",
            "cancer_type": "CML",
            "stage": "I",
            "ecog": 2,
            "robot_category": "RT Positioning",
        },
    ]


def _parse_robot_status() -> dict:
    """Return robot fleet status for this hour."""
    return {
        "total_instances": 29,
        "active": 21,
        "standby": 7,
        "maintenance": 1,
        "fleet": ROBOT_FLEET,
    }


def _parse_procedure_completions() -> int:
    """Return count of procedures completed this hour."""
    return 19


def _parse_adverse_events() -> list[dict]:
    """Return adverse events logged this hour."""
    adverse_events: list[dict] = []
    return adverse_events if isinstance(adverse_events, list) else []


def generate_sponsor_directives(hour: int = HOUR) -> dict:
    """Generate 12 sponsor decisions at 5-min intervals for hour 081.

    Parameters
    ----------
    hour : int
        The global hour (0-167) to generate directives for. Defaults to 81.

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
            "timestamp": "2026-03-26T09:00:00Z",
            "decision_type": "INIT",
            "agent_responsible": "portfolio_agent",
            "confidence_score": 89,
            "action_taken": "sponsor_cycle_init",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Initialize hour-081 monitoring cycle for 21 patients",
        },
        {
            "timestamp": "2026-03-26T09:05:00Z",
            "decision_type": "ENROLL",
            "agent_responsible": "asset_lead_agent",
            "confidence_score": 85,
            "action_taken": "enrollment_gate_check",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Validate enrollment criteria for cohort at 09:05",
        },
        {
            "timestamp": "2026-03-26T09:10:00Z",
            "decision_type": "MONITOR",
            "agent_responsible": "clinical_accountability_agent",
            "confidence_score": 92,
            "action_taken": "telemetry_aggregation",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Vitals and robot telemetry monitoring at 09:10",
        },
        {
            "timestamp": "2026-03-26T09:15:00Z",
            "decision_type": "AUTH",
            "agent_responsible": "study_orchestrator",
            "confidence_score": 88,
            "action_taken": "procedure_authorization_g4",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Authorize procedure with gate G4 (mandatory-human-oversight)",
        },
        {
            "timestamp": "2026-03-26T09:20:00Z",
            "decision_type": "SAFETY_CHECK",
            "agent_responsible": "clinops_agent",
            "confidence_score": 95,
            "action_taken": "safety_interlock_verification",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Verify all robot interlocks and patient positioning",
        },
        {
            "timestamp": "2026-03-26T09:25:00Z",
            "decision_type": "SUPPLY",
            "agent_responsible": "safety_agent",
            "confidence_score": 91,
            "action_taken": "inventory_level_audit",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Confirm consumable levels for hour-081 procedures",
        },
        {
            "timestamp": "2026-03-26T09:30:00Z",
            "decision_type": "DATA_QUALITY",
            "agent_responsible": "regulatory_agent",
            "confidence_score": 87,
            "action_taken": "crf_validation_sweep",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Validate CRF data integrity for hour 081",
        },
        {
            "timestamp": "2026-03-26T09:35:00Z",
            "decision_type": "ESCALATION",
            "agent_responsible": "quality_agent",
            "confidence_score": 94,
            "action_taken": "anomaly_triage",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Anomaly in robot telemetry at 09:35",
        },
        {
            "timestamp": "2026-03-26T09:40:00Z",
            "decision_type": "STATUS",
            "agent_responsible": "supply_agent",
            "confidence_score": 90,
            "action_taken": "pathway_status_sync",
            "escalation_required": True,
            "safety_gate": "G1",
            "rationale": "Status aggregation for 21 active pathways",
        },
        {
            "timestamp": "2026-03-26T09:45:00Z",
            "decision_type": "PROCEDURE",
            "agent_responsible": "data_biostats_agent",
            "confidence_score": 86,
            "action_taken": "milestone_verification",
            "escalation_required": True,
            "safety_gate": "G2",
            "rationale": "Procedure milestone check at 09:45",
        },
        {
            "timestamp": "2026-03-26T09:50:00Z",
            "decision_type": "DISCHARGE",
            "agent_responsible": "site_gateway",
            "confidence_score": 93,
            "action_taken": "discharge_readiness_eval",
            "escalation_required": True,
            "safety_gate": "G3",
            "rationale": "Discharge readiness for hour-081 treatment cycle",
        },
        {
            "timestamp": "2026-03-26T09:55:00Z",
            "decision_type": "REGULATORY",
            "agent_responsible": "robot_execution_gateway",
            "confidence_score": 89,
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
