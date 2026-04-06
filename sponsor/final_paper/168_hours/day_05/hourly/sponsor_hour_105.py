"""Sponsor activity script for Hour 105 (Day 5, 09:00 - Peak operational period with maximum throughput)."""

from __future__ import annotations

import json
from pathlib import Path

HOUR = 105
DAY = 5
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
            "patient_id": "PAT-168H-1576",
            "arrival_minute": 40,
            "age": 73,
            "sex": "M",
            "cancer_type": "Renal cell carcinoma",
            "stage": "HR",
            "ecog": 2,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-1577",
            "arrival_minute": 2,
            "age": 39,
            "sex": "F",
            "cancer_type": "Neuroblastoma",
            "stage": "IIB",
            "ecog": 3,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-1578",
            "arrival_minute": 19,
            "age": 56,
            "sex": "M",
            "cancer_type": "Endometrial",
            "stage": "I",
            "ecog": 0,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-1579",
            "arrival_minute": 36,
            "age": 73,
            "sex": "F",
            "cancer_type": "HCC",
            "stage": "III",
            "ecog": 1,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-1580",
            "arrival_minute": 53,
            "age": 39,
            "sex": "M",
            "cancer_type": "Multiple myeloma",
            "stage": "IA",
            "ecog": 2,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-1581",
            "arrival_minute": 15,
            "age": 56,
            "sex": "F",
            "cancer_type": "Glioblastoma",
            "stage": "IIIA",
            "ecog": 3,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-1582",
            "arrival_minute": 32,
            "age": 73,
            "sex": "M",
            "cancer_type": "Cholangiocarcinoma",
            "stage": "IB",
            "ecog": 0,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-1583",
            "arrival_minute": 49,
            "age": 39,
            "sex": "F",
            "cancer_type": "Pancreatic ductal",
            "stage": "IIIB",
            "ecog": 1,
            "robot_category": "Rehabilitation Exoskeleton",
        },
        {
            "patient_id": "PAT-168H-1584",
            "arrival_minute": 11,
            "age": 56,
            "sex": "M",
            "cancer_type": "Wilms tumor",
            "stage": "II",
            "ecog": 2,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-168H-1585",
            "arrival_minute": 28,
            "age": 73,
            "sex": "F",
            "cancer_type": "Thyroid papillary",
            "stage": "IV",
            "ecog": 3,
            "robot_category": "Environmental Monitoring",
        },
        {
            "patient_id": "PAT-168H-1586",
            "arrival_minute": 45,
            "age": 39,
            "sex": "M",
            "cancer_type": "NSCLC adenocarcinoma",
            "stage": "IIA",
            "ecog": 0,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-168H-1587",
            "arrival_minute": 7,
            "age": 56,
            "sex": "F",
            "cancer_type": "Cervical SCC",
            "stage": "HR",
            "ecog": 1,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-168H-1588",
            "arrival_minute": 24,
            "age": 73,
            "sex": "M",
            "cancer_type": "Pediatric ALL",
            "stage": "IIB",
            "ecog": 2,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-168H-1589",
            "arrival_minute": 41,
            "age": 39,
            "sex": "F",
            "cancer_type": "AML",
            "stage": "I",
            "ecog": 3,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-168H-1590",
            "arrival_minute": 3,
            "age": 56,
            "sex": "M",
            "cancer_type": "Ovarian serous",
            "stage": "III",
            "ecog": 0,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-168H-1591",
            "arrival_minute": 20,
            "age": 73,
            "sex": "F",
            "cancer_type": "Testicular seminoma",
            "stage": "IA",
            "ecog": 1,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-168H-1592",
            "arrival_minute": 37,
            "age": 39,
            "sex": "M",
            "cancer_type": "Esophageal SCC",
            "stage": "IIIA",
            "ecog": 2,
            "robot_category": "Cobots (Franka Emika)",
        },
        {
            "patient_id": "PAT-168H-1593",
            "arrival_minute": 54,
            "age": 56,
            "sex": "F",
            "cancer_type": "Retinoblastoma",
            "stage": "IB",
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
    """Generate 12 sponsor decisions at 5-min intervals for hour 105.

    Parameters
    ----------
    hour : int
        The global hour (0-167) to generate directives for. Defaults to 105.

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
            "timestamp": "2026-03-27T09:00:00Z",
            "decision_type": "INIT",
            "agent_responsible": "portfolio_agent",
            "confidence_score": 91,
            "action_taken": "sponsor_cycle_init",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Initialize hour-105 monitoring cycle for 18 patients",
        },
        {
            "timestamp": "2026-03-27T09:05:00Z",
            "decision_type": "ENROLL",
            "agent_responsible": "asset_lead_agent",
            "confidence_score": 87,
            "action_taken": "enrollment_gate_check",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Validate enrollment criteria for cohort at 09:05",
        },
        {
            "timestamp": "2026-03-27T09:10:00Z",
            "decision_type": "MONITOR",
            "agent_responsible": "clinical_accountability_agent",
            "confidence_score": 94,
            "action_taken": "telemetry_aggregation",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Vitals and robot telemetry monitoring at 09:10",
        },
        {
            "timestamp": "2026-03-27T09:15:00Z",
            "decision_type": "AUTH",
            "agent_responsible": "study_orchestrator",
            "confidence_score": 90,
            "action_taken": "procedure_authorization_g4",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Authorize procedure with gate G4 (mandatory-human-oversight)",
        },
        {
            "timestamp": "2026-03-27T09:20:00Z",
            "decision_type": "SAFETY_CHECK",
            "agent_responsible": "clinops_agent",
            "confidence_score": 86,
            "action_taken": "safety_interlock_verification",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Verify all robot interlocks and patient positioning",
        },
        {
            "timestamp": "2026-03-27T09:25:00Z",
            "decision_type": "SUPPLY",
            "agent_responsible": "safety_agent",
            "confidence_score": 93,
            "action_taken": "inventory_level_audit",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Confirm consumable levels for hour-105 procedures",
        },
        {
            "timestamp": "2026-03-27T09:30:00Z",
            "decision_type": "DATA_QUALITY",
            "agent_responsible": "regulatory_agent",
            "confidence_score": 89,
            "action_taken": "crf_validation_sweep",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Validate CRF data integrity for hour 105",
        },
        {
            "timestamp": "2026-03-27T09:35:00Z",
            "decision_type": "ESCALATION",
            "agent_responsible": "quality_agent",
            "confidence_score": 85,
            "action_taken": "anomaly_triage",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Anomaly in robot telemetry at 09:35",
        },
        {
            "timestamp": "2026-03-27T09:40:00Z",
            "decision_type": "STATUS",
            "agent_responsible": "supply_agent",
            "confidence_score": 92,
            "action_taken": "pathway_status_sync",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Status aggregation for 18 active pathways",
        },
        {
            "timestamp": "2026-03-27T09:45:00Z",
            "decision_type": "PROCEDURE",
            "agent_responsible": "data_biostats_agent",
            "confidence_score": 88,
            "action_taken": "milestone_verification",
            "escalation_required": True,
            "safety_gate": "G2",
            "rationale": "Procedure milestone check at 09:45",
        },
        {
            "timestamp": "2026-03-27T09:50:00Z",
            "decision_type": "DISCHARGE",
            "agent_responsible": "site_gateway",
            "confidence_score": 95,
            "action_taken": "discharge_readiness_eval",
            "escalation_required": True,
            "safety_gate": "G3",
            "rationale": "Discharge readiness for hour-105 treatment cycle",
        },
        {
            "timestamp": "2026-03-27T09:55:00Z",
            "decision_type": "REGULATORY",
            "agent_responsible": "robot_execution_gateway",
            "confidence_score": 91,
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
