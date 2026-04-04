"""Sponsor activity script for Hour 09 (Peak morning activity with maximum procedure throughput)."""

from __future__ import annotations

import json
from pathlib import Path

HOUR = 9
NEW_TRIAL_DIR = Path(__file__).resolve().parents[4] / "new-trial" / "hour-09"
PATIENT_COUNT = 15
PERIOD = "peak_morning"

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


def _load_trial_data() -> dict:
    """Load and parse new-trial hour data if available."""
    data: dict = {"patients": [], "robot_logs": [], "simulation": ""}
    patient_file = NEW_TRIAL_DIR / "hour_09_patient_records.md"
    robot_file = NEW_TRIAL_DIR / "hour_09_robot_logs.md"
    sim_file = NEW_TRIAL_DIR / "hour_09_simulation.md"
    if patient_file.exists():
        data["patients"] = patient_file.read_text(
            encoding="utf-8",
        ).splitlines()
    if robot_file.exists():
        data["robot_logs"] = robot_file.read_text(
            encoding="utf-8",
        ).splitlines()
    if sim_file.exists():
        data["simulation"] = sim_file.read_text(encoding="utf-8")
    return data


def _parse_patient_arrivals() -> list[dict]:
    """Return simulated patient arrivals for this hour."""
    return [
        {
            "patient_id": "PAT-ODMND-0053",
            "arrival_minute": 2,
            "age": 51,
            "sex": "F",
            "cancer_type": "Pediatric ALL",
            "stage": "HR",
            "ecog": 1,
            "robot_category": "Social Companion",
        },
        {
            "patient_id": "PAT-ODMND-0054",
            "arrival_minute": 5,
            "age": 58,
            "sex": "M",
            "cancer_type": "Pancreatic ductal",
            "stage": "IIB",
            "ecog": 2,
            "robot_category": "Collaborative (Cobot)",
        },
        {
            "patient_id": "PAT-ODMND-0055",
            "arrival_minute": 9,
            "age": 65,
            "sex": "F",
            "cancer_type": "Glioblastoma",
            "stage": "IV",
            "ecog": 3,
            "robot_category": "RT Positioning",
        },
        {
            "patient_id": "PAT-ODMND-0056",
            "arrival_minute": 13,
            "age": 72,
            "sex": "M",
            "cancer_type": "Colorectal adenocarcinoma",
            "stage": "III",
            "ecog": 0,
            "robot_category": "Transport/Logistics",
        },
        {
            "patient_id": "PAT-ODMND-0057",
            "arrival_minute": 16,
            "age": 79,
            "sex": "F",
            "cancer_type": "Renal cell carcinoma",
            "stage": "II",
            "ecog": 1,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-ODMND-0058",
            "arrival_minute": 20,
            "age": 36,
            "sex": "M",
            "cancer_type": "Soft-tissue sarcoma",
            "stage": "IIA",
            "ecog": 2,
            "robot_category": "Rehabilitation Exoskeleton",
        },
        {
            "patient_id": "PAT-ODMND-0059",
            "arrival_minute": 24,
            "age": 43,
            "sex": "F",
            "cancer_type": "Mediastinal tumor",
            "stage": "IIIB",
            "ecog": 3,
            "robot_category": "Surgical (da Vinci Xi)",
        },
        {
            "patient_id": "PAT-ODMND-0060",
            "arrival_minute": 27,
            "age": 50,
            "sex": "M",
            "cancer_type": "Cervical SCC",
            "stage": "IB2",
            "ecog": 0,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-ODMND-0061",
            "arrival_minute": 31,
            "age": 57,
            "sex": "F",
            "cancer_type": "Melanoma",
            "stage": "IIIC",
            "ecog": 1,
            "robot_category": "Environmental Monitoring",
        },
        {
            "patient_id": "PAT-ODMND-0062",
            "arrival_minute": 35,
            "age": 64,
            "sex": "M",
            "cancer_type": "Ovarian serous",
            "stage": "IIIA",
            "ecog": 2,
            "robot_category": "Collaborative (Cobot)",
        },
        {
            "patient_id": "PAT-ODMND-0063",
            "arrival_minute": 38,
            "age": 71,
            "sex": "F",
            "cancer_type": "Esophageal SCC",
            "stage": "III",
            "ecog": 3,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-ODMND-0064",
            "arrival_minute": 42,
            "age": 78,
            "sex": "M",
            "cancer_type": "Bladder urothelial",
            "stage": "II",
            "ecog": 0,
            "robot_category": "Needle Placement",
        },
        {
            "patient_id": "PAT-ODMND-0065",
            "arrival_minute": 46,
            "age": 35,
            "sex": "F",
            "cancer_type": "NSCLC adenocarcinoma",
            "stage": "IIIA",
            "ecog": 1,
            "robot_category": "RT Motion-Tracking",
        },
        {
            "patient_id": "PAT-ODMND-0066",
            "arrival_minute": 49,
            "age": 42,
            "sex": "M",
            "cancer_type": "HCC",
            "stage": "II",
            "ecog": 2,
            "robot_category": "Imaging Assistance",
        },
        {
            "patient_id": "PAT-ODMND-0067",
            "arrival_minute": 53,
            "age": 49,
            "sex": "F",
            "cancer_type": "Breast IDC",
            "stage": "IIA",
            "ecog": 3,
            "robot_category": "Surgical (da Vinci Xi)",
        },
    ]


def _parse_robot_status() -> dict:
    """Return robot fleet status for this hour."""
    return {
        "total_instances": 29,
        "active": 22,
        "standby": 5,
        "maintenance": 2,
        "fleet": ROBOT_FLEET,
    }


def _parse_procedure_completions() -> int:
    """Return count of procedures completed this hour."""
    return 14


def _parse_adverse_events() -> list[dict]:
    """Return adverse events logged this hour."""
    adverse_events = [
        {
            "event_id": "AE-09-001",
            "patient_id": "PAT-ODMND-0053",
            "time": "09:22",
            "severity": "Grade 1",
            "description": "Transient mild nausea during positioning",
            "robot_involved": False,
            "resolved": True,
            "resolution_time_min": 8,
        }
    ]
    return adverse_events if isinstance(adverse_events, list) else []


def generate_sponsor_directives(hour: int = HOUR) -> dict:
    """Generate 12 sponsor decisions at 5-min intervals for hour 09.

    Parameters
    ----------
    hour : int
        The hour (0-23) to generate directives for. Defaults to 9.

    Returns
    -------
    dict
        Serializable dictionary of sponsor directives and context.
    """
    trial_data = _load_trial_data()
    patients = _parse_patient_arrivals()
    robot_status = _parse_robot_status()
    procedure_completions = _parse_procedure_completions()
    adverse_events = _parse_adverse_events()

    decisions: list[dict] = [
        {
            "timestamp": "2026-03-23T09:00:00Z",
            "decision_type": "INIT",
            "agent_responsible": "portfolio_agent",
            "confidence_score": 90,
            "action_taken": "sponsor_cycle_init",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Initialize hour-09 monitoring cycle for 15 patients",
        },
        {
            "timestamp": "2026-03-23T09:05:00Z",
            "decision_type": "ENROLL",
            "agent_responsible": "asset_lead_agent",
            "confidence_score": 91,
            "action_taken": "enrollment_gate_check",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Validate enrollment criteria for cohort at 09:05",
        },
        {
            "timestamp": "2026-03-23T09:10:00Z",
            "decision_type": "MONITOR",
            "agent_responsible": "clinical_accountability_agent",
            "confidence_score": 92,
            "action_taken": "telemetry_aggregation",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Vitals and robot telemetry monitoring at 09:10",
        },
        {
            "timestamp": "2026-03-23T09:15:00Z",
            "decision_type": "AUTH",
            "agent_responsible": "study_orchestrator",
            "confidence_score": 93,
            "action_taken": "procedure_authorization_g4",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Authorize procedure with gate G4 (mandatory-human-oversight)",
        },
        {
            "timestamp": "2026-03-23T09:20:00Z",
            "decision_type": "SAFETY_CHECK",
            "agent_responsible": "clinops_agent",
            "confidence_score": 94,
            "action_taken": "safety_interlock_verification",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Verify all robot interlocks and patient positioning",
        },
        {
            "timestamp": "2026-03-23T09:25:00Z",
            "decision_type": "SUPPLY",
            "agent_responsible": "safety_agent",
            "confidence_score": 90,
            "action_taken": "inventory_level_audit",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Confirm consumable levels for hour-09 procedures",
        },
        {
            "timestamp": "2026-03-23T09:30:00Z",
            "decision_type": "DATA_QUALITY",
            "agent_responsible": "regulatory_agent",
            "confidence_score": 91,
            "action_taken": "crf_validation_sweep",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Validate CRF data integrity for hour 09",
        },
        {
            "timestamp": "2026-03-23T09:35:00Z",
            "decision_type": "ESCALATION",
            "agent_responsible": "quality_agent",
            "confidence_score": 92,
            "action_taken": "anomaly_triage",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "Anomaly in robot telemetry at 09:35",
        },
        {
            "timestamp": "2026-03-23T09:40:00Z",
            "decision_type": "STATUS",
            "agent_responsible": "supply_agent",
            "confidence_score": 93,
            "action_taken": "pathway_status_sync",
            "escalation_required": False,
            "safety_gate": "G1",
            "rationale": "Status aggregation for 15 active pathways",
        },
        {
            "timestamp": "2026-03-23T09:45:00Z",
            "decision_type": "PROCEDURE",
            "agent_responsible": "data_biostats_agent",
            "confidence_score": 94,
            "action_taken": "milestone_verification",
            "escalation_required": False,
            "safety_gate": "G2",
            "rationale": "Procedure milestone check at 09:45",
        },
        {
            "timestamp": "2026-03-23T09:50:00Z",
            "decision_type": "DISCHARGE",
            "agent_responsible": "site_gateway",
            "confidence_score": 90,
            "action_taken": "discharge_readiness_eval",
            "escalation_required": False,
            "safety_gate": "G3",
            "rationale": "Discharge readiness for hour-09 treatment cycle",
        },
        {
            "timestamp": "2026-03-23T09:55:00Z",
            "decision_type": "REGULATORY",
            "agent_responsible": "robot_execution_gateway",
            "confidence_score": 91,
            "action_taken": "compliance_snapshot",
            "escalation_required": False,
            "safety_gate": "G4",
            "rationale": "GCP and 21 CFR Part 11 compliance snapshot",
        },
    ]

    return {
        "hour": hour,
        "period": PERIOD,
        "description": "Peak morning activity with maximum procedure throughput",
        "patient_count": PATIENT_COUNT,
        "patient_arrivals": patients,
        "robot_status": robot_status,
        "procedure_completions": procedure_completions,
        "adverse_events": adverse_events,
        "decisions": decisions,
        "trial_data_loaded": bool(trial_data["simulation"]),
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
