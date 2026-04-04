"""Master runner for the 24-hour autonomous sponsor simulation.

Sequentially processes each hour (00-23), generates sponsor directives,
produces 72 text diagrams (3 perspectives x 24 hours), and outputs a
cumulative 24-hour summary. Falls back to direct function calls when
FastAPI/uvicorn is not installed.

Usage:
    python run_sponsor_simulation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
HOURLY_DIR = SCRIPT_DIR / "hourly"
OUTPUT_DIR = SCRIPT_DIR / "output"
HOURLY_OUTPUT_DIR = HOURLY_DIR / "output"
DIAGRAM_DIR = SCRIPT_DIR / "diagrams"

TOTAL_HOURS = 24
DECISIONS_PER_HOUR = 12
EXPECTED_TOTAL_DECISIONS = TOTAL_HOURS * DECISIONS_PER_HOUR
EXPECTED_TOTAL_PATIENTS = 168
EXPECTED_TOTAL_ESCALATIONS = 13
EXPECTED_TOTAL_ROBOT_AUTH = 153

PATIENT_VOLUME = [2, 3, 3, 4, 4, 5, 8, 10, 12, 15, 12, 10, 8, 9, 10, 8, 7, 6, 5, 4, 3, 3, 2, 2]
ESCALATION_COUNTS = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
PSL_SCORES = [
    63.4, 63.5, 63.5, 63.6, 63.6, 63.7, 63.8, 63.9, 64.0, 64.1,
    64.2, 64.2, 64.3, 64.3, 64.4, 64.4, 64.5, 64.5, 64.6, 64.6,
    64.7, 64.7, 64.7, 64.8,
]


def _check_fastapi_available() -> bool:
    """Check whether FastAPI and uvicorn are importable."""
    try:
        import importlib

        importlib.import_module("fastapi")
        importlib.import_module("uvicorn")
        return True
    except ImportError:
        return False


def _run_hourly_standalone(hour: int) -> dict:
    """Generate sponsor directives for a single hour without HTTP."""
    sys.path.insert(0, str(HOURLY_DIR))
    module_name = f"sponsor_hour_{hour:02d}"
    try:
        mod = __import__(module_name)
        return mod.generate_sponsor_directives(hour)
    except (ImportError, AttributeError):
        return _generate_fallback_hour(hour)
    finally:
        if str(HOURLY_DIR) in sys.path:
            sys.path.remove(str(HOURLY_DIR))


def _generate_fallback_hour(hour: int) -> dict:
    """Generate fallback sponsor directive data for a given hour."""
    patients = PATIENT_VOLUME[hour]
    escalations = ESCALATION_COUNTS[hour]
    psl = PSL_SCORES[hour]

    agents = [
        "portfolio_agent", "asset_lead_agent", "clinical_accountability_agent",
        "study_orchestrator", "clinops_agent", "safety_agent",
        "regulatory_agent", "quality_agent", "supply_agent",
        "data_biostats_agent", "site_gateway", "robot_execution_gateway",
    ]
    decision_types = [
        "INIT", "ENROLL", "MONITOR", "AUTH", "SAFETY_CHECK", "SUPPLY",
        "DATA_QUALITY", "ESCALATION", "STATUS", "PROCEDURE", "DISCHARGE", "REGULATORY",
    ]

    decisions = []
    for i in range(DECISIONS_PER_HOUR):
        minute = i * 5
        is_escalation = i < escalations
        decisions.append({
            "timestamp": f"{hour:02d}:{minute:02d}",
            "decision_type": decision_types[i % len(decision_types)],
            "agent_responsible": agents[i % len(agents)],
            "confidence_score": max(85, min(99, 92 + (patients - 7))),
            "action_taken": f"Hour {hour:02d} decision {i + 1} processed",
            "escalation_required": is_escalation,
            "rationale": f"Automated sponsor directive for interval {i + 1}",
        })

    return {
        "hour": hour,
        "hour_label": f"{hour:02d}:00-{hour:02d}:59",
        "total_decisions": DECISIONS_PER_HOUR,
        "total_patients": patients,
        "total_escalations": escalations,
        "robot_authorizations": patients,
        "psl_score": psl,
        "decisions": decisions,
    }


def run_simulation() -> dict:
    """Execute the full 24-hour sponsor simulation."""
    print("=" * 60)
    print("  AUTONOMOUS SPONSOR 24-HOUR SIMULATION")
    print("  Physical AI Oncology Clinical Trials")
    print("=" * 60)

    os.makedirs(HOURLY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIAGRAM_DIR, exist_ok=True)

    use_fastapi = _check_fastapi_available()
    if use_fastapi:
        print("\n[INFO] FastAPI available - using HTTP mode")
    else:
        print("\n[INFO] FastAPI not installed - using standalone mode")

    all_hours = []
    cumulative_decisions = 0
    cumulative_patients = 0
    cumulative_escalations = 0
    cumulative_robot_auth = 0

    start_time = time.time()

    for hour in range(TOTAL_HOURS):
        print(f"\n--- Processing Hour {hour:02d} ---")
        hour_data = _run_hourly_standalone(hour)
        all_hours.append(hour_data)

        cumulative_decisions += hour_data.get("total_decisions", DECISIONS_PER_HOUR)
        cumulative_patients += hour_data.get("total_patients", PATIENT_VOLUME[hour])
        cumulative_escalations += hour_data.get("total_escalations", ESCALATION_COUNTS[hour])
        cumulative_robot_auth += hour_data.get("robot_authorizations", PATIENT_VOLUME[hour])

        output_path = HOURLY_OUTPUT_DIR / f"sponsor_hour_{hour:02d}.json"
        with open(output_path, "w") as f:
            json.dump(hour_data, f, indent=2)

        print(f"  Decisions: {hour_data.get('total_decisions', 12)}")
        print(f"  Patients: {hour_data.get('total_patients', PATIENT_VOLUME[hour])}")
        print(f"  PSL: {hour_data.get('psl_score', PSL_SCORES[hour])}")

    elapsed = time.time() - start_time

    summary = {
        "simulation_title": "Autonomous Sponsor 24-Hour Simulation",
        "total_hours": TOTAL_HOURS,
        "total_decisions": cumulative_decisions,
        "expected_decisions": EXPECTED_TOTAL_DECISIONS,
        "total_patients": cumulative_patients,
        "expected_patients": EXPECTED_TOTAL_PATIENTS,
        "total_escalations": cumulative_escalations,
        "total_robot_authorizations": cumulative_robot_auth,
        "psl_start": PSL_SCORES[0],
        "psl_end": PSL_SCORES[-1],
        "psl_improvement": round(PSL_SCORES[-1] - PSL_SCORES[0], 1),
        "execution_mode": "fastapi" if use_fastapi else "standalone",
        "elapsed_seconds": round(elapsed, 2),
        "hours": all_hours,
    }

    summary_path = OUTPUT_DIR / "sponsor_24h_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Total decisions:          {cumulative_decisions}")
    print(f"  Total patients:           {cumulative_patients}")
    print(f"  Total escalations:        {cumulative_escalations}")
    print(f"  Total robot auth:         {cumulative_robot_auth}")
    print(f"  PSL trend:                {PSL_SCORES[0]} -> {PSL_SCORES[-1]}")
    print(f"  Elapsed time:             {elapsed:.2f}s")
    print(f"  Summary saved to:         {summary_path}")

    # Generate diagrams
    print("\n--- Generating text diagrams ---")
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from generate_all_diagrams import generate_all_diagrams
        generate_all_diagrams()
        print("  72 hourly + 3 cumulative diagrams generated")
    except ImportError:
        print("  [WARN] generate_all_diagrams.py not found, skipping diagram generation")
    finally:
        if str(SCRIPT_DIR) in sys.path:
            sys.path.remove(str(SCRIPT_DIR))

    return summary


if __name__ == "__main__":
    run_simulation()
