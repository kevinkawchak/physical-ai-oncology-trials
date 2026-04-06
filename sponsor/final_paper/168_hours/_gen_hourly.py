"""Generator for hourly sponsor activity scripts and JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

from _config import (
    AGENTS,
    BASE_DATE,
    CANCER_TYPES,
    DECISION_TYPES,
    ESCALATION_COUNTS,
    GATE_LABELS,
    PATIENT_VOLUMES,
    PERIOD_DESCRIPTIONS,
    PERIOD_LABELS,
    PSL_RANGES,
    ROBOT_ASSIGNMENT_CATEGORIES,
    ROBOT_CATEGORIES_SHORT,
    ROBOT_FLEET,
    STAGES,
)


def _hash_pick(seed, choices):
    return choices[seed % len(choices)]


def _hash_int(seed, lo, hi):
    return lo + (seed % (hi - lo + 1))


def _psl_for_hour(day, hour_of_day):
    start, end = PSL_RANGES[day]
    return round(start + (end - start) * hour_of_day / 23, 1)


def _date_for_day(day):
    base_day = 23 + (day - 1)
    if base_day > 31:
        return f"2026-04-{base_day - 31:02d}"
    return f"2026-03-{base_day:02d}"


def generate_hourly_script(global_hour, day, hour_of_day, output_dir):
    """Generate sponsor_hour_XXX.py for a given global hour."""
    patients = PATIENT_VOLUMES[day][hour_of_day]
    period = PERIOD_LABELS[hour_of_day]
    description = PERIOD_DESCRIPTIONS[period]
    date_str = _date_for_day(day)

    # Generate patient arrivals
    patient_lines = []
    pat_start_id = global_hour * 15 + 1
    for i in range(patients):
        seed = global_hour * 1000 + i * 17 + 3
        age = _hash_int(seed, 28, 78)
        sex = "F" if seed % 2 == 0 else "M"
        cancer = _hash_pick(seed + 7, CANCER_TYPES)
        stage = _hash_pick(seed + 13, STAGES)
        ecog = _hash_int(seed + 19, 0, 3)
        robot_cat = _hash_pick(seed + 23, ROBOT_ASSIGNMENT_CATEGORIES)
        minute = _hash_int(seed + 31, 1, 55)
        pat_id = f"PAT-168H-{pat_start_id + i:04d}"
        patient_lines.append(
            f'        {{\n'
            f'            "patient_id": "{pat_id}",\n'
            f'            "arrival_minute": {minute},\n'
            f'            "age": {age},\n'
            f'            "sex": "{sex}",\n'
            f'            "cancer_type": "{cancer}",\n'
            f'            "stage": "{stage}",\n'
            f'            "ecog": {ecog},\n'
            f'            "robot_category": "{robot_cat}",\n'
            f'        }},'
        )

    # Generate decisions
    escalations = ESCALATION_COUNTS[day][hour_of_day]
    decision_lines = []
    actions = [
        "sponsor_cycle_init", "enrollment_gate_check", "telemetry_aggregation",
        "procedure_authorization_g4", "safety_interlock_verification",
        "inventory_level_audit", "crf_validation_sweep", "anomaly_triage",
        "pathway_status_sync", "milestone_verification",
        "discharge_readiness_eval", "compliance_snapshot",
    ]
    rationales = [
        f"Initialize hour-{global_hour:03d} monitoring cycle for {patients} patients",
        f"Validate enrollment criteria for cohort at {hour_of_day:02d}:05",
        f"Vitals and robot telemetry monitoring at {hour_of_day:02d}:10",
        f"Authorize procedure with gate G4 (mandatory-human-oversight)",
        f"Verify all robot interlocks and patient positioning",
        f"Confirm consumable levels for hour-{global_hour:03d} procedures",
        f"Validate CRF data integrity for hour {global_hour:03d}",
        f"Anomaly in robot telemetry at {hour_of_day:02d}:35",
        f"Status aggregation for {patients} active pathways",
        f"Procedure milestone check at {hour_of_day:02d}:45",
        f"Discharge readiness for hour-{global_hour:03d} treatment cycle",
        f"GCP and 21 CFR Part 11 compliance snapshot",
    ]
    gates = ["G1", "G2", "G3", "G4", "G1", "G2", "G3", "G4", "G1", "G2", "G3", "G4"]

    for i in range(12):
        minute = i * 5
        conf = _hash_int(global_hour * 100 + i * 7, 85, 95)
        esc = i >= (12 - escalations)
        decision_lines.append(
            f'        {{\n'
            f'            "timestamp": "{date_str}T{hour_of_day:02d}:{minute:02d}:00Z",\n'
            f'            "decision_type": "{DECISION_TYPES[i]}",\n'
            f'            "agent_responsible": "{AGENTS[i]}",\n'
            f'            "confidence_score": {conf},\n'
            f'            "action_taken": "{actions[i]}",\n'
            f'            "escalation_required": {str(esc)},\n'
            f'            "safety_gate": "{gates[i]}",\n'
            f'            "rationale": "{rationales[i]}",\n'
            f'        }},'
        )

    # Robot fleet status
    active = min(patients, 29)
    standby = max(0, 29 - active - 1)
    maint = 1

    # Procedure completions based on time of day
    completions = max(0, patients - 2) if hour_of_day >= 6 else 0

    # Adverse events
    ae_lines = "adverse_events: list[dict] = []\n    return adverse_events if isinstance(adverse_events, list) else []"

    script_content = f'''"""Sponsor activity script for Hour {global_hour:03d} (Day {day}, {hour_of_day:02d}:00 - {description})."""

from __future__ import annotations

import json
from pathlib import Path

HOUR = {global_hour}
DAY = {day}
HOUR_OF_DAY = {hour_of_day}
PATIENT_COUNT = {patients}
PERIOD = "{period}"

ROBOT_FLEET = [
    {{"category": "Surgical Robots (da Vinci Xi)", "instances": 3}},
    {{"category": "Collaborative Robots (Cobots, Franka Emika)", "instances": 4}},
    {{"category": "RT Positioning Robots", "instances": 3}},
    {{"category": "Needle Placement Systems", "instances": 2}},
    {{"category": "Social Companion Robots", "instances": 5}},
    {{"category": "Rehabilitation Exoskeletons", "instances": 2}},
    {{"category": "RT Motion-Tracking Systems", "instances": 3}},
    {{"category": "Imaging Assistance Robots", "instances": 4}},
    {{"category": "Transport/Logistics Robots", "instances": 2}},
    {{"category": "Environmental Monitoring Robots", "instances": 1}},
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
{chr(10).join(patient_lines)}
    ]


def _parse_robot_status() -> dict:
    """Return robot fleet status for this hour."""
    return {{
        "total_instances": 29,
        "active": {active},
        "standby": {standby},
        "maintenance": {maint},
        "fleet": ROBOT_FLEET,
    }}


def _parse_procedure_completions() -> int:
    """Return count of procedures completed this hour."""
    return {completions}


def _parse_adverse_events() -> list[dict]:
    """Return adverse events logged this hour."""
    {ae_lines}


def generate_sponsor_directives(hour: int = HOUR) -> dict:
    """Generate 12 sponsor decisions at 5-min intervals for hour {global_hour:03d}.

    Parameters
    ----------
    hour : int
        The global hour (0-167) to generate directives for. Defaults to {global_hour}.

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
{chr(10).join(decision_lines)}
    ]

    return {{
        "hour": hour,
        "day": DAY,
        "hour_of_day": HOUR_OF_DAY,
        "period": PERIOD,
        "description": "{description}",
        "patient_count": PATIENT_COUNT,
        "patient_arrivals": patients,
        "robot_status": robot_status,
        "procedure_completions": procedure_completions,
        "adverse_events": adverse_events,
        "decisions": decisions,
        "summary": {{
            "total_decisions": len(decisions),
            "escalations": sum(1 for d in decisions if d["escalation_required"]),
            "safety_checks": sum(
                1 for d in decisions if d["decision_type"] == "SAFETY_CHECK"
            ),
            "avg_confidence": round(
                sum(d["confidence_score"] for d in decisions) / len(decisions),
                1,
            ),
        }},
    }}


if __name__ == "__main__":
    result = generate_sponsor_directives()
    print(json.dumps(result, indent=2))
'''

    script_path = output_dir / f"sponsor_hour_{global_hour:03d}.py"
    script_path.write_text(script_content)
    return script_path


def generate_hourly_json(global_hour, day, hour_of_day, output_dir):
    """Generate JSON output for a given global hour."""
    import sys
    sys.path.insert(0, str(output_dir))
    try:
        mod_name = f"sponsor_hour_{global_hour:03d}"
        mod = __import__(mod_name)
        result = mod.generate_sponsor_directives(global_hour)
    finally:
        if str(output_dir) in sys.path:
            sys.path.remove(str(output_dir))

    json_path = output_dir / "output" / f"sponsor_hour_{global_hour:03d}_output.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return json_path


def generate_diagrams(global_hour, day, hour_of_day, diagram_dir):
    """Generate 3 text diagrams for a given global hour."""
    patients = PATIENT_VOLUMES[day][hour_of_day]
    escalations = ESCALATION_COUNTS[day][hour_of_day]
    psl = _psl_for_hour(day, hour_of_day)
    robot_auth_count = patients

    hline = "-" * 60

    # 1. Sponsor Decision Flow
    lines = [hline]
    title = f"SPONSOR DECISION FLOW - HOUR {global_hour:03d} ({hour_of_day:02d}:00-{hour_of_day:02d}:59)"
    lines.append(title.center(60))
    lines.append(hline)
    lines.append("")
    hdr = f"{'Time':<6} {'Agent':<18} {'Type':<13} {'Conf':<5} {'Esc':<4}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    esc_remaining = escalations
    for i in range(12):
        minute = i * 5
        seed = global_hour * 1000 + i * 7 + 3
        agent = _hash_pick(seed + 11, AGENTS)[:17]
        dtype = _hash_pick(seed + 23, DECISION_TYPES)[:12]
        conf = _hash_int(seed + 37, 82, 99)
        is_esc = i >= (12 - escalations)
        if is_esc:
            dtype = "ESCALATION"
        esc_flag = "Y" if is_esc else "-"
        row = f"{hour_of_day:02d}:{minute:02d}  {agent:<18} {dtype:<13} {conf}%  {esc_flag}"
        lines.append(row)

    lines.append("")
    lines.append(f"Summary: 12 decisions | {escalations} escalation(s) | PSL {psl}")
    lines.append(hline)

    flow_path = diagram_dir / f"sponsor_decision_flow_hour_{global_hour:03d}.txt"
    flow_path.write_text("\n".join(lines) + "\n")

    # 2. Agent Workload Distribution
    lines = [hline]
    title = f"AGENT WORKLOAD DISTRIBUTION - HOUR {global_hour:03d}"
    lines.append(title.center(60))
    lines.append(hline)
    lines.append("")
    hdr = f"{'Agent':<30} {'Tasks':>6} {'Dec':>6} {'Esc':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    total_tasks = patients + 4
    esc_pool = escalations
    tot_t = tot_d = tot_e = 0
    workload_rows = []
    for idx, agent in enumerate(AGENTS):
        seed = global_hour * 100 + idx * 13 + 7
        tasks = max(1, _hash_int(seed, 1, total_tasks // 3 + 1))
        decs = max(1, _hash_int(seed + 5, 1, tasks))
        esc = 0
        if esc_pool > 0 and agent in (
            "safety_agent", "clinical_accountability_agent", "study_orchestrator"
        ):
            esc = min(1, esc_pool)
            esc_pool -= esc
        workload_rows.append({"agent": agent, "tasks": tasks, "decisions": decs, "escalations": esc})
        name = agent[:29]
        lines.append(f"{name:<30} {tasks:>6} {decs:>6} {esc:>6}")
        tot_t += tasks
        tot_d += decs
        tot_e += esc

    lines.append("-" * len(hdr))
    lines.append(f"{'TOTAL':<30} {tot_t:>6} {tot_d:>6} {tot_e:>6}")
    lines.append("")
    bar_max = max(r["tasks"] for r in workload_rows)
    lines.append("Task distribution:")
    for w in workload_rows:
        bar_len = int(w["tasks"] / max(bar_max, 1) * 20)
        bar = "#" * bar_len
        short = w["agent"][:14]
        lines.append(f"  {short:<15} |{bar}")
    lines.append(hline)

    workload_path = diagram_dir / f"agent_workload_hour_{global_hour:03d}.txt"
    workload_path.write_text("\n".join(lines) + "\n")

    # 3. Robot Authorization Timeline
    lines = [hline]
    title = f"ROBOT AUTHORIZATION TIMELINE - HOUR {global_hour:03d}"
    lines.append(title.center(60))
    lines.append(hline)
    lines.append("")
    hdr = f"{'Time':<6} {'Robot Category':<15} {'Patient':<8} {'Gate':<5} {'Status':<9}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    auth_rows = []
    for i in range(robot_auth_count):
        seed = global_hour * 200 + i * 11 + 41
        minute = _hash_int(seed, 0, 59)
        cat = _hash_pick(seed + 3, ROBOT_CATEGORIES_SHORT)
        patient = f"P-{_hash_int(seed + 7, 1001, 2520):04d}"
        gate = _hash_pick(seed + 17, list(GATE_LABELS.keys()))
        status_opts = ["APPROVED", "APPROVED", "APPROVED", "PENDING", "APPROVED"]
        status = _hash_pick(seed + 29, status_opts)
        auth_rows.append({"time": f"{hour_of_day:02d}:{minute:02d}", "cat": cat, "patient": patient, "gate": gate, "status": status})

    auth_rows.sort(key=lambda r: r["time"])
    for r in auth_rows:
        lines.append(f"{r['time']:<6} {r['cat']:<15} {r['patient']:<8} {r['gate']:<5} {r['status']:<9}")

    lines.append("")
    approved = sum(1 for r in auth_rows if r["status"] == "APPROVED")
    pending = len(auth_rows) - approved
    gate_counts = {}
    for r in auth_rows:
        gate_counts[r["gate"]] = gate_counts.get(r["gate"], 0) + 1
    gate_str = " ".join(f"{g}:{c}" for g, c in sorted(gate_counts.items()))
    lines.append(f"Total: {len(auth_rows)} auth(s) | Approved: {approved} | Pending: {pending}")
    lines.append(f"Gates: {gate_str}")
    lines.append(hline)

    auth_path = diagram_dir / f"robot_auth_timeline_hour_{global_hour:03d}.txt"
    auth_path.write_text("\n".join(lines) + "\n")

    return [flow_path, workload_path, auth_path]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python _gen_hourly.py <global_hour>")
        sys.exit(1)
    gh = int(sys.argv[1])
    day = gh // 24 + 1
    hod = gh % 24
    base = Path(__file__).resolve().parent
    day_dir = base / f"day_{day:02d}"
    hourly_dir = day_dir / "hourly"
    diagram_dir = day_dir / "diagrams"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    (hourly_dir / "output").mkdir(parents=True, exist_ok=True)
    diagram_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating hour {gh:03d} (Day {day}, H{hod:02d})...")
    sp = generate_hourly_script(gh, day, hod, hourly_dir)
    print(f"  Script: {sp.name}")
    jp = generate_hourly_json(gh, day, hod, hourly_dir)
    print(f"  JSON: {jp.name}")
    dp = generate_diagrams(gh, day, hod, diagram_dir)
    for d in dp:
        print(f"  Diagram: {d.name}")
    print("Done.")
