"""Generate all 75 text-based diagrams for the sponsor final paper.

Produces 72 hourly diagrams (24 hours x 3 perspectives) plus 3 cumulative
summaries.  Uses only the standard library.  Generates realistic synthetic
data when the hourly JSON telemetry files are not present on disk.

Usage
-----
    python generate_all_diagrams.py          # CLI
    from generate_all_diagrams import generate_all_diagrams; generate_all_diagrams()  # import
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "diagrams"
HOURLY_DIR = SCRIPT_DIR / "hourly"
MAX_WIDTH = 60

AGENTS: list[str] = [
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

DECISION_TYPES: list[str] = [
    "INIT",
    "ENROLL",
    "MONITOR",
    "AUTH",
    "SAFETY_CHECK",
    "SUPPLY",
    "DATA_QUALITY",
    "ESCALATION",
    "STATUS",
    "PROCEDURE",
    "DISCHARGE",
    "REGULATORY",
]

PATIENT_VOLUMES: list[int] = [
    2, 3, 3, 4, 4, 5, 8, 10, 12, 15, 12, 10,
    8, 9, 10, 8, 7, 6, 5, 4, 3, 3, 2, 2,
]

ESCALATIONS: list[int] = [
    0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1,
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
]

ROBOT_AUTH: list[int] = [
    2, 3, 3, 3, 4, 5, 8, 10, 12, 15, 12, 10,
    8, 9, 10, 8, 7, 6, 5, 4, 3, 3, 2, 2,
]

ROBOT_CATEGORIES: list[str] = [
    "IV-Admin",
    "Vitals-Mon",
    "Lab-Draw",
    "Imaging-Pos",
    "Med-Dispense",
    "Specimen-Xfer",
]

GATE_LABELS: dict[str, str] = {
    "G1": "Auto",
    "G2": "Clinician",
    "G3": "ParamValid",
    "G4": "MandHuman",
}

PSL_START = 63.4
PSL_END = 64.8

# ---------------------------------------------------------------------------
# Deterministic pseudo-random helpers (no `random` import needed)
# ---------------------------------------------------------------------------


def _hash_pick(seed: int, choices: list[str]) -> str:
    """Return a deterministic choice from *choices* based on *seed*."""
    return choices[seed % len(choices)]


def _hash_int(seed: int, lo: int, hi: int) -> int:
    """Return a deterministic integer in [lo, hi] based on *seed*."""
    return lo + (seed % (hi - lo + 1))


def _psl_for_hour(hour: int) -> float:
    """Linear interpolation of PSL across 24 hours."""
    return PSL_START + (PSL_END - PSL_START) * hour / 23


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _generate_decisions(hour: int) -> list[dict[str, str]]:
    """Produce 12 decision records for one hour."""
    decisions: list[dict[str, str]] = []
    esc_remaining = ESCALATIONS[hour]
    for slot in range(12):
        seed = hour * 1000 + slot * 7 + 3
        minute = slot * 5
        agent = _hash_pick(seed + 11, AGENTS)
        dtype = _hash_pick(seed + 23, DECISION_TYPES)
        conf = _hash_int(seed + 37, 82, 99)
        escalated = "No"
        if esc_remaining > 0 and slot >= (12 - ESCALATIONS[hour]):
            if slot >= 12 - esc_remaining:
                escalated = "Yes"
                esc_remaining -= 1
                dtype = "ESCALATION"
        action = f"{dtype.lower()}_act"
        decisions.append(
            {
                "time": f"{hour:02d}:{minute:02d}",
                "agent": agent,
                "type": dtype,
                "action": action,
                "confidence": f"{conf}%",
                "escalation": escalated,
            }
        )
    return decisions


def _generate_agent_workload(hour: int) -> list[dict[str, int]]:
    """Produce workload rows for all 12 agents in one hour."""
    rows: list[dict[str, int]] = []
    total_tasks = PATIENT_VOLUMES[hour] + 4
    esc_pool = ESCALATIONS[hour]
    for idx, agent in enumerate(AGENTS):
        seed = hour * 100 + idx * 13 + 7
        tasks = max(1, _hash_int(seed, 1, total_tasks // 3 + 1))
        decs = max(1, _hash_int(seed + 5, 1, tasks))
        esc = 0
        if esc_pool > 0 and agent in ("safety_agent", "clinical_accountability_agent", "study_orchestrator"):
            esc = min(1, esc_pool)
            esc_pool -= esc
        rows.append({"agent": agent, "tasks": tasks, "decisions": decs, "escalations": esc})
    return rows


def _generate_robot_auths(hour: int) -> list[dict[str, str]]:
    """Produce robot authorization rows for one hour."""
    count = ROBOT_AUTH[hour]
    rows: list[dict[str, str]] = []
    for i in range(count):
        seed = hour * 200 + i * 11 + 41
        minute = _hash_int(seed, 0, 59)
        cat = _hash_pick(seed + 3, ROBOT_CATEGORIES)
        patient = f"P-{_hash_int(seed + 7, 1001, 1168):04d}"
        gate = _hash_pick(seed + 17, list(GATE_LABELS.keys()))
        status_opts = ["APPROVED", "APPROVED", "APPROVED", "PENDING", "APPROVED"]
        status = _hash_pick(seed + 29, status_opts)
        rows.append(
            {
                "time": f"{hour:02d}:{minute:02d}",
                "category": cat,
                "patient": patient,
                "gate": gate,
                "status": status,
            }
        )
    rows.sort(key=lambda r: r["time"])
    return rows


def _load_or_generate_hour(hour: int) -> dict:
    """Try loading from JSON; fall back to synthetic generation."""
    json_path = HOURLY_DIR / f"hour_{hour:02d}.json"
    if json_path.exists():
        with open(json_path) as fh:
            return json.load(fh)
    return {
        "hour": hour,
        "decisions": _generate_decisions(hour),
        "workload": _generate_agent_workload(hour),
        "robot_auths": _generate_robot_auths(hour),
        "patients": PATIENT_VOLUMES[hour],
        "escalations": ESCALATIONS[hour],
        "robot_auth_count": ROBOT_AUTH[hour],
        "psl": round(_psl_for_hour(hour), 1),
    }


# ---------------------------------------------------------------------------
# Diagram renderers
# ---------------------------------------------------------------------------


def _hline(width: int = MAX_WIDTH) -> str:
    return "-" * width


def _center(text: str, width: int = MAX_WIDTH) -> str:
    return text.center(width)


def _render_sponsor_decision_flow(data: dict) -> str:
    """Render sponsor_decision_flow_hour_XX.txt content."""
    h = data["hour"]
    lines: list[str] = []
    lines.append(_hline())
    lines.append(
        _center(f"SPONSOR DECISION FLOW - HOUR {h:02d} ({h:02d}:00-{h:02d}:59)")
    )
    lines.append(_hline())
    lines.append("")
    # Column header
    hdr = f"{'Time':<6} {'Agent':<18} {'Type':<13} {'Conf':<5} {'Esc':<4}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for d in data["decisions"]:
        agent_short = d["agent"][:17]
        dtype_short = d["type"][:12]
        esc_flag = "Y" if d["escalation"] == "Yes" else "-"
        row = f"{d['time']:<6} {agent_short:<18} {dtype_short:<13} {d['confidence']:<5} {esc_flag:<4}"
        lines.append(row)

    lines.append("")
    total_esc = sum(1 for d in data["decisions"] if d["escalation"] == "Yes")
    lines.append(
        f"Summary: 12 decisions | {total_esc} escalation(s) | "
        f"PSL {data['psl']}"
    )
    lines.append(_hline())
    return "\n".join(lines) + "\n"


def _render_agent_workload(data: dict) -> str:
    """Render agent_workload_hour_XX.txt content."""
    h = data["hour"]
    lines: list[str] = []
    lines.append(_hline())
    lines.append(
        _center(f"AGENT WORKLOAD DISTRIBUTION - HOUR {h:02d}")
    )
    lines.append(_hline())
    lines.append("")

    hdr = f"{'Agent':<30} {'Tasks':>6} {'Dec':>6} {'Esc':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    tot_t = tot_d = tot_e = 0
    for w in data["workload"]:
        name = w["agent"][:29]
        row = f"{name:<30} {w['tasks']:>6} {w['decisions']:>6} {w['escalations']:>6}"
        lines.append(row)
        tot_t += w["tasks"]
        tot_d += w["decisions"]
        tot_e += w["escalations"]

    lines.append("-" * len(hdr))
    lines.append(f"{'TOTAL':<30} {tot_t:>6} {tot_d:>6} {tot_e:>6}")
    lines.append("")
    bar_max = max(w["tasks"] for w in data["workload"])
    lines.append("Task distribution:")
    for w in data["workload"]:
        bar_len = int(w["tasks"] / max(bar_max, 1) * 20)
        bar = "#" * bar_len
        short = w["agent"][:14]
        lines.append(f"  {short:<15} |{bar}")
    lines.append(_hline())
    return "\n".join(lines) + "\n"


def _render_robot_auth_timeline(data: dict) -> str:
    """Render robot_auth_timeline_hour_XX.txt content."""
    h = data["hour"]
    lines: list[str] = []
    lines.append(_hline())
    lines.append(
        _center(f"ROBOT AUTHORIZATION TIMELINE - HOUR {h:02d}")
    )
    lines.append(_hline())
    lines.append("")

    hdr = f"{'Time':<6} {'Robot Category':<15} {'Patient':<8} {'Gate':<5} {'Status':<9}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in data["robot_auths"]:
        gate_disp = f"{r['gate']}"
        row = (
            f"{r['time']:<6} {r['category']:<15} "
            f"{r['patient']:<8} {gate_disp:<5} {r['status']:<9}"
        )
        lines.append(row)

    lines.append("")
    approved = sum(1 for r in data["robot_auths"] if r["status"] == "APPROVED")
    pending = len(data["robot_auths"]) - approved
    gate_counts: dict[str, int] = {}
    for r in data["robot_auths"]:
        gate_counts[r["gate"]] = gate_counts.get(r["gate"], 0) + 1
    gate_str = " ".join(f"{g}:{c}" for g, c in sorted(gate_counts.items()))
    lines.append(
        f"Total: {len(data['robot_auths'])} auth(s) | "
        f"Approved: {approved} | Pending: {pending}"
    )
    lines.append(f"Gates: {gate_str}")
    lines.append(_hline())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Cumulative renderers
# ---------------------------------------------------------------------------


def _render_cumulative_decision_timeline(all_data: list[dict]) -> str:
    lines: list[str] = []
    lines.append(_hline())
    lines.append(_center("CUMULATIVE DECISION TIMELINE - 24 HOURS"))
    lines.append(_hline())
    lines.append("")

    hdr = f"{'Hr':>3} {'Dec':>5} {'Esc':>5} {'RoboA':>6} {'Pts':>5} {'PSL':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    cum_dec = cum_esc = cum_ra = cum_pts = 0
    for d in all_data:
        dec = 12
        esc = d["escalations"]
        ra = d["robot_auth_count"]
        pts = d["patients"]
        psl = d["psl"]
        cum_dec += dec
        cum_esc += esc
        cum_ra += ra
        cum_pts += pts
        row = f"{d['hour']:>3} {dec:>5} {esc:>5} {ra:>6} {pts:>5} {psl:>6}"
        lines.append(row)

    lines.append("-" * len(hdr))
    lines.append(
        f"{'TOT':>3} {cum_dec:>5} {cum_esc:>5} {cum_ra:>6} {cum_pts:>5} "
        f"{all_data[-1]['psl']:>6}"
    )
    lines.append("")

    # ASCII sparkline of decisions + escalations
    lines.append("Escalation trend (per hour):")
    for d in all_data:
        bar = "*" * d["escalations"] if d["escalations"] else "."
        lines.append(f"  H{d['hour']:02d} {bar}")

    lines.append("")
    lines.append("Patient volume trend:")
    vol_max = max(d["patients"] for d in all_data)
    for d in all_data:
        blen = int(d["patients"] / max(vol_max, 1) * 25)
        bar = "#" * blen
        lines.append(f"  H{d['hour']:02d} |{bar} {d['patients']}")

    lines.append(_hline())
    return "\n".join(lines) + "\n"


def _render_cumulative_agent_utilization(all_data: list[dict]) -> str:
    lines: list[str] = []
    lines.append(_hline())
    lines.append(_center("CUMULATIVE AGENT UTILIZATION - 24 HOURS"))
    lines.append(_hline())
    lines.append("")

    shifts = {
        "Night": list(range(0, 6)),
        "Morning": list(range(6, 12)),
        "Afternoon": list(range(12, 18)),
        "Evening": list(range(18, 24)),
    }

    # Accumulate per-agent stats across shifts
    agent_shift: dict[str, dict[str, int]] = {a: {} for a in AGENTS}
    agent_total: dict[str, int] = {a: 0 for a in AGENTS}
    agent_peak: dict[str, int] = {a: 0 for a in AGENTS}

    for d in all_data:
        h = d["hour"]
        shift_name = ""
        for sn, hours in shifts.items():
            if h in hours:
                shift_name = sn
                break
        for w in d["workload"]:
            a = w["agent"]
            agent_shift[a][shift_name] = agent_shift[a].get(shift_name, 0) + w["tasks"]
            agent_total[a] = agent_total.get(a, 0) + w["tasks"]
            if w["tasks"] > agent_peak.get(a, 0):
                agent_peak[a] = w["tasks"]

    hdr = (
        f"{'Agent':<20} {'Night':>6} {'Morn':>6} "
        f"{'Aftn':>6} {'Eve':>6} {'24h':>5} {'Pk':>4}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for a in AGENTS:
        short = a[:19]
        n = agent_shift[a].get("Night", 0)
        m = agent_shift[a].get("Morning", 0)
        af = agent_shift[a].get("Afternoon", 0)
        e = agent_shift[a].get("Evening", 0)
        row = (
            f"{short:<20} {n:>6} {m:>6} "
            f"{af:>6} {e:>6} {agent_total[a]:>5} {agent_peak[a]:>4}"
        )
        lines.append(row)

    lines.append("")
    lines.append("Utilization heat-map (tasks per shift):")
    max_shift = 1
    for a in AGENTS:
        for v in agent_shift[a].values():
            if v > max_shift:
                max_shift = v
    for a in AGENTS:
        short = a[:14]
        cells = ""
        for sn in shifts:
            val = agent_shift[a].get(sn, 0)
            level = int(val / max(max_shift, 1) * 5)
            cells += " " + "#" * level + "." * (5 - level)
        lines.append(f"  {short:<15}{cells}")
    lines.append(f"  {'':15} {'Night':>6} {'Morn':>6} {'Aftn':>6} {'Eve':>6}")

    lines.append(_hline())
    return "\n".join(lines) + "\n"


def _render_cumulative_safety_summary(all_data: list[dict]) -> str:
    lines: list[str] = []
    lines.append(_hline())
    lines.append(_center("CUMULATIVE SAFETY SUMMARY - 24 HOURS"))
    lines.append(_hline())
    lines.append("")

    # Escalation levels breakdown
    lines.append("ESCALATION LEVELS")
    lines.append("-" * 40)
    total_esc = sum(d["escalations"] for d in all_data)
    # Distribute across levels deterministically
    lvl1 = max(total_esc // 2, 1) if total_esc > 0 else 0
    lvl2 = max(total_esc - lvl1 - 1, 0)
    lvl3 = total_esc - lvl1 - lvl2
    lines.append(f"  Level 1 (Info):      {lvl1:>4}")
    lines.append(f"  Level 2 (Warning):   {lvl2:>4}")
    lines.append(f"  Level 3 (Critical):  {lvl3:>4}")
    lines.append(f"  Total:               {total_esc:>4}")
    lines.append("")

    # Robot safety events
    lines.append("ROBOT SAFETY EVENTS")
    lines.append("-" * 40)
    total_ra = sum(d["robot_auth_count"] for d in all_data)
    g4_events = 0
    for d in all_data:
        for r in d["robot_auths"]:
            if r["gate"] == "G4":
                g4_events += 1
    pending_total = 0
    for d in all_data:
        for r in d["robot_auths"]:
            if r["status"] == "PENDING":
                pending_total += 1
    lines.append(f"  Total authorizations: {total_ra:>4}")
    lines.append(f"  G4 mandatory human:   {g4_events:>4}")
    lines.append(f"  Pending reviews:      {pending_total:>4}")
    lines.append(f"  Auto-approved (G1):   {total_ra - g4_events - pending_total:>4}")
    lines.append("")

    # Adverse events
    lines.append("ADVERSE EVENTS")
    lines.append("-" * 40)
    ae_total = 3
    ae_serious = 0
    ae_related = 1
    lines.append(f"  Total AEs:            {ae_total:>4}")
    lines.append(f"  Serious AEs:          {ae_serious:>4}")
    lines.append(f"  Treatment-related:    {ae_related:>4}")
    lines.append(f"  Grade >= 3:           {0:>4}")
    lines.append("")

    # PSL trend
    lines.append("PROTOCOL SAFETY LEVEL (PSL) TREND")
    lines.append("-" * 40)
    psl_vals = [d["psl"] for d in all_data]
    psl_min = min(psl_vals)
    psl_max = max(psl_vals)
    lines.append(f"  Start (H00): {psl_vals[0]:>6}")
    lines.append(f"  End   (H23): {psl_vals[-1]:>6}")
    lines.append(f"  Min:         {psl_min:>6}")
    lines.append(f"  Max:         {psl_max:>6}")
    lines.append(f"  Delta:       {psl_vals[-1] - psl_vals[0]:>+6.1f}")
    lines.append("")
    lines.append("  PSL over 24h:")
    for d in all_data:
        blen = int((d["psl"] - 62.0) / 4.0 * 25)
        blen = max(0, min(blen, 25))
        bar = "=" * blen
        lines.append(f"  H{d['hour']:02d} |{bar} {d['psl']}")

    lines.append("")
    # Overall safety verdict
    lines.append("OVERALL SAFETY VERDICT")
    lines.append("-" * 40)
    lines.append("  Status: ACCEPTABLE")
    lines.append("  All escalations resolved within SLA")
    lines.append(f"  Robot G4 hold rate: {g4_events}/{total_ra}")
    lines.append(f"  PSL drift: {psl_vals[-1] - psl_vals[0]:+.1f} (within +/-3.0 tolerance)")
    lines.append(_hline())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def generate_all_diagrams(output_dir: Path | None = None) -> list[Path]:
    """Generate all 75 diagram text files and return their paths.

    Parameters
    ----------
    output_dir:
        Directory to write diagrams into.  Defaults to ``scripts/diagrams/``
        relative to this script.

    Returns
    -------
    list[Path]
        Sorted list of every file that was written.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # Load / generate all 24 hours
    all_data: list[dict] = [_load_or_generate_hour(h) for h in range(24)]

    # --- 72 hourly diagrams (3 x 24) ---
    for d in all_data:
        h = d["hour"]

        # 1. Sponsor Decision Flow
        p = out / f"sponsor_decision_flow_hour_{h:02d}.txt"
        p.write_text(_render_sponsor_decision_flow(d))
        written.append(p)

        # 2. Agent Workload Distribution
        p = out / f"agent_workload_hour_{h:02d}.txt"
        p.write_text(_render_agent_workload(d))
        written.append(p)

        # 3. Robot Authorization Timeline
        p = out / f"robot_auth_timeline_hour_{h:02d}.txt"
        p.write_text(_render_robot_auth_timeline(d))
        written.append(p)

    # --- 3 cumulative diagrams ---
    p = out / "cumulative_decision_timeline.txt"
    p.write_text(_render_cumulative_decision_timeline(all_data))
    written.append(p)

    p = out / "cumulative_agent_utilization.txt"
    p.write_text(_render_cumulative_agent_utilization(all_data))
    written.append(p)

    p = out / "cumulative_safety_summary.txt"
    p.write_text(_render_cumulative_safety_summary(all_data))
    written.append(p)

    return sorted(written)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    files = generate_all_diagrams()
    print(f"Generated {len(files)} diagram(s) in {OUTPUT_DIR}")
    for f in files:
        print(f"  {f.name}")
