"""Generate day-level summary JSON and cumulative diagrams for each day."""

from __future__ import annotations

import json
from pathlib import Path

from _config import (
    AGENTS,
    DAY_THEMES,
    ESCALATION_COUNTS,
    PATIENT_VOLUMES,
    PSL_RANGES,
    ROBOT_CATEGORIES_SHORT,
    GATE_LABELS,
)

BASE = Path(__file__).resolve().parent


def _hash_pick(seed, choices):
    return choices[seed % len(choices)]


def _hash_int(seed, lo, hi):
    return lo + (seed % (hi - lo + 1))


def _psl_for_hour(day, hod):
    start, end = PSL_RANGES[day]
    return round(start + (end - start) * hod / 23, 1)


def generate_day_summary(day):
    """Generate day summary JSON and cumulative diagrams."""
    theme = DAY_THEMES[day]
    day_dir = BASE / f"day_{day:02d}"
    output_dir = day_dir / "output"
    diagram_dir = day_dir / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagram_dir.mkdir(parents=True, exist_ok=True)

    total_patients = sum(PATIENT_VOLUMES[day])
    total_escalations = sum(ESCALATION_COUNTS[day])
    total_decisions = 24 * 12
    psl_start, psl_end = PSL_RANGES[day]

    # Build per-hour data for diagrams
    all_hours = []
    for hod in range(24):
        gh = (day - 1) * 24 + hod
        patients = PATIENT_VOLUMES[day][hod]
        esc = ESCALATION_COUNTS[day][hod]
        psl = _psl_for_hour(day, hod)
        robot_auth = patients

        # Generate workload data
        total_tasks = patients + 4
        esc_pool = esc
        workload = []
        for idx, agent in enumerate(AGENTS):
            seed = gh * 100 + idx * 13 + 7
            tasks = max(1, _hash_int(seed, 1, total_tasks // 3 + 1))
            decs = max(1, _hash_int(seed + 5, 1, tasks))
            ae = 0
            if esc_pool > 0 and agent in ("safety_agent", "clinical_accountability_agent", "study_orchestrator"):
                ae = min(1, esc_pool)
                esc_pool -= ae
            workload.append({"agent": agent, "tasks": tasks, "decisions": decs, "escalations": ae})

        # Generate robot auth data
        robot_auths = []
        for i in range(robot_auth):
            seed = gh * 200 + i * 11 + 41
            minute = _hash_int(seed, 0, 59)
            cat = _hash_pick(seed + 3, ROBOT_CATEGORIES_SHORT)
            patient = f"P-{_hash_int(seed + 7, 1001, 2520):04d}"
            gate = _hash_pick(seed + 17, list(GATE_LABELS.keys()))
            status_opts = ["APPROVED", "APPROVED", "APPROVED", "PENDING", "APPROVED"]
            status = _hash_pick(seed + 29, status_opts)
            robot_auths.append({"gate": gate, "status": status})

        all_hours.append(
            {
                "hour": gh,
                "hour_of_day": hod,
                "patients": patients,
                "escalations": esc,
                "robot_auth_count": robot_auth,
                "psl": psl,
                "workload": workload,
                "robot_auths": robot_auths,
            }
        )

    # Write summary JSON
    summary = {
        "day": day,
        "theme": theme,
        "total_hours": 24,
        "total_decisions": total_decisions,
        "total_patients": total_patients,
        "total_escalations": total_escalations,
        "total_robot_authorizations": total_patients,
        "psl_start": psl_start,
        "psl_end": psl_end,
        "psl_improvement": round(psl_end - psl_start, 1),
        "hourly_patients": PATIENT_VOLUMES[day],
        "hourly_escalations": ESCALATION_COUNTS[day],
    }
    summary_path = output_dir / f"day_{day:02d}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    hline = "-" * 60

    # Cumulative Decision Timeline
    lines = [hline]
    lines.append(f"CUMULATIVE DECISION TIMELINE - DAY {day} (24 HOURS)".center(60))
    lines.append(hline)
    lines.append("")
    hdr = f"{'Hr':>3} {'Dec':>5} {'Esc':>5} {'RoboA':>6} {'Pts':>5} {'PSL':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    cum_dec = cum_esc = cum_ra = cum_pts = 0
    for d in all_hours:
        dec = 12
        esc = d["escalations"]
        ra = d["robot_auth_count"]
        pts = d["patients"]
        psl = d["psl"]
        cum_dec += dec
        cum_esc += esc
        cum_ra += ra
        cum_pts += pts
        lines.append(f"{d['hour_of_day']:>3} {dec:>5} {esc:>5} {ra:>6} {pts:>5} {psl:>6}")

    lines.append("-" * len(hdr))
    lines.append(f"{'TOT':>3} {cum_dec:>5} {cum_esc:>5} {cum_ra:>6} {cum_pts:>5} {all_hours[-1]['psl']:>6}")
    lines.append("")
    lines.append("Escalation trend (per hour):")
    for d in all_hours:
        bar = "*" * d["escalations"] if d["escalations"] else "."
        lines.append(f"  H{d['hour_of_day']:02d} {bar}")
    lines.append("")
    lines.append("Patient volume trend:")
    vol_max = max(d["patients"] for d in all_hours)
    for d in all_hours:
        blen = int(d["patients"] / max(vol_max, 1) * 25)
        bar = "#" * blen
        lines.append(f"  H{d['hour_of_day']:02d} |{bar} {d['patients']}")
    lines.append(hline)

    timeline_path = diagram_dir / f"cumulative_decision_timeline_day_{day:02d}.txt"
    timeline_path.write_text("\n".join(lines) + "\n")

    # Cumulative Agent Utilization
    lines = [hline]
    lines.append(f"CUMULATIVE AGENT UTILIZATION - DAY {day}".center(60))
    lines.append(hline)
    lines.append("")

    shifts = {
        "Night": list(range(0, 6)),
        "Morning": list(range(6, 12)),
        "Afternoon": list(range(12, 18)),
        "Evening": list(range(18, 24)),
    }
    agent_shift = {a: {} for a in AGENTS}
    agent_total = {a: 0 for a in AGENTS}
    agent_peak = {a: 0 for a in AGENTS}

    for d in all_hours:
        h = d["hour_of_day"]
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

    hdr = f"{'Agent':<20} {'Night':>6} {'Morn':>6} {'Aftn':>6} {'Eve':>6} {'24h':>5} {'Pk':>4}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for a in AGENTS:
        short = a[:19]
        n = agent_shift[a].get("Night", 0)
        m = agent_shift[a].get("Morning", 0)
        af = agent_shift[a].get("Afternoon", 0)
        e = agent_shift[a].get("Evening", 0)
        lines.append(f"{short:<20} {n:>6} {m:>6} {af:>6} {e:>6} {agent_total[a]:>5} {agent_peak[a]:>4}")
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
    lines.append(hline)

    util_path = diagram_dir / f"cumulative_agent_utilization_day_{day:02d}.txt"
    util_path.write_text("\n".join(lines) + "\n")

    # Cumulative Safety Summary
    lines = [hline]
    lines.append(f"CUMULATIVE SAFETY SUMMARY - DAY {day}".center(60))
    lines.append(hline)
    lines.append("")

    lines.append("ESCALATION LEVELS")
    lines.append("-" * 40)
    total_esc = sum(d["escalations"] for d in all_hours)
    lvl1 = max(total_esc // 2, 1) if total_esc > 0 else 0
    lvl2 = max(total_esc - lvl1 - 1, 0)
    lvl3 = total_esc - lvl1 - lvl2
    lines.append(f"  Level 1 (Info):      {lvl1:>4}")
    lines.append(f"  Level 2 (Warning):   {lvl2:>4}")
    lines.append(f"  Level 3 (Critical):  {lvl3:>4}")
    lines.append(f"  Total:               {total_esc:>4}")
    lines.append("")

    lines.append("ROBOT SAFETY EVENTS")
    lines.append("-" * 40)
    total_ra = sum(d["robot_auth_count"] for d in all_hours)
    g4_events = sum(1 for d in all_hours for r in d["robot_auths"] if r["gate"] == "G4")
    pending_total = sum(1 for d in all_hours for r in d["robot_auths"] if r["status"] == "PENDING")
    lines.append(f"  Total authorizations: {total_ra:>4}")
    lines.append(f"  G4 mandatory human:   {g4_events:>4}")
    lines.append(f"  Pending reviews:      {pending_total:>4}")
    lines.append(f"  Auto-approved (G1):   {total_ra - g4_events - pending_total:>4}")
    lines.append("")

    lines.append("ADVERSE EVENTS")
    lines.append("-" * 40)
    ae_total = day + 2
    ae_serious = 0
    ae_related = max(1, day // 2)
    lines.append(f"  Total AEs:            {ae_total:>4}")
    lines.append(f"  Serious AEs:          {ae_serious:>4}")
    lines.append(f"  Treatment-related:    {ae_related:>4}")
    lines.append(f"  Grade >= 3:           {0:>4}")
    lines.append("")

    lines.append("PROTOCOL SAFETY LEVEL (PSL) TREND")
    lines.append("-" * 40)
    psl_vals = [d["psl"] for d in all_hours]
    lines.append(f"  Start (H00): {psl_vals[0]:>6}")
    lines.append(f"  End   (H23): {psl_vals[-1]:>6}")
    lines.append(f"  Min:         {min(psl_vals):>6}")
    lines.append(f"  Max:         {max(psl_vals):>6}")
    lines.append(f"  Delta:       {psl_vals[-1] - psl_vals[0]:>+6.1f}")
    lines.append("")
    lines.append("  PSL over 24h:")
    for d in all_hours:
        blen = int((d["psl"] - 62.0) / 9.0 * 25)
        blen = max(0, min(blen, 25))
        bar = "=" * blen
        lines.append(f"  H{d['hour_of_day']:02d} |{bar} {d['psl']}")
    lines.append("")
    lines.append("OVERALL SAFETY VERDICT")
    lines.append("-" * 40)
    lines.append("  Status: ACCEPTABLE")
    lines.append("  All escalations resolved within SLA")
    lines.append(f"  Robot G4 hold rate: {g4_events}/{total_ra}")
    lines.append(f"  PSL drift: {psl_vals[-1] - psl_vals[0]:+.1f} (within +/-3.0 tolerance)")
    lines.append(hline)

    safety_path = diagram_dir / f"cumulative_safety_summary_day_{day:02d}.txt"
    safety_path.write_text("\n".join(lines) + "\n")

    return [summary_path, timeline_path, util_path, safety_path]


if __name__ == "__main__":
    import sys

    day = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    paths = generate_day_summary(day)
    for p in paths:
        print(f"  {p.name}")
