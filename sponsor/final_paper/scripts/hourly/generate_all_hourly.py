"""Run all 24 hourly sponsor scripts and write JSON output."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Execute all 24 hourly sponsor directive generators."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    total_patients = 0
    total_decisions = 0
    total_escalations = 0
    total_adverse_events = 0

    for hour in range(24):
        module_name = f"sponsor_hour_{hour:02d}"
        module_path = Path(__file__).resolve().parent / f"{module_name}.py"

        if not module_path.exists():
            print(
                f"WARNING: {module_name}.py not found, skipping hour {hour:02d}",
            )
            continue

        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        if spec is None or spec.loader is None:
            print(
                f"WARNING: Could not load {module_name}.py, skipping hour {hour:02d}",
            )
            continue

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.generate_sponsor_directives(hour)
        all_results[f"hour_{hour:02d}"] = result

        total_patients += result["patient_count"]
        total_decisions += result["summary"]["total_decisions"]
        total_escalations += result["summary"]["escalations"]
        total_adverse_events += len(result["adverse_events"])

        hour_file = OUTPUT_DIR / f"sponsor_hour_{hour:02d}_output.json"
        hour_file.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(
            f"Hour {hour:02d}: {result['patient_count']} patients, "
            f"{result['summary']['total_decisions']} decisions, "
            f"{result['summary']['escalations']} escalations",
        )

    combined = {
        "simulation_date": "2026-03-23",
        "total_hours": len(all_results),
        "total_patients": total_patients,
        "total_decisions": total_decisions,
        "total_escalations": total_escalations,
        "total_adverse_events": total_adverse_events,
        "robot_fleet_size": 29,
        "hours": all_results,
    }

    combined_file = OUTPUT_DIR / "sponsor_all_hours_combined.json"
    combined_file.write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )
    print(f"\nCombined output written to {combined_file}")
    print(
        f"Total: {total_patients} patients, "
        f"{total_decisions} decisions, "
        f"{total_escalations} escalations, "
        f"{total_adverse_events} adverse events",
    )

    if total_patients != 168:
        print(
            f"WARNING: Expected 168 total patients but got {total_patients}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
