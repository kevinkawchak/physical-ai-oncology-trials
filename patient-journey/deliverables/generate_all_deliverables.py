#!/usr/bin/env python3
"""
Generate All Deliverables -- Physical AI Oncology Trials v2.6.0
================================================================

Master script that generates all deliverable artifacts for the
single-patient journey of PAT-2026-0042 (58F, Stage IIIB NSCLC).

Outputs:
    - 10 Plotly HTML charts in deliverables/charts/output/
    - Verification report to stdout

Usage:
    python generate_all_deliverables.py

DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DELIVERABLES_DIR = Path(__file__).resolve().parent
CHARTS_DIR = DELIVERABLES_DIR / "charts"
TABLES_DIR = DELIVERABLES_DIR / "tables"
DIAGRAMS_DIR = DELIVERABLES_DIR / "diagrams"
FDA_DIR = DELIVERABLES_DIR / "fda_cost_savings"
GUIDANCE_DIR = DELIVERABLES_DIR / "guidance"
OUTPUT_DIR = CHARTS_DIR / "output"


def _load_chart_module(name: str):
    """Load a chart module by name from the charts directory."""
    filepath = CHARTS_DIR / f"{name}.py"
    if not filepath.exists():
        logger.warning("Chart module not found: %s", filepath)
        return None
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        logger.warning("Cannot load %s: %s", name, exc)
        return None
    return mod


CHART_MODULES = [
    "chart_01_prescreening_timeline",
    "chart_02_enrollment_eligibility",
    "chart_03_digital_twin_validation",
    "chart_04_robot_usl_scores",
    "chart_05_surgery_metrics",
    "chart_06_recovery_ae_timeline",
    "chart_07_immunotherapy_cycles",
    "chart_08_federation_convergence",
    "chart_09_surveillance_risk",
    "chart_10_closeout_summary",
]


def generate_charts() -> list[str]:
    """Generate all 10 Plotly charts, saving HTML to output directory.

    Returns:
        List of generated chart file paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    for chart_name in CHART_MODULES:
        mod = _load_chart_module(chart_name)
        if mod is None:
            continue

        # Each chart module exposes a generate_*() function
        gen_func = None
        for attr_name in dir(mod):
            if attr_name.startswith("generate_"):
                gen_func = getattr(mod, attr_name)
                break

        if gen_func is None:
            logger.warning("No generate_* function found in %s", chart_name)
            continue

        try:
            fig = gen_func()
            output_path = OUTPUT_DIR / f"{chart_name}.html"
            fig.write_html(str(output_path))
            generated.append(str(output_path))
            logger.info("Generated: %s", output_path.name)
        except Exception as exc:
            logger.error("Failed to generate %s: %s", chart_name, exc)

    return generated


def verify_text_deliverables() -> dict[str, list[str]]:
    """Verify that all text-based deliverables exist.

    Returns:
        Dictionary with categories and lists of found files.
    """
    results: dict[str, list[str]] = {}

    # Tables
    table_files = sorted(TABLES_DIR.glob("table_*.txt"))
    results["tables"] = [f.name for f in table_files]

    # Diagrams
    diagram_files = sorted(DIAGRAMS_DIR.glob("diagram_*.txt"))
    results["diagrams"] = [f.name for f in diagram_files]

    # FDA cost savings
    fda_files = sorted(FDA_DIR.glob("fda_cost_savings_*"))
    results["fda_cost_savings"] = [f.name for f in fda_files]

    # Guidance
    guidance_files = sorted(GUIDANCE_DIR.glob("guidance_*.txt"))
    results["guidance"] = [f.name for f in guidance_files]

    return results


def generate_fda_report() -> str | None:
    """Generate the FDA cost-savings report text.

    Returns:
        Report text or None if module not available.
    """
    filepath = FDA_DIR / "fda_cost_savings_analysis.py"
    if not filepath.exists():
        logger.warning("FDA cost-savings module not found")
        return None
    spec = importlib.util.spec_from_file_location("fda_cost_savings_analysis", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        logger.warning("Cannot load FDA module: %s", exc)
        return None

    if hasattr(mod, "generate_cost_savings_report"):
        return mod.generate_cost_savings_report()
    return None


def main() -> int:
    """Run all deliverable generation and verification."""
    logger.info("=" * 60)
    logger.info("Physical AI Oncology Trials v2.6.0 -- Deliverables")
    logger.info("Patient: PAT-2026-0042 (58F, Stage IIIB NSCLC)")
    logger.info("=" * 60)

    # Verify text deliverables
    logger.info("\n--- Text Deliverables Verification ---")
    text_results = verify_text_deliverables()
    total_text = 0
    for category, files in text_results.items():
        logger.info("  %s: %d files", category, len(files))
        for f in files:
            logger.info("    - %s", f)
        total_text += len(files)
    logger.info("  Total text deliverables: %d", total_text)

    # Generate charts (requires plotly)
    logger.info("\n--- Chart Generation ---")
    try:
        charts = generate_charts()
        logger.info("  Generated %d charts", len(charts))
    except Exception as exc:
        logger.warning("  Chart generation skipped: %s", exc)
        charts = []

    # FDA report
    logger.info("\n--- FDA Cost-Savings Report ---")
    report = generate_fda_report()
    if report:
        logger.info("  Report generated (%d chars)", len(report))
    else:
        logger.info("  Report generation skipped (module not available)")

    # Summary
    logger.info("\n--- Summary ---")
    logger.info("  Charts: %d / 10", len(charts))
    logger.info("  Tables: %d / 6", len(text_results.get("tables", [])))
    logger.info("  Diagrams: %d / 6", len(text_results.get("diagrams", [])))
    logger.info("  FDA files: %d / 2", len(text_results.get("fda_cost_savings", [])))
    logger.info("  Guidance: %d / 4", len(text_results.get("guidance", [])))
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
