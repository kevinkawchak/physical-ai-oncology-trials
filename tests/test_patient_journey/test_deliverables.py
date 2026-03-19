#!/usr/bin/env python3
"""
Tests for Deliverables Package
===============================

Validates that all deliverable artifacts exist, have sufficient content,
and that chart generators and FDA analysis module are functional.

Last updated: March 2026
DISCLAIMER: RESEARCH USE ONLY - Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DELIVERABLES_DIR = PROJECT_ROOT / "patient-journey" / "deliverables"
CHARTS_DIR = DELIVERABLES_DIR / "charts"
TABLES_DIR = DELIVERABLES_DIR / "tables"
DIAGRAMS_DIR = DELIVERABLES_DIR / "diagrams"
FDA_DIR = DELIVERABLES_DIR / "fda_cost_savings"
GUIDANCE_DIR = DELIVERABLES_DIR / "guidance"


def load_module(name: str, filepath: Path):
    if name in sys.modules:
        return sys.modules[name]
    if not filepath.exists():
        pytest.skip(f"Source file not found: {filepath}")
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        sys.modules.pop(name, None)
        pytest.skip(f"Module {name!r} requires dependency not installed: {exc}")
    return mod


# ---------------------------------------------------------------------------
# Table file tests
# ---------------------------------------------------------------------------


class TestTableFiles:
    """Verify all 6 table text files exist and have content."""

    TABLE_FILES = [
        "table_01_patient_demographics.txt",
        "table_02_regulatory_framework.txt",
        "table_03_stage_summary.txt",
        "table_04_adverse_events.txt",
        "table_05_robot_qualifications.txt",
        "table_06_treatment_outcomes.txt",
    ]

    @pytest.mark.parametrize("filename", TABLE_FILES)
    def test_table_exists(self, filename):
        fpath = TABLES_DIR / filename
        assert fpath.exists(), f"Missing table: {fpath}"

    @pytest.mark.parametrize("filename", TABLE_FILES)
    def test_table_has_content(self, filename):
        fpath = TABLES_DIR / filename
        if not fpath.exists():
            pytest.skip(f"File not found: {fpath}")
        content = fpath.read_text()
        assert len(content) > 200, f"Table too small: {filename} ({len(content)} chars)"

    def test_demographics_table_contains_patient_id(self):
        fpath = TABLES_DIR / "table_01_patient_demographics.txt"
        if not fpath.exists():
            pytest.skip("Demographics table not found")
        content = fpath.read_text()
        assert "PAT-2026-0042" in content


# ---------------------------------------------------------------------------
# Diagram file tests
# ---------------------------------------------------------------------------


class TestDiagramFiles:
    """Verify all 6 deliverable diagram files exist."""

    DIAGRAM_FILES = [
        "diagram_01_journey_overview.txt",
        "diagram_02_regulatory_mapping.txt",
        "diagram_03_data_flow.txt",
        "diagram_04_physical_ai_ecosystem.txt",
        "diagram_05_safety_architecture.txt",
        "diagram_06_trial_timeline.txt",
    ]

    @pytest.mark.parametrize("filename", DIAGRAM_FILES)
    def test_diagram_exists(self, filename):
        fpath = DIAGRAMS_DIR / filename
        assert fpath.exists(), f"Missing diagram: {fpath}"

    @pytest.mark.parametrize("filename", DIAGRAM_FILES)
    def test_diagram_has_content(self, filename):
        fpath = DIAGRAMS_DIR / filename
        if not fpath.exists():
            pytest.skip(f"File not found: {fpath}")
        content = fpath.read_text()
        assert len(content) > 300, f"Diagram too small: {filename} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Chart module tests
# ---------------------------------------------------------------------------


class TestChartModules:
    """Verify all 10 chart generator modules exist and have generate functions."""

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

    @pytest.mark.parametrize("module_name", CHART_MODULES)
    def test_chart_module_exists(self, module_name):
        fpath = CHARTS_DIR / f"{module_name}.py"
        assert fpath.exists(), f"Missing chart module: {fpath}"

    @pytest.mark.parametrize("module_name", CHART_MODULES)
    def test_chart_module_has_generate_function(self, module_name):
        fpath = CHARTS_DIR / f"{module_name}.py"
        if not fpath.exists():
            pytest.skip(f"Chart module not found: {fpath}")
        mod = load_module(f"chart_{module_name}", fpath)
        gen_funcs = [attr for attr in dir(mod) if attr.startswith("generate_")]
        assert len(gen_funcs) >= 1, f"No generate_* function in {module_name}"


# ---------------------------------------------------------------------------
# FDA cost-savings tests
# ---------------------------------------------------------------------------


class TestFDACostSavings:
    """Verify FDA cost-savings analysis module and summary."""

    def test_analysis_module_exists(self):
        fpath = FDA_DIR / "fda_cost_savings_analysis.py"
        assert fpath.exists()

    def test_summary_file_exists(self):
        fpath = FDA_DIR / "fda_cost_savings_summary.txt"
        assert fpath.exists()

    def test_analysis_has_calculate_function(self):
        fpath = FDA_DIR / "fda_cost_savings_analysis.py"
        if not fpath.exists():
            pytest.skip("FDA module not found")
        mod = load_module("fda_cost_savings_analysis", fpath)
        assert hasattr(mod, "calculate_cost_savings")
        result = mod.calculate_cost_savings()
        assert isinstance(result, dict)

    def test_analysis_has_report_function(self):
        fpath = FDA_DIR / "fda_cost_savings_analysis.py"
        if not fpath.exists():
            pytest.skip("FDA module not found")
        mod = load_module("fda_cost_savings_analysis", fpath)
        assert hasattr(mod, "generate_cost_savings_report")
        report = mod.generate_cost_savings_report()
        assert isinstance(report, str)
        assert len(report) > 500

    def test_summary_mentions_cost_reduction(self):
        fpath = FDA_DIR / "fda_cost_savings_summary.txt"
        if not fpath.exists():
            pytest.skip("FDA summary not found")
        content = fpath.read_text()
        assert "cost" in content.lower()


# ---------------------------------------------------------------------------
# Guidance document tests
# ---------------------------------------------------------------------------


class TestGuidanceDocuments:
    """Verify all 4 guidance documents exist and have content."""

    GUIDANCE_FILES = [
        "guidance_01_pharmaceutical_industry.txt",
        "guidance_02_field_observer.txt",
        "guidance_03_site_activation.txt",
        "guidance_04_patient_information.txt",
    ]

    @pytest.mark.parametrize("filename", GUIDANCE_FILES)
    def test_guidance_exists(self, filename):
        fpath = GUIDANCE_DIR / filename
        assert fpath.exists(), f"Missing guidance: {fpath}"

    @pytest.mark.parametrize("filename", GUIDANCE_FILES)
    def test_guidance_has_content(self, filename):
        fpath = GUIDANCE_DIR / filename
        if not fpath.exists():
            pytest.skip(f"File not found: {fpath}")
        content = fpath.read_text()
        assert len(content) > 500, f"Guidance too small: {filename}"

    @pytest.mark.parametrize("filename", GUIDANCE_FILES)
    def test_guidance_has_disclaimer(self, filename):
        fpath = GUIDANCE_DIR / filename
        if not fpath.exists():
            pytest.skip(f"File not found: {fpath}")
        content = fpath.read_text()
        assert "RESEARCH USE ONLY" in content


# ---------------------------------------------------------------------------
# Master generator tests
# ---------------------------------------------------------------------------


class TestMasterGenerator:
    """Verify the generate_all_deliverables.py master script."""

    def test_master_script_exists(self):
        fpath = DELIVERABLES_DIR / "generate_all_deliverables.py"
        assert fpath.exists()

    def test_master_script_has_main(self):
        fpath = DELIVERABLES_DIR / "generate_all_deliverables.py"
        if not fpath.exists():
            pytest.skip("Master script not found")
        mod = load_module("generate_all_deliverables", fpath)
        assert hasattr(mod, "main")
        assert hasattr(mod, "verify_text_deliverables")
