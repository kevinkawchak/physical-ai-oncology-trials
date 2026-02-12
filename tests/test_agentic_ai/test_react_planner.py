"""Unit tests for agentic-ai/examples-agentic-ai/02_react_procedure_planner.py.

Tests cover TumorType, RiskLevel enums, AnatomicalStructure, PatientAnatomy,
SurgicalInstrument, ProcedureStep, ProcedurePlan dataclasses, and planning tools.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

MOD_PATH = "agentic-ai/examples-agentic-ai/02_react_procedure_planner.py"


@pytest.fixture()
def mod():
    return load_module("react_procedure_planner", MOD_PATH)


class TestEnums:
    def test_tumor_type(self, mod):
        names = [e.name for e in mod.TumorType]
        for t in ("NSCLC", "RENAL_CELL", "PROSTATE"):
            assert t in names

    def test_risk_level(self, mod):
        names = [e.name for e in mod.RiskLevel]
        for r in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
            assert r in names


class TestAnatomicalStructure:
    def test_creation(self, mod):
        struct = mod.AnatomicalStructure(
            name="Aorta",
            structure_type="vessel",
            centroid=np.array([0.0, 0.0, 0.0]),
            volume_cc=50.0,
            risk_if_damaged="CRITICAL",
            min_clearance_mm=5.0,
            segmentation_confidence=0.95,
        )
        assert struct.name == "Aorta"
        assert struct.segmentation_confidence > 0.9


class TestPatientAnatomy:
    def test_creation(self, mod):
        tumor = mod.AnatomicalStructure(
            name="Tumor",
            structure_type="lesion",
            centroid=np.array([10.0, 5.0, -3.0]),
            volume_cc=15.0,
            risk_if_damaged="N/A",
            min_clearance_mm=0.0,
            segmentation_confidence=0.92,
        )
        anatomy = mod.PatientAnatomy(
            patient_id="PT-PLAN-001",
            structures=[tumor],
            tumor=tumor,
            imaging_date="2026-01-15",
            imaging_modality="CT",
        )
        assert anatomy.patient_id == "PT-PLAN-001"
        assert len(anatomy.structures) == 1


class TestSurgicalInstrument:
    def test_creation(self, mod):
        inst = mod.SurgicalInstrument(
            name="Harmonic Scalpel",
            instrument_type="dissection",
            diameter_mm=5.0,
            working_length_mm=350.0,
            articulation_degrees=60.0,
            energy_type="ultrasonic",
        )
        assert inst.diameter_mm == 5.0


class TestProcedureStep:
    def test_creation(self, mod):
        step = mod.ProcedureStep(
            step_number=1,
            action="Port placement",
            rationale="Establish laparoscopic access",
            instruments_needed=["Trocar"],
            risk_level=mod.RiskLevel.LOW,
            estimated_duration_minutes=10,
            safety_checks=["Verify entry point"],
            contingencies=["Switch to open if adhesions"],
        )
        assert step.step_number == 1
        assert step.risk_level == mod.RiskLevel.LOW


class TestProcedurePlanningTools:
    def test_instantiation(self, mod):
        tumor = mod.AnatomicalStructure(
            name="Tumor",
            structure_type="lesion",
            centroid=np.array([0.0, 0.0, 0.0]),
            volume_cc=10.0,
            risk_if_damaged="N/A",
            min_clearance_mm=0.0,
            segmentation_confidence=0.9,
        )
        anatomy = mod.PatientAnatomy(
            patient_id="PT-TOOLS-001",
            structures=[tumor],
            tumor=tumor,
            imaging_date="2026-01-15",
            imaging_modality="CT",
        )
        tools = mod.ProcedurePlanningTools(anatomy=anatomy)
        assert tools is not None
        assert len(tools._instrument_catalog) > 0
