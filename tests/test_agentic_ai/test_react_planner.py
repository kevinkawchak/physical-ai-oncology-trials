"""Tests for agentic-ai/examples-agentic-ai/02_react_procedure_planner.py"""

from __future__ import annotations


import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "react_procedure_planner",
        "agentic-ai/examples-agentic-ai/02_react_procedure_planner.py",
    )


@pytest.fixture()
def sample_anatomy(mod):
    """Build a patient anatomy with tumor and nearby structures."""
    anatomy = mod.PatientAnatomy(
        patient_id="PT-TEST-001",
        imaging_date="2026-01-15",
        imaging_modality="CT",
        registration_error_mm=1.0,
    )
    anatomy.tumor = mod.AnatomicalStructure(
        name="Right upper lobe mass",
        structure_type="tumor",
        centroid=[0.05, -0.02, 0.15],
        volume_cc=4.2,
        risk_if_damaged=mod.RiskLevel.LOW,
        min_clearance_mm=0.0,
        segmentation_confidence=0.97,
    )
    anatomy.structures = [
        mod.AnatomicalStructure(
            name="Right pulmonary artery",
            structure_type="vessel",
            centroid=[0.04, -0.01, 0.14],
            volume_cc=0.5,
            risk_if_damaged=mod.RiskLevel.CRITICAL,
            min_clearance_mm=5.0,
        ),
        mod.AnatomicalStructure(
            name="Right upper lobe bronchus",
            structure_type="airway",
            centroid=[0.045, -0.015, 0.145],
            volume_cc=0.3,
            risk_if_damaged=mod.RiskLevel.HIGH,
            min_clearance_mm=3.0,
        ),
    ]
    return anatomy


# ============================================================
# Enum tests
# ============================================================


class TestTumorType:
    def test_nsclc_value(self, mod):
        assert mod.TumorType.NSCLC.value == "non_small_cell_lung_cancer"

    def test_renal_cell_value(self, mod):
        assert mod.TumorType.RENAL_CELL.value == "renal_cell_carcinoma"

    def test_prostate_value(self, mod):
        assert mod.TumorType.PROSTATE.value == "prostate_adenocarcinoma"

    def test_all_members_count(self, mod):
        assert len(mod.TumorType) == 6

    def test_hepatocellular_value(self, mod):
        assert mod.TumorType.HEPATOCELLULAR.value == "hepatocellular_carcinoma"


class TestRiskLevel:
    def test_values(self, mod):
        assert mod.RiskLevel.LOW.value == "low"
        assert mod.RiskLevel.MODERATE.value == "moderate"
        assert mod.RiskLevel.HIGH.value == "high"
        assert mod.RiskLevel.CRITICAL.value == "critical"

    def test_ordering_by_list_index(self, mod):
        levels = list(mod.RiskLevel)
        assert levels.index(mod.RiskLevel.LOW) < levels.index(mod.RiskLevel.CRITICAL)

    def test_member_count(self, mod):
        assert len(mod.RiskLevel) == 4


# ============================================================
# Dataclass tests
# ============================================================


class TestAnatomicalStructure:
    def test_creation_with_defaults(self, mod):
        s = mod.AnatomicalStructure(name="Aorta", structure_type="vessel")
        assert s.name == "Aorta"
        assert s.structure_type == "vessel"
        assert s.centroid == [0.0, 0.0, 0.0]
        assert s.volume_cc == 0.0

    def test_risk_default(self, mod):
        s = mod.AnatomicalStructure(name="Nerve", structure_type="nerve")
        assert s.risk_if_damaged == mod.RiskLevel.MODERATE

    def test_custom_fields(self, mod):
        s = mod.AnatomicalStructure(
            name="Tumor",
            structure_type="tumor",
            centroid=[1.0, 2.0, 3.0],
            volume_cc=10.5,
            min_clearance_mm=8.0,
            segmentation_confidence=0.88,
        )
        assert s.volume_cc == 10.5
        assert s.min_clearance_mm == 8.0
        assert s.segmentation_confidence == 0.88


class TestPatientAnatomy:
    def test_creation(self, mod):
        pa = mod.PatientAnatomy(patient_id="P1")
        assert pa.patient_id == "P1"
        assert pa.structures == []
        assert pa.tumor is None
        assert pa.imaging_modality == "CT"

    def test_get_critical_structures_no_tumor(self, mod):
        pa = mod.PatientAnatomy(patient_id="P1")
        assert pa.get_critical_structures() == []

    def test_get_critical_structures_with_nearby(self, sample_anatomy):
        critical = sample_anatomy.get_critical_structures(max_distance_mm=30.0)
        assert isinstance(critical, list)
        assert len(critical) > 0
        assert "name" in critical[0]


class TestSurgicalInstrument:
    def test_creation_defaults(self, mod):
        inst = mod.SurgicalInstrument(name="Scissors", instrument_type="scissors")
        assert inst.diameter_mm == 8.0
        assert inst.working_length_mm == 400.0
        assert inst.energy_type == "none"

    def test_creation_custom(self, mod):
        inst = mod.SurgicalInstrument(
            name="Harmonic",
            instrument_type="energy_device",
            diameter_mm=5.0,
            energy_type="ultrasonic",
        )
        assert inst.energy_type == "ultrasonic"
        assert inst.diameter_mm == 5.0


class TestProcedureStep:
    def test_creation(self, mod):
        step = mod.ProcedureStep(
            step_number=1,
            action="Port placement",
            rationale="Required by protocol",
            estimated_duration_minutes=15,
        )
        assert step.step_number == 1
        assert step.estimated_duration_minutes == 15
        assert step.risk_level == mod.RiskLevel.LOW

    def test_defaults(self, mod):
        step = mod.ProcedureStep(step_number=1, action="test", rationale="test")
        assert step.instruments_needed == []
        assert step.safety_checks == []
        assert step.contingencies == []


class TestProcedurePlan:
    def test_creation(self, mod):
        plan = mod.ProcedurePlan(procedure_name="lobectomy", patient_id="P1")
        assert plan.procedure_name == "lobectomy"
        assert plan.steps == []
        assert plan.total_estimated_minutes == 0
        assert plan.overall_risk == mod.RiskLevel.MODERATE


# ============================================================
# ProcedurePlanningTools tests
# ============================================================


class TestProcedurePlanningTools:
    def test_instrument_catalog_count(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        assert len(tools._instrument_catalog) == 8

    def test_instrument_catalog_names(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        assert "maryland_bipolar" in tools._instrument_catalog
        assert "endoscope_30" in tools._instrument_catalog
        assert "harmonic_scalpel" in tools._instrument_catalog

    def test_analyze_tumor(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        result = tools.analyze_tumor()
        assert "volume_cc" in result
        assert result["volume_cc"] == 4.2

    def test_analyze_tumor_no_tumor(self, mod):
        anatomy = mod.PatientAnatomy(patient_id="P1")
        tools = mod.ProcedurePlanningTools(anatomy)
        result = tools.analyze_tumor()
        assert "error" in result

    def test_select_instruments_grasping(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        result = tools.select_instruments(["grasping"])
        assert "selected_instruments" in result
        assert len(result["selected_instruments"]) >= 1

    def test_evaluate_approach_safety_returns_dict(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        result = tools.evaluate_approach_safety([0.0, 0.0, 0.3], [0.05, -0.02, 0.15])
        assert isinstance(result, dict)
        assert "approach_safe" in result

    def test_lookup_protocol_lobectomy(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        result = tools.lookup_protocol("lobectomy")
        assert "required_steps" in result
        assert len(result["required_steps"]) > 0

    def test_lookup_protocol_unknown(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        result = tools.lookup_protocol("unknown_procedure")
        assert "error" in result

    def test_estimate_margin(self, mod, sample_anatomy):
        tools = mod.ProcedurePlanningTools(sample_anatomy)
        boundary = [[0.08, 0.0, 0.18], [0.02, -0.04, 0.12]]
        result = tools.estimate_margin(boundary)
        assert "minimum_margin_mm" in result
        assert isinstance(result["minimum_margin_mm"], float)


# ============================================================
# ReActStep tests
# ============================================================


class TestReActStep:
    def test_creation(self, mod):
        step = mod.ReActStep("thought", "Testing reasoning")
        assert step.step_type == "thought"
        assert step.content == "Testing reasoning"
        assert step.timestamp > 0

    def test_to_dict(self, mod):
        step = mod.ReActStep("action", "Run tool", {"key": "val"})
        d = step.to_dict()
        assert d["type"] == "action"
        assert "content" in d
        assert "timestamp" in d


# ============================================================
# ReActProcedurePlanner tests
# ============================================================


class TestReActProcedurePlanner:
    def test_init(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        assert planner.procedure_type == "lobectomy"
        assert planner.reasoning_trace == []

    def test_plan_procedure_returns_plan(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        plan = planner.plan_procedure()
        assert isinstance(plan, mod.ProcedurePlan)
        assert plan.patient_id == "PT-TEST-001"

    def test_plan_has_steps(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        plan = planner.plan_procedure()
        assert len(plan.steps) > 0

    def test_plan_has_reasoning_trace(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        plan = planner.plan_procedure()
        assert len(plan.reasoning_trace) > 0

    def test_plan_total_estimated_minutes(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        plan = planner.plan_procedure()
        assert plan.total_estimated_minutes > 0

    def test_plan_overall_risk(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        plan = planner.plan_procedure()
        assert plan.overall_risk in list(mod.RiskLevel)

    def test_plan_unknown_procedure_fallback(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "unknown_surgery")
        plan = planner.plan_procedure()
        assert len(plan.steps) > 0


# ============================================================
# format_procedure_plan test
# ============================================================


class TestFormatProcedurePlan:
    def test_format_returns_string(self, mod, sample_anatomy):
        planner = mod.ReActProcedurePlanner(sample_anatomy, "lobectomy")
        plan = planner.plan_procedure()
        report = mod.format_procedure_plan(plan)
        assert isinstance(report, str)
        assert "ROBOTIC ONCOLOGY PROCEDURE PLAN" in report
