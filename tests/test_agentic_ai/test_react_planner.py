"""Tests for agentic-ai/examples-agentic-ai/02_react_procedure_planner.py

Covers ReAct procedure planning, anatomy models, and surgical instruments.
"""

from __future__ import annotations


# ── Enum values ──────────────────────────────────────────────────────────


class TestEnums:
    def test_tumor_types(self, react_mod):
        assert react_mod.TumorType.NSCLC.value == "non_small_cell_lung_cancer"
        assert react_mod.TumorType.RENAL_CELL.value == "renal_cell_carcinoma"

    def test_risk_levels(self, react_mod):
        rl = react_mod.RiskLevel
        assert rl.LOW.value == "low"
        assert rl.CRITICAL.value == "critical"


# ── AnatomicalStructure dataclass ────────────────────────────────────────


class TestAnatomicalStructure:
    def test_creation(self, react_mod):
        s = react_mod.AnatomicalStructure(
            name="aorta",
            structure_type="vessel",
        )
        assert s.name == "aorta"
        assert s.risk_if_damaged == react_mod.RiskLevel.MODERATE


# ── PatientAnatomy dataclass ─────────────────────────────────────────────


class TestPatientAnatomy:
    def test_creation(self, react_mod):
        anatomy = react_mod.PatientAnatomy(patient_id="PA-001")
        assert anatomy.patient_id == "PA-001"
        assert anatomy.structures == []


# ── SurgicalInstrument dataclass ─────────────────────────────────────────


class TestSurgicalInstrument:
    def test_defaults(self, react_mod):
        inst = react_mod.SurgicalInstrument(
            name="grasper",
            instrument_type="manipulation",
        )
        assert inst.diameter_mm == 8.0
        assert inst.working_length_mm == 400.0


# ── ProcedureStep dataclass ──────────────────────────────────────────────


class TestProcedureStep:
    def test_creation(self, react_mod):
        step = react_mod.ProcedureStep(
            step_number=1,
            action="Incision",
            rationale="Access tumor site",
        )
        assert step.step_number == 1
        assert step.risk_level == react_mod.RiskLevel.LOW


# ── ProcedurePlanningTools ───────────────────────────────────────────────


class TestProcedurePlanningTools:
    def _make_anatomy(self, react_mod):
        tumor = react_mod.AnatomicalStructure(
            name="tumor",
            structure_type="target",
            centroid=[0.0, 0.0, 0.0],
            volume_cc=15.0,
        )
        anatomy = react_mod.PatientAnatomy(
            patient_id="TOOL-001",
            structures=[tumor],
            tumor=tumor,
        )
        return anatomy

    def test_creation(self, react_mod):
        anatomy = self._make_anatomy(react_mod)
        tools = react_mod.ProcedurePlanningTools(anatomy=anatomy)
        assert tools.anatomy.patient_id == "TOOL-001"

    def test_analyze_tumor(self, react_mod):
        anatomy = self._make_anatomy(react_mod)
        tools = react_mod.ProcedurePlanningTools(anatomy=anatomy)
        result = tools.analyze_tumor()
        assert isinstance(result, dict)
        assert "volume_cc" in result

    def test_lookup_protocol(self, react_mod):
        anatomy = self._make_anatomy(react_mod)
        tools = react_mod.ProcedurePlanningTools(anatomy=anatomy)
        protocol = tools.lookup_protocol("lobectomy")
        assert isinstance(protocol, dict)


# ── ReActProcedurePlanner ────────────────────────────────────────────────


class TestReActProcedurePlanner:
    def _make_anatomy(self, react_mod):
        tumor = react_mod.AnatomicalStructure(
            name="tumor",
            structure_type="target",
            centroid=[0.0, 0.0, 0.0],
            volume_cc=15.0,
        )
        return react_mod.PatientAnatomy(
            patient_id="REACT-001",
            structures=[tumor],
            tumor=tumor,
        )

    def test_creation(self, react_mod):
        anatomy = self._make_anatomy(react_mod)
        planner = react_mod.ReActProcedurePlanner(
            anatomy=anatomy,
            procedure_type="lobectomy",
        )
        assert planner.anatomy.patient_id == "REACT-001"

    def test_plan_procedure_returns_plan(self, react_mod):
        anatomy = self._make_anatomy(react_mod)
        planner = react_mod.ReActProcedurePlanner(
            anatomy=anatomy,
            procedure_type="lobectomy",
        )
        plan = planner.plan_procedure()
        assert isinstance(plan, react_mod.ProcedurePlan)
        assert len(plan.steps) > 0
        assert plan.patient_id == "REACT-001"


# ── Utility function ─────────────────────────────────────────────────────


class TestEuclideanDistance:
    def test_same_point(self, react_mod):
        dist = react_mod._euclidean_distance([0, 0, 0], [0, 0, 0])
        assert dist == 0.0

    def test_known_distance(self, react_mod):
        dist = react_mod._euclidean_distance([0, 0, 0], [3, 4, 0])
        assert abs(dist - 5.0) < 1e-9
