from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("digital_twin_surgical_planning", "examples/02_digital_twin_surgical_planning.py")


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for TumorType enum."""

    def test_lung_nsclc(self, mod):
        assert mod.TumorType.LUNG_NSCLC.value == "non_small_cell_lung_cancer"

    def test_lung_sclc(self, mod):
        assert mod.TumorType.LUNG_SCLC.value == "small_cell_lung_cancer"

    def test_liver_hcc(self, mod):
        assert mod.TumorType.LIVER_HCC.value == "hepatocellular_carcinoma"

    def test_brain_gbm(self, mod):
        assert mod.TumorType.BRAIN_GBM.value == "glioblastoma"

    def test_pancreatic(self, mod):
        assert mod.TumorType.PANCREATIC.value == "pancreatic_adenocarcinoma"


# ---------------------------------------------------------------------------
# TestPatientImagingData
# ---------------------------------------------------------------------------


class TestPatientImagingData:
    """Tests for PatientImagingData dataclass."""

    def test_creation(self, mod):
        data = mod.PatientImagingData(patient_id="P001")
        assert data.patient_id == "P001"

    def test_default_ct_none(self, mod):
        data = mod.PatientImagingData(patient_id="P001")
        assert data.ct_volume is None

    def test_default_voxel_spacing(self, mod):
        data = mod.PatientImagingData(patient_id="P001")
        assert data.voxel_spacing == (1.0, 1.0, 1.0)

    def test_default_orientation_identity(self, mod):
        data = mod.PatientImagingData(patient_id="P001")
        np.testing.assert_array_equal(data.orientation, np.eye(4))


# ---------------------------------------------------------------------------
# TestTumorSegmentation
# ---------------------------------------------------------------------------


class TestTumorSegmentation:
    """Tests for TumorSegmentation dataclass."""

    def test_creation(self, mod):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        seg = mod.TumorSegmentation(mask=mask, gross_tumor_volume=5.0)
        assert seg.gross_tumor_volume == 5.0

    def test_default_margin(self, mod):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        seg = mod.TumorSegmentation(mask=mask)
        assert seg.margin_mm == 5.0

    def test_confidence_default(self, mod):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        seg = mod.TumorSegmentation(mask=mask)
        assert seg.confidence == 0.0


# ---------------------------------------------------------------------------
# TestCriticalStructures
# ---------------------------------------------------------------------------


class TestCriticalStructures:
    """Tests for CriticalStructures dataclass."""

    def test_creation(self, mod):
        cs = mod.CriticalStructures()
        assert cs.vessels == {}
        assert cs.nerves == {}
        assert cs.organs_at_risk == {}


# ---------------------------------------------------------------------------
# TestSurgicalDigitalTwinBuilder
# ---------------------------------------------------------------------------


class TestSurgicalDigitalTwinBuilder:
    """Tests for SurgicalDigitalTwinBuilder class."""

    def test_init(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        assert builder is not None

    def test_build_produces_twin(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        twin = builder.build(
            patient_id="TEST-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=mod.TumorType.LUNG_NSCLC,
        )
        assert isinstance(twin, mod.SurgicalDigitalTwin)
        assert twin.patient_id == "TEST-001"

    def test_build_twin_has_tumor(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        twin = builder.build(
            patient_id="TEST-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=mod.TumorType.LUNG_NSCLC,
        )
        assert twin.tumor is not None
        assert twin.tumor.gross_tumor_volume > 0


# ---------------------------------------------------------------------------
# TestSurgicalDigitalTwin
# ---------------------------------------------------------------------------


class TestSurgicalDigitalTwin:
    """Tests for SurgicalDigitalTwin class."""

    def _make_twin(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        return builder.build(
            patient_id="TEST-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=mod.TumorType.LUNG_NSCLC,
        )

    def test_plan_resection_returns_plan(self, mod):
        twin = self._make_twin(mod)
        plan = twin.plan_resection(margin_mm=10.0)
        assert isinstance(plan, mod.ResectionPlan)

    def test_plan_resection_has_trajectory(self, mod):
        twin = self._make_twin(mod)
        plan = twin.plan_resection(margin_mm=10.0, approach="thoracoscopic")
        assert "entry_point" in plan.trajectory
        assert "target_point" in plan.trajectory

    def test_plan_resection_feasibility_score(self, mod):
        twin = self._make_twin(mod)
        plan = twin.plan_resection(margin_mm=10.0)
        assert 0.0 <= plan.feasibility_score <= 1.0

    def test_resection_plan_generate_report(self, mod):
        twin = self._make_twin(mod)
        plan = twin.plan_resection(margin_mm=10.0)
        report = plan.generate_report()
        assert isinstance(report, str)
        assert "SURGICAL PLANNING REPORT" in report


# ---------------------------------------------------------------------------
# TestVirtualSurgerySimulator
# ---------------------------------------------------------------------------


class TestVirtualSurgerySimulator:
    """Tests for VirtualSurgerySimulator class."""

    def test_init(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        twin = builder.build(
            patient_id="TEST-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=mod.TumorType.LUNG_NSCLC,
        )
        sim = mod.VirtualSurgerySimulator(twin)
        assert sim.twin is twin

    def test_start_simulation(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        twin = builder.build(
            patient_id="TEST-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=mod.TumorType.LUNG_NSCLC,
        )
        sim = mod.VirtualSurgerySimulator(twin)
        plan = twin.plan_resection(margin_mm=10.0)
        result = sim.start_simulation(plan)
        assert result["status"] == "ready"

    def test_analyze_rehearsal(self, mod):
        builder = mod.SurgicalDigitalTwinBuilder()
        twin = builder.build(
            patient_id="TEST-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=mod.TumorType.LUNG_NSCLC,
        )
        sim = mod.VirtualSurgerySimulator(twin)
        trajectory = np.random.randn(50, 7)
        analysis = sim.analyze_rehearsal(trajectory)
        assert "total_path_length_mm" in analysis
        assert "margin_violations" in analysis
