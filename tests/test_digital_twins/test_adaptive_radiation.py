"""Tests for digital-twins/examples-twins/03_adaptive_radiation_therapy_dt.py."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module(
        "adaptive_radiation",
        "digital-twins/examples-twins/03_adaptive_radiation_therapy_dt.py",
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestStructureType:
    def test_gtv(self, mod):
        assert mod.StructureType.GTV.value == "gross_tumor_volume"

    def test_ctv(self, mod):
        assert mod.StructureType.CTV.value == "clinical_target_volume"

    def test_ptv(self, mod):
        assert mod.StructureType.PTV.value == "planning_target_volume"

    def test_oar(self, mod):
        assert mod.StructureType.OAR.value == "organ_at_risk"

    def test_body(self, mod):
        assert mod.StructureType.BODY.value == "body_contour"


# ---------------------------------------------------------------------------
# DoseConstraint tests
# ---------------------------------------------------------------------------


class TestDoseConstraint:
    def test_creation(self, mod):
        dc = mod.DoseConstraint(structure_name="PTV", constraint_type="d_95", limit_gy=60.0, volume_percent=95.0)
        assert dc.structure_name == "PTV"
        assert dc.limit_gy == pytest.approx(60.0)

    def test_fields(self, mod):
        dc = mod.DoseConstraint(structure_name="Cord", constraint_type="max_dose", limit_gy=45.0, priority=1)
        assert dc.constraint_type == "max_dose"
        assert dc.priority == 1
        assert dc.volume_percent == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestStructure:
    def test_creation_with_type(self, mod):
        mask = np.zeros((8, 8, 8))
        mask[3:5, 3:5, 3:5] = 1.0
        s = mod.Structure(name="GTV", structure_type=mod.StructureType.GTV, mask=mask)
        assert s.structure_type == mod.StructureType.GTV
        assert s.alpha_beta == pytest.approx(10.0)

    def test_volume(self, mod):
        mask = np.zeros((8, 8, 8))
        mask[2:6, 2:6, 2:6] = 1.0
        s = mod.Structure(name="PTV", structure_type=mod.StructureType.PTV, mask=mask)
        assert np.sum(s.mask) == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# BSplineRegistration tests
# ---------------------------------------------------------------------------


class TestBSplineRegistration:
    def test_init(self, mod):
        reg = mod.BSplineRegistration(control_point_spacing=20.0)
        assert reg.control_point_spacing == pytest.approx(20.0)
        assert reg.max_iterations == 100

    def test_register_returns_deformation_field(self, mod):
        shape = (8, 16, 16)
        fixed = np.random.randn(*shape)
        moving = fixed + np.random.randn(*shape) * 0.1
        reg = mod.BSplineRegistration(control_point_spacing=10.0, smoothing_sigma=1.0)
        dvf = reg.register(fixed, moving)
        assert isinstance(dvf, mod.DeformationField)
        assert dvf.displacement.shape == (3,) + shape

    def test_register_metric_finite(self, mod):
        shape = (8, 16, 16)
        fixed = np.random.randn(*shape)
        moving = fixed.copy()
        reg = mod.BSplineRegistration(control_point_spacing=10.0, smoothing_sigma=1.0)
        dvf = reg.register(fixed, moving)
        assert np.isfinite(dvf.registration_metric)


# ---------------------------------------------------------------------------
# DoseAccumulator tests
# ---------------------------------------------------------------------------


class TestDoseAccumulator:
    def _make_accumulator(self, mod, shape=(8, 16, 16)):
        mask = np.zeros(shape)
        mask[2:6, 4:12, 4:12] = 1.0
        structure = mod.Structure(name="GTV", structure_type=mod.StructureType.GTV, mask=mask)
        return mod.DoseAccumulator(reference_shape=shape, structures=[structure], total_fractions=5)

    def test_init(self, mod):
        acc = self._make_accumulator(mod)
        assert acc.total_fractions == 5
        assert np.allclose(acc.accumulated_dose, 0.0)

    def test_add_fraction(self, mod):
        shape = (8, 16, 16)
        acc = self._make_accumulator(mod, shape)
        dose = np.ones(shape) * 2.0
        disp = np.zeros((3,) + shape)
        dvf = mod.DeformationField(displacement=disp)
        record = acc.add_fraction(1, dose, dvf)
        assert record.fraction_number == 1
        assert np.max(acc.accumulated_dose) > 0

    def test_get_cumulative_dvh(self, mod):
        shape = (8, 16, 16)
        acc = self._make_accumulator(mod, shape)
        dose = np.ones(shape) * 2.0
        disp = np.zeros((3,) + shape)
        dvf = mod.DeformationField(displacement=disp)
        record = acc.add_fraction(1, dose, dvf)
        # dvh_metrics should have GTV key
        assert "GTV" in record.dvh_metrics
        assert "max_dose_gy" in record.dvh_metrics["GTV"]

    def test_bed_zero_fractions(self, mod):
        acc = self._make_accumulator(mod)
        bed = acc.compute_bed()
        assert np.allclose(bed, 0.0)


# ---------------------------------------------------------------------------
# AdaptiveRTDigitalTwin tests
# ---------------------------------------------------------------------------


class TestAdaptiveRTDigitalTwin:
    def _make_art(self, mod, shape=(8, 16, 16)):
        planning_ct = np.random.randn(*shape) * 100 + 1024
        mask = np.zeros(shape)
        mask[2:6, 4:12, 4:12] = 1.0
        structure = mod.Structure(
            name="GTV",
            structure_type=mod.StructureType.GTV,
            mask=mask,
            constraints=[mod.DoseConstraint("GTV", "d_95", 60.0, 95.0, 1)],
        )
        planned_dose = np.ones(shape) * 60.0
        return mod.AdaptiveRTDigitalTwin(
            patient_id="ART-TEST",
            planning_ct=planning_ct,
            structures=[structure],
            planned_dose=planned_dose,
            total_fractions=5,
            prescribed_dose_gy=60.0,
        )

    def test_init(self, mod):
        art = self._make_art(mod)
        assert art.patient_id == "ART-TEST"
        assert art.total_fractions == 5

    def test_process_fraction(self, mod):
        shape = (8, 16, 16)
        art = self._make_art(mod, shape)
        daily_cbct = np.random.randn(*shape) * 100 + 1024
        result = art.process_fraction(1, daily_cbct)
        assert result["fraction"] == 1
        assert "registration_quality" in result
        assert "replanning" in result

    def test_get_accumulated_dose(self, mod):
        shape = (8, 16, 16)
        art = self._make_art(mod, shape)
        daily_cbct = np.random.randn(*shape) * 100 + 1024
        art.process_fraction(1, daily_cbct)
        dose = art.get_accumulated_dose()
        assert dose.shape == shape
        assert np.max(dose) > 0

    def test_get_treatment_summary(self, mod):
        art = self._make_art(mod)
        summary = art.get_treatment_summary()
        assert summary["patient_id"] == "ART-TEST"
        assert "fractions_delivered" in summary
