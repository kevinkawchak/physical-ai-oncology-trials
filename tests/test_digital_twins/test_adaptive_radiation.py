"""Tests for digital-twins/examples-twins/03_adaptive_radiation_therapy_dt.py

Covers StructureType enum, BSplineRegistration, DoseAccumulator, and
AdaptiveRTDigitalTwin.
"""

from __future__ import annotations

import numpy as np


# ── Enum values ──────────────────────────────────────────────────────────


class TestStructureTypeEnum:
    def test_members(self, adaptive_rt_mod):
        st = adaptive_rt_mod.StructureType
        assert st.GTV.value == "gross_tumor_volume"
        assert st.CTV.value == "clinical_target_volume"
        assert st.PTV.value == "planning_target_volume"
        assert st.OAR.value == "organ_at_risk"
        assert st.BODY.value == "body_contour"


# ── DoseConstraint dataclass ─────────────────────────────────────────────


class TestDoseConstraint:
    def test_creation(self, adaptive_rt_mod):
        dc = adaptive_rt_mod.DoseConstraint(
            structure_name="PTV",
            constraint_type="D95",
            limit_gy=57.0,
        )
        assert dc.structure_name == "PTV"
        assert dc.limit_gy == 57.0
        assert dc.priority == 1


# ── BSplineRegistration ─────────────────────────────────────────────────


class TestBSplineRegistration:
    def test_creation(self, adaptive_rt_mod):
        reg = adaptive_rt_mod.BSplineRegistration(
            control_point_spacing=20.0,
            smoothing_sigma=2.0,
        )
        assert reg.control_point_spacing == 20.0

    def test_register_returns_deformation_field(self, adaptive_rt_mod):
        reg = adaptive_rt_mod.BSplineRegistration()
        fixed = np.random.rand(16, 16, 16).astype(np.float32)
        moving = np.random.rand(16, 16, 16).astype(np.float32)
        dvf = reg.register(fixed, moving)
        assert isinstance(dvf, adaptive_rt_mod.DeformationField)
        assert dvf.displacement.shape[0] == 3

    def test_warp_mask_preserves_shape(self, adaptive_rt_mod):
        reg = adaptive_rt_mod.BSplineRegistration()
        fixed = np.random.rand(16, 16, 16).astype(np.float32)
        moving = np.random.rand(16, 16, 16).astype(np.float32)
        dvf = reg.register(fixed, moving)
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[6:10, 6:10, 6:10] = 1.0
        warped = reg.warp_mask(mask, dvf)
        assert warped.shape == mask.shape

    def test_warp_dose_preserves_shape(self, adaptive_rt_mod):
        reg = adaptive_rt_mod.BSplineRegistration()
        fixed = np.random.rand(16, 16, 16).astype(np.float32)
        moving = np.random.rand(16, 16, 16).astype(np.float32)
        dvf = reg.register(fixed, moving)
        dose = np.random.rand(16, 16, 16).astype(np.float32) * 2.0
        warped = reg.warp_dose(dose, dvf)
        assert warped.shape == dose.shape


# ── DoseAccumulator ──────────────────────────────────────────────────────


class TestDoseAccumulator:
    def _make_accumulator(self, adaptive_rt_mod):
        shape = (16, 16, 16)
        mask = np.zeros(shape, dtype=np.float32)
        mask[6:10, 6:10, 6:10] = 1.0
        structures = [
            adaptive_rt_mod.Structure(
                name="PTV",
                structure_type=adaptive_rt_mod.StructureType.PTV,
                mask=mask,
                alpha_beta=10.0,
            ),
        ]
        return adaptive_rt_mod.DoseAccumulator(
            reference_shape=shape,
            structures=structures,
            total_fractions=30,
        )

    def test_creation(self, adaptive_rt_mod):
        acc = self._make_accumulator(adaptive_rt_mod)
        assert acc.total_fractions == 30

    def test_compute_bed(self, adaptive_rt_mod):
        acc = self._make_accumulator(adaptive_rt_mod)
        bed = acc.compute_bed(alpha_beta=10.0)
        assert bed.shape == (16, 16, 16)
        # No dose accumulated yet, BED should be zero
        assert np.allclose(bed, 0.0)

    def test_get_accumulation_summary(self, adaptive_rt_mod):
        acc = self._make_accumulator(adaptive_rt_mod)
        summary = acc.get_accumulation_summary()
        assert "fractions_delivered" in summary
        assert summary["fractions_delivered"] == 0
