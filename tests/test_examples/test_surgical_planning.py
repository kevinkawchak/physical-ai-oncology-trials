"""Tests for examples/02_digital_twin_surgical_planning.py

Covers TumorType, imaging data models, and surgical planning.
"""

from __future__ import annotations

import numpy as np


class TestEnums:
    def test_tumor_types(self, surgical_planning_mod):
        tt = surgical_planning_mod.TumorType
        assert tt.LUNG_NSCLC.value == "non_small_cell_lung_cancer"
        assert tt.BRAIN_GBM.value == "glioblastoma"


class TestPatientImagingData:
    def test_creation(self, surgical_planning_mod):
        data = surgical_planning_mod.PatientImagingData(
            patient_id="SP-001",
            ct_volume=np.zeros((10, 10, 10)),
            voxel_spacing=(1.0, 1.0, 1.0),
        )
        assert data.patient_id == "SP-001"


class TestSurgicalDigitalTwinBuilder:
    def test_creation(self, surgical_planning_mod):
        builder = surgical_planning_mod.SurgicalDigitalTwinBuilder()
        assert builder is not None

    def test_build(self, surgical_planning_mod):
        builder = surgical_planning_mod.SurgicalDigitalTwinBuilder()
        twin = builder.build(
            patient_id="BUILD-001",
            imaging_dir="/tmp/test_imaging",
            tumor_type=surgical_planning_mod.TumorType.LUNG_NSCLC,
        )
        assert twin is not None
