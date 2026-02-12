"""Unit tests for tools/dicom-inspector/dicom_inspector.py.

Tests cover PHI_TAGS, REQUIRED_TRIAL_TAGS, ONCOLOGY_MODALITIES constants,
InspectionResult dataclass, and file validation logic.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

MOD_PATH = "tools/dicom-inspector/dicom_inspector.py"


@pytest.fixture()
def mod():
    return load_module("dicom_inspector", MOD_PATH)


class TestConstants:
    def test_phi_tags_count(self, mod):
        assert len(mod.PHI_TAGS) >= 15

    def test_phi_tags_include_hipaa_identifiers(self, mod):
        # Should include key HIPAA identifiers
        tag_names = list(mod.PHI_TAGS.values())
        has_name_tag = any("name" in v.lower() for v in tag_names)
        assert has_name_tag

    def test_required_trial_tags(self, mod):
        assert len(mod.REQUIRED_TRIAL_TAGS) >= 3

    def test_oncology_modalities(self, mod):
        for m in ("CT", "MR", "PT", "RTPLAN", "RTSTRUCT", "RTDOSE"):
            assert m in mod.ONCOLOGY_MODALITIES

    def test_dicom_rt_tags(self, mod):
        assert len(mod.DICOM_RT_TAGS) >= 3


class TestInspectionResult:
    def test_creation(self, mod):
        result = mod.InspectionResult(
            filepath="/data/test.dcm",
            modality="CT",
            sop_class="1.2.840.10008.5.1.4.1.1.2",
            study_date="20260115",
            series_description="Chest CT",
            phi_tags_present=[],
            missing_required_tags=[],
            warnings=[],
            is_valid=True,
        )
        assert result.modality == "CT"
        assert result.is_valid is True

    def test_to_dict(self, mod):
        result = mod.InspectionResult(
            filepath="/data/test.dcm",
            modality="MR",
            sop_class="",
            study_date="20260120",
            series_description="Brain MRI",
            phi_tags_present=["PatientName"],
            missing_required_tags=[],
            warnings=["PHI detected"],
            is_valid=True,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["modality"] == "MR"
        assert len(d["phi_tags_present"]) == 1

    def test_invalid_result(self, mod):
        result = mod.InspectionResult(
            filepath="/data/bad.dcm",
            modality="",
            sop_class="",
            study_date="",
            series_description="",
            phi_tags_present=[],
            missing_required_tags=["Modality", "StudyDate"],
            warnings=["Missing required tags"],
            is_valid=False,
        )
        assert result.is_valid is False
        assert len(result.missing_required_tags) == 2
