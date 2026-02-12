from __future__ import annotations

import pytest

from tests.conftest import load_module


@pytest.fixture()
def mod():
    return load_module("dicom_inspector", "tools/dicom-inspector/dicom_inspector.py")


# ---------------------------------------------------------------------------
# TestInspectionResult
# ---------------------------------------------------------------------------


class TestInspectionResult:
    """Tests for the InspectionResult dataclass."""

    def test_creation_minimal(self, mod):
        result = mod.InspectionResult(filepath="/tmp/test.dcm")
        assert result.filepath == "/tmp/test.dcm"

    def test_default_is_valid(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        assert result.is_valid is True

    def test_default_modality_empty(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        assert result.modality == ""

    def test_default_lists_empty(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        assert result.phi_tags_present == []
        assert result.missing_required_tags == []
        assert result.warnings == []

    def test_default_pixel_data_false(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        assert result.pixel_data_present is False

    def test_default_dimensions_zero(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        assert result.rows == 0
        assert result.columns == 0

    def test_to_dict_returns_dict(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_filepath(self, mod):
        result = mod.InspectionResult(filepath="/data/scan.dcm")
        d = result.to_dict()
        assert d["filepath"] == "/data/scan.dcm"

    def test_to_dict_dimensions_format(self, mod):
        result = mod.InspectionResult(filepath="x.dcm", rows=512, columns=512)
        d = result.to_dict()
        assert d["dimensions"] == "512x512"

    def test_to_dict_dimensions_empty_when_no_rows(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        d = result.to_dict()
        assert d["dimensions"] == ""

    def test_custom_fields(self, mod):
        result = mod.InspectionResult(
            filepath="x.dcm",
            modality="CT",
            sop_class="1.2.840.10008.5.1.4.1.1.2",
            study_date="20260101",
            bits_allocated=16,
        )
        assert result.modality == "CT"
        assert result.study_date == "20260101"
        assert result.bits_allocated == 16

    def test_to_dict_all_keys(self, mod):
        result = mod.InspectionResult(filepath="x.dcm")
        d = result.to_dict()
        expected_keys = {
            "filepath",
            "modality",
            "sop_class",
            "study_date",
            "series_description",
            "phi_tags_present",
            "missing_required_tags",
            "warnings",
            "is_valid",
            "pixel_data_present",
            "dimensions",
            "bits_allocated",
            "transfer_syntax",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# TestDicomTagConstants
# ---------------------------------------------------------------------------


class TestDicomTagConstants:
    """Tests for DICOM tag constant dictionaries."""

    def test_phi_tags_exists(self, mod):
        assert hasattr(mod, "PHI_TAGS")
        assert isinstance(mod.PHI_TAGS, dict)

    def test_phi_tags_contains_patient_name(self, mod):
        assert (0x0010, 0x0010) in mod.PHI_TAGS
        assert mod.PHI_TAGS[(0x0010, 0x0010)] == "PatientName"

    def test_phi_tags_contains_patient_id(self, mod):
        assert (0x0010, 0x0020) in mod.PHI_TAGS

    def test_required_trial_tags_exists(self, mod):
        assert hasattr(mod, "REQUIRED_TRIAL_TAGS")
        assert isinstance(mod.REQUIRED_TRIAL_TAGS, dict)

    def test_required_trial_tags_contains_modality(self, mod):
        assert (0x0008, 0x0060) in mod.REQUIRED_TRIAL_TAGS

    def test_dicom_rt_tags_exists(self, mod):
        assert hasattr(mod, "DICOM_RT_TAGS")
        assert isinstance(mod.DICOM_RT_TAGS, dict)

    def test_oncology_modalities_exists(self, mod):
        assert hasattr(mod, "ONCOLOGY_MODALITIES")
        assert "CT" in mod.ONCOLOGY_MODALITIES
        assert "MR" in mod.ONCOLOGY_MODALITIES
        assert "RTSTRUCT" in mod.ONCOLOGY_MODALITIES

    def test_phi_tags_nonempty(self, mod):
        assert len(mod.PHI_TAGS) > 10


# ---------------------------------------------------------------------------
# TestInspectFile
# ---------------------------------------------------------------------------


class TestInspectFile:
    """Tests for inspect_file and related logic."""

    def test_inspect_file_exists(self, mod):
        assert callable(mod.inspect_file)

    def test_inspect_nonexistent_file_returns_invalid(self, mod):
        if not mod.HAS_PYDICOM:
            pytest.skip("pydicom not installed")
        result = mod.inspect_file("/nonexistent/path/file.dcm")
        assert result.is_valid is False

    def test_inspect_nonexistent_file_has_warnings(self, mod):
        if not mod.HAS_PYDICOM:
            pytest.skip("pydicom not installed")
        result = mod.inspect_file("/nonexistent/path/file.dcm")
        assert len(result.warnings) > 0

    def test_has_pydicom_flag_exists(self, mod):
        assert isinstance(mod.HAS_PYDICOM, bool)

    def test_require_pydicom_exists(self, mod):
        assert callable(mod._require_pydicom)

    def test_collect_dicom_files_exists(self, mod):
        assert callable(mod._collect_dicom_files)
