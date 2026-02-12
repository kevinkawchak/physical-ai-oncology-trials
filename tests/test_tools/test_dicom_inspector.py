"""Tests for tools/dicom-inspector/dicom_inspector.py

Covers InspectionResult dataclass and utility functions.
"""

from __future__ import annotations


class TestInspectionResult:
    def test_creation_defaults(self, dicom_mod):
        result = dicom_mod.InspectionResult(filepath="/tmp/test.dcm")
        assert result.filepath == "/tmp/test.dcm"
        assert result.is_valid is True
        assert result.phi_tags_present == []
        assert result.missing_required_tags == []

    def test_to_dict(self, dicom_mod):
        result = dicom_mod.InspectionResult(
            filepath="/tmp/test.dcm",
            modality="CT",
            sop_class="1.2.840.10008.5.1.4.1.1.2",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["filepath"] == "/tmp/test.dcm"
        assert d["modality"] == "CT"

    def test_with_phi_tags(self, dicom_mod):
        result = dicom_mod.InspectionResult(
            filepath="/tmp/phi.dcm",
            phi_tags_present=["PatientName", "PatientID"],
        )
        assert len(result.phi_tags_present) == 2

    def test_with_warnings(self, dicom_mod):
        result = dicom_mod.InspectionResult(
            filepath="/tmp/warn.dcm",
            warnings=["Missing StudyDate", "No pixel data"],
            is_valid=False,
        )
        assert result.is_valid is False
        assert len(result.warnings) == 2
