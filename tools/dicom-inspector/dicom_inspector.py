#!/usr/bin/env python3
"""DICOM Inspector: CLI for inspecting, validating, and auditing DICOM files
in oncology clinical trial imaging pipelines.

Checks de-identification status, validates required DICOM tags for trial
compliance, and generates batch reports across imaging directories.

Usage:
    python dicom_inspector.py inspect <file>
    python dicom_inspector.py audit-phi <directory>
    python dicom_inspector.py validate <directory> [--standard DICOM-RT]
    python dicom_inspector.py summarize <directory> [--output report.json]

Requirements:
    pydicom (listed in project requirements.txt)

Note: All illustrative parameters should be validated against your
institution's imaging protocols before clinical use.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

try:
    import pydicom
    from pydicom.errors import InvalidDicomError

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ---------------------------------------------------------------------------
# DICOM tag groups relevant to oncology trial compliance
# ---------------------------------------------------------------------------

# Tags that must be empty or absent after de-identification (DICOM PS3.15 E.1)
PHI_TAGS = {
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x0040): "PatientSex",
    (0x0010, 0x1000): "OtherPatientIDs",
    (0x0010, 0x1001): "OtherPatientNames",
    (0x0010, 0x1010): "PatientAge",
    (0x0010, 0x1040): "PatientAddress",
    (0x0010, 0x2154): "PatientTelephoneNumbers",
    (0x0008, 0x0050): "AccessionNumber",
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0081): "InstitutionAddress",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x1048): "PhysiciansOfRecord",
    (0x0008, 0x1050): "PerformingPhysicianName",
    (0x0008, 0x1070): "OperatorsName",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
}

# Minimum required tags for trial DICOM submission
REQUIRED_TRIAL_TAGS = {
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x0020): "StudyDate",
    (0x0020, 0x0010): "StudyID",
    (0x0020, 0x0011): "SeriesNumber",
    (0x0020, 0x0013): "InstanceNumber",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
}

# Additional tags for radiation therapy DICOM-RT objects
DICOM_RT_TAGS = {
    (0x3006, 0x0002): "StructureSetLabel",
    (0x3006, 0x0080): "ReferencedFrameOfReferenceSequence",
    (0x300A, 0x0010): "DoseReferenceSequence",
    (0x300C, 0x0002): "ReferencedRTPlanSequence",
}

# Modalities common in oncology imaging trials
ONCOLOGY_MODALITIES = {"CT", "MR", "PT", "NM", "US", "RTPLAN", "RTSTRUCT", "RTDOSE", "RTIMAGE", "REG", "SEG"}


@dataclass
class InspectionResult:
    """Result of a single DICOM file inspection."""

    filepath: str
    modality: str = ""
    sop_class: str = ""
    study_date: str = ""
    series_description: str = ""
    phi_tags_present: list = field(default_factory=list)
    missing_required_tags: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    is_valid: bool = True
    pixel_data_present: bool = False
    rows: int = 0
    columns: int = 0
    bits_allocated: int = 0
    transfer_syntax: str = ""

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "modality": self.modality,
            "sop_class": self.sop_class,
            "study_date": self.study_date,
            "series_description": self.series_description,
            "phi_tags_present": self.phi_tags_present,
            "missing_required_tags": self.missing_required_tags,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
            "pixel_data_present": self.pixel_data_present,
            "dimensions": f"{self.rows}x{self.columns}" if self.rows else "",
            "bits_allocated": self.bits_allocated,
            "transfer_syntax": self.transfer_syntax,
        }


def _require_pydicom():
    """Exit with message if pydicom is not installed."""
    if not HAS_PYDICOM:
        print("ERROR: pydicom is required. Install with: pip install pydicom", file=sys.stderr)
        sys.exit(1)


def _read_dicom(filepath: str) -> "pydicom.Dataset | None":
    """Read a DICOM file, returning None on failure."""
    try:
        return pydicom.dcmread(filepath, force=True)
    except (InvalidDicomError, Exception) as e:
        print(f"  WARNING: Cannot read {filepath}: {e}", file=sys.stderr)
        return None


def _collect_dicom_files(directory: str) -> list[str]:
    """Recursively collect DICOM files from a directory."""
    dicom_files = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            fpath = os.path.join(root, f)
            if f.lower().endswith((".dcm", ".dicom")):
                dicom_files.append(fpath)
            elif not f.startswith(".") and "." not in f:
                # DICOM files often lack extensions; attempt read
                dicom_files.append(fpath)
    return sorted(dicom_files)


def inspect_file(filepath: str) -> InspectionResult:
    """Inspect a single DICOM file and return structured metadata."""
    result = InspectionResult(filepath=filepath)
    ds = _read_dicom(filepath)
    if ds is None:
        result.is_valid = False
        result.warnings.append("File could not be parsed as DICOM")
        return result

    # Core metadata
    result.modality = getattr(ds, "Modality", "UNKNOWN")
    result.sop_class = str(getattr(ds, "SOPClassUID", ""))
    result.study_date = str(getattr(ds, "StudyDate", ""))
    result.series_description = getattr(ds, "SeriesDescription", "")
    result.pixel_data_present = hasattr(ds, "PixelData")
    result.rows = int(getattr(ds, "Rows", 0))
    result.columns = int(getattr(ds, "Columns", 0))
    result.bits_allocated = int(getattr(ds, "BitsAllocated", 0))

    # Transfer syntax
    if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "TransferSyntaxUID"):
        result.transfer_syntax = str(ds.file_meta.TransferSyntaxUID)

    # PHI check
    for tag, name in PHI_TAGS.items():
        if tag in ds:
            value = str(ds[tag].value).strip()
            if value and value.lower() not in ("", "none", "anonymous", "deidentified"):
                result.phi_tags_present.append(f"{name} ({tag}): present")

    # Required tag check
    for tag, name in REQUIRED_TRIAL_TAGS.items():
        if tag not in ds:
            result.missing_required_tags.append(name)

    # Modality-specific warnings
    if result.modality not in ONCOLOGY_MODALITIES:
        result.warnings.append(f"Modality '{result.modality}' not in standard oncology set")

    if result.modality in ("CT", "MR", "PT") and not result.pixel_data_present:
        result.warnings.append("Imaging modality without pixel data")

    return result


def cmd_inspect(args):
    """Handle 'inspect' subcommand: inspect a single DICOM file."""
    _require_pydicom()
    filepath = args.file
    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    result = inspect_file(filepath)
    _print_inspection(result)

    if args.output:
        _write_json(args.output, result.to_dict())
        print(f"\nReport written to {args.output}")


def cmd_audit_phi(args):
    """Handle 'audit-phi' subcommand: scan directory for PHI leaks."""
    _require_pydicom()
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    files = _collect_dicom_files(directory)
    print(f"Scanning {len(files)} files for PHI in {directory}...\n")

    phi_files = []
    clean_files = 0
    error_files = 0

    for fpath in files:
        result = inspect_file(fpath)
        if not result.is_valid:
            error_files += 1
            continue
        if result.phi_tags_present:
            phi_files.append(result)
        else:
            clean_files += 1

    # Summary
    print("=" * 60)
    print("PHI AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Total files scanned:     {len(files)}")
    print(f"  Clean (no PHI detected): {clean_files}")
    print(f"  PHI detected:            {len(phi_files)}")
    print(f"  Unreadable:              {error_files}")
    print()

    if phi_files:
        print("FILES WITH PHI TAGS:")
        print("-" * 60)
        for r in phi_files:
            rel = os.path.relpath(r.filepath, directory)
            print(f"  {rel}")
            for tag_info in r.phi_tags_present:
                print(f"    - {tag_info}")
        print()

    if args.output:
        report = {
            "audit_type": "phi_detection",
            "directory": directory,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "clean_files": clean_files,
            "phi_files": len(phi_files),
            "error_files": error_files,
            "phi_details": [r.to_dict() for r in phi_files],
        }
        _write_json(args.output, report)
        print(f"Report written to {args.output}")

    sys.exit(2 if phi_files else 0)


def cmd_validate(args):
    """Handle 'validate' subcommand: check DICOM compliance for trial submission."""
    _require_pydicom()
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    standard = args.standard
    files = _collect_dicom_files(directory)
    print(f"Validating {len(files)} files against {standard} standard...\n")

    compliant = 0
    non_compliant = []
    error_files = 0

    for fpath in files:
        result = inspect_file(fpath)
        if not result.is_valid:
            error_files += 1
            continue

        issues = list(result.missing_required_tags)

        # Additional RT-specific checks
        if standard == "DICOM-RT" and result.modality in ("RTPLAN", "RTSTRUCT", "RTDOSE"):
            ds = _read_dicom(fpath)
            if ds:
                for tag, name in DICOM_RT_TAGS.items():
                    if tag not in ds:
                        issues.append(f"RT-specific: {name}")

        if issues:
            non_compliant.append((fpath, issues))
        else:
            compliant += 1

    print("=" * 60)
    print(f"VALIDATION SUMMARY ({standard})")
    print("=" * 60)
    print(f"  Total files:    {len(files)}")
    print(f"  Compliant:      {compliant}")
    print(f"  Non-compliant:  {len(non_compliant)}")
    print(f"  Unreadable:     {error_files}")
    print()

    if non_compliant:
        print("NON-COMPLIANT FILES:")
        print("-" * 60)
        for fpath, issues in non_compliant[:20]:  # Limit output
            rel = os.path.relpath(fpath, directory)
            print(f"  {rel}")
            for issue in issues:
                print(f"    - Missing: {issue}")
        if len(non_compliant) > 20:
            print(f"  ... and {len(non_compliant) - 20} more files")
        print()

    if args.output:
        report = {
            "validation_type": "dicom_compliance",
            "standard": standard,
            "directory": directory,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "compliant": compliant,
            "non_compliant": len(non_compliant),
            "error_files": error_files,
            "issues": [{"file": f, "missing": i} for f, i in non_compliant],
        }
        _write_json(args.output, report)
        print(f"Report written to {args.output}")

    sys.exit(1 if non_compliant else 0)


def cmd_summarize(args):
    """Handle 'summarize' subcommand: generate study-level summary."""
    _require_pydicom()
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    files = _collect_dicom_files(directory)
    print(f"Summarizing {len(files)} files from {directory}...\n")

    modality_counts: dict[str, int] = {}
    series_map: dict[str, list[str]] = {}
    study_dates: set[str] = set()
    total_phi = 0
    total_errors = 0

    for fpath in files:
        result = inspect_file(fpath)
        if not result.is_valid:
            total_errors += 1
            continue

        modality_counts[result.modality] = modality_counts.get(result.modality, 0) + 1
        if result.study_date:
            study_dates.add(result.study_date)
        if result.phi_tags_present:
            total_phi += 1

        series_key = result.series_description or "Unnamed"
        if series_key not in series_map:
            series_map[series_key] = []
        series_map[series_key].append(result.modality)

    print("=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"  Total files:     {len(files)}")
    print(f"  Readable:        {len(files) - total_errors}")
    print(f"  Unreadable:      {total_errors}")
    print(f"  Files with PHI:  {total_phi}")
    print()

    print("MODALITY DISTRIBUTION:")
    for mod, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        print(f"  {mod:12s} {count:6d} files")
    print()

    print(f"STUDY DATES: {', '.join(sorted(study_dates)) if study_dates else 'None found'}")
    print()

    print(f"SERIES ({len(series_map)} unique):")
    for series, mods in sorted(series_map.items()):
        print(f"  {series}: {len(mods)} files ({mods[0]})")
    print()

    if args.output:
        report = {
            "summary_type": "study_overview",
            "directory": directory,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "readable_files": len(files) - total_errors,
            "files_with_phi": total_phi,
            "modality_counts": modality_counts,
            "study_dates": sorted(study_dates),
            "series_count": len(series_map),
            "series": {k: len(v) for k, v in series_map.items()},
        }
        _write_json(args.output, report)
        print(f"Report written to {args.output}")


def _print_inspection(result: InspectionResult):
    """Pretty-print a single file inspection result."""
    print("=" * 60)
    print(f"DICOM INSPECTION: {os.path.basename(result.filepath)}")
    print("=" * 60)
    print(f"  File:               {result.filepath}")
    print(f"  Valid DICOM:        {'Yes' if result.is_valid else 'No'}")
    print(f"  Modality:           {result.modality}")
    print(f"  SOP Class UID:      {result.sop_class}")
    print(f"  Study Date:         {result.study_date}")
    print(f"  Series Description: {result.series_description}")
    print(f"  Transfer Syntax:    {result.transfer_syntax}")
    print(f"  Pixel Data:         {'Yes' if result.pixel_data_present else 'No'}")
    if result.rows:
        print(f"  Dimensions:         {result.rows} x {result.columns}")
        print(f"  Bits Allocated:     {result.bits_allocated}")
    print()

    if result.phi_tags_present:
        print("  PHI TAGS DETECTED:")
        for tag_info in result.phi_tags_present:
            print(f"    - {tag_info}")
    else:
        print("  PHI STATUS: Clean (no identifiable PHI tags detected)")
    print()

    if result.missing_required_tags:
        print("  MISSING REQUIRED TAGS:")
        for tag_name in result.missing_required_tags:
            print(f"    - {tag_name}")
    else:
        print("  REQUIRED TAGS: All present")
    print()

    if result.warnings:
        print("  WARNINGS:")
        for w in result.warnings:
            print(f"    - {w}")


def _write_json(filepath: str, data: dict):
    """Write data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        prog="dicom_inspector",
        description="DICOM Inspector: Inspect, validate, and audit DICOM files for oncology trial compliance.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a single DICOM file")
    p_inspect.add_argument("file", help="Path to DICOM file")
    p_inspect.add_argument("--output", "-o", help="Write JSON report to file")

    # audit-phi
    p_audit = subparsers.add_parser("audit-phi", help="Scan directory for PHI leaks in DICOM headers")
    p_audit.add_argument("directory", help="Directory containing DICOM files")
    p_audit.add_argument("--output", "-o", help="Write JSON report to file")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate DICOM compliance for trial submission")
    p_validate.add_argument("directory", help="Directory containing DICOM files")
    p_validate.add_argument(
        "--standard",
        default="DICOM-BASE",
        choices=["DICOM-BASE", "DICOM-RT"],
        help="Validation standard (default: DICOM-BASE)",
    )
    p_validate.add_argument("--output", "-o", help="Write JSON report to file")

    # summarize
    p_summarize = subparsers.add_parser("summarize", help="Generate study-level summary of DICOM directory")
    p_summarize.add_argument("directory", help="Directory containing DICOM files")
    p_summarize.add_argument("--output", "-o", help="Write JSON report to file")

    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "audit-phi":
        cmd_audit_phi(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
