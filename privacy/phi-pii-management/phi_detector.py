"""
=============================================================================
PHI/PII Detection Pipeline for Physical AI Oncology Trials
=============================================================================

Automated detection and classification of Protected Health Information (PHI)
and Personally Identifiable Information (PII) across clinical trial data
streams including structured EHR data, clinical notes, DICOM imaging
headers, and genomic metadata.

CLINICAL CONTEXT:
-----------------
Physical AI oncology trials generate multi-modal patient data that flows
through AI training pipelines, robotic systems, and multi-site coordination
platforms. PHI must be detected and flagged before data enters any system
that lacks appropriate safeguards.

FRAMEWORK REQUIREMENTS:
-----------------------
Required:
    - Python 3.10+
    - pydicom 2.4.0+ (DICOM header parsing)
    - re (standard library, regex patterns)

Optional:
    - presidio-analyzer (Microsoft Presidio for NLP-based detection)
    - presidio-anonymizer (for inline redaction)
    - spacy (NLP backend for Presidio)

REFERENCES:
    - HIPAA Safe Harbor: 45 CFR 164.514(b)(2) - 18 identifiers
    - Microsoft Presidio: https://github.com/microsoft/presidio
    - DICOM PS3.15: Security and System Management Profiles

DISCLAIMER: RESEARCH USE ONLY. Not approved for clinical decision-making.
    Requires institutional validation and regulatory review before deployment.

LICENSE: MIT
VERSION: 1.0.0
LAST UPDATED: February 2026
=============================================================================
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: PHI CATEGORY DEFINITIONS
# =============================================================================


class PHICategory(Enum):
    """
    HIPAA Safe Harbor 18 identifiers (45 CFR 164.514(b)(2)).

    Each category maps to a specific identifier type that must be
    removed or generalized for de-identification.
    """

    NAME = "names"
    GEOGRAPHIC = "geographic_data"
    DATE = "dates"
    PHONE = "phone_numbers"
    FAX = "fax_numbers"
    EMAIL = "email_addresses"
    SSN = "social_security_numbers"
    MRN = "medical_record_numbers"
    HEALTH_PLAN = "health_plan_beneficiary_numbers"
    ACCOUNT = "account_numbers"
    LICENSE = "certificate_license_numbers"
    VEHICLE = "vehicle_identifiers"
    DEVICE = "device_identifiers"
    URL = "web_urls"
    IP_ADDRESS = "ip_addresses"
    BIOMETRIC = "biometric_identifiers"
    PHOTO = "full_face_photographs"
    UNIQUE_CODE = "unique_identifying_codes"


class RiskLevel(Enum):
    """Risk level classification for PHI findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PHIFinding:
    """Individual PHI detection finding."""

    phi_type: PHICategory
    location: str
    value_preview: str  # Redacted preview for logging
    confidence: float
    risk_level: RiskLevel
    source_file: str = ""
    line_number: int = 0
    context: str = ""

    def to_dict(self) -> dict:
        """Convert finding to dictionary for reporting."""
        return {
            "phi_type": self.phi_type.value,
            "location": self.location,
            "value_preview": self.value_preview,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }


@dataclass
class ScanResult:
    """Results from a PHI detection scan."""

    scan_id: str
    timestamp: str
    dataset_path: str
    total_findings: int = 0
    findings: list = field(default_factory=list)
    risk_assessment: str = "not_assessed"
    files_scanned: int = 0
    errors: list = field(default_factory=list)

    def get_findings_by_category(self, category: PHICategory) -> list:
        """Filter findings by PHI category."""
        return [f for f in self.findings if f.phi_type == category]

    def get_findings_by_risk(self, risk_level: RiskLevel) -> list:
        """Filter findings by risk level."""
        return [f for f in self.findings if f.risk_level == risk_level]

    def to_dict(self) -> dict:
        """Convert scan result to dictionary."""
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "total_findings": self.total_findings,
            "risk_assessment": self.risk_assessment,
            "files_scanned": self.files_scanned,
            "findings_by_category": {
                cat.value: len(self.get_findings_by_category(cat))
                for cat in PHICategory
                if self.get_findings_by_category(cat)
            },
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
        }


# =============================================================================
# SECTION 2: DETECTION PATTERNS
# =============================================================================

# Regex patterns for HIPAA Safe Harbor identifiers
# These patterns are designed for clinical trial data contexts
PHI_PATTERNS: dict[PHICategory, list[dict[str, Any]]] = {
    PHICategory.SSN: [
        {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "description": "SSN format XXX-XX-XXXX", "confidence": 0.95},
        {"pattern": r"\bSSN\s*[:=]\s*\d{9}\b", "description": "SSN with label", "confidence": 0.98},
    ],
    PHICategory.PHONE: [
        {"pattern": r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "description": "US phone number", "confidence": 0.85}
    ],
    PHICategory.EMAIL: [
        {
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "description": "Email address",
            "confidence": 0.95,
        }
    ],
    PHICategory.MRN: [
        {
            "pattern": r"\bMRN\s*[-:=]\s*\d{6,10}\b",
            "description": "Medical Record Number with label",
            "confidence": 0.95,
        },
        {
            "pattern": r"\bPatient\s*ID\s*[-:=]\s*[A-Z0-9]{6,12}\b",
            "description": "Patient ID with label",
            "confidence": 0.90,
        },
    ],
    PHICategory.DATE: [
        {"pattern": r"\b\d{1,2}/\d{1,2}/\d{4}\b", "description": "Date MM/DD/YYYY", "confidence": 0.80},
        {
            "pattern": r"\b\d{4}-\d{2}-\d{2}\b",
            "description": "Date YYYY-MM-DD (ISO)",
            "confidence": 0.75,  # Lower confidence due to common non-PHI use
        },
        {
            "pattern": r"\b(?:DOB|Date\s*of\s*Birth)\s*[-:=]\s*\S+",
            "description": "Date of birth with label",
            "confidence": 0.98,
        },
    ],
    PHICategory.IP_ADDRESS: [
        {"pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "description": "IPv4 address", "confidence": 0.85}
    ],
    PHICategory.URL: [
        {
            "pattern": r"https?://[^\s<>\"']+",
            "description": "Web URL",
            "confidence": 0.70,  # Many URLs are non-PHI
        }
    ],
    PHICategory.GEOGRAPHIC: [
        {
            "pattern": r"\b\d{5}(?:-\d{4})?\b",
            "description": "ZIP code (5 or 9 digit)",
            "confidence": 0.60,  # Context-dependent
        },
        {
            "pattern": r"\b(?:Address|Street|Ave|Blvd|Dr|Rd)\s*[-:=]?\s*.+",
            "description": "Street address with label",
            "confidence": 0.85,
        },
    ],
    PHICategory.ACCOUNT: [
        {
            "pattern": r"\b(?:Account|Acct)\s*#?\s*[-:=]\s*\d{8,16}\b",
            "description": "Account number with label",
            "confidence": 0.90,
        }
    ],
    PHICategory.DEVICE: [
        {
            "pattern": r"\b(?:UDI|Device\s*ID|Serial)\s*[-:=]\s*[A-Z0-9]{8,20}\b",
            "description": "Device identifier with label",
            "confidence": 0.90,
        }
    ],
}

# DICOM tags that contain PHI
# Reference: DICOM PS3.15, Table E.1-1
PHI_DICOM_TAGS: dict[str, dict[str, Any]] = {
    "(0010,0010)": {"name": "PatientName", "category": PHICategory.NAME, "action": "remove"},
    "(0010,0020)": {"name": "PatientID", "category": PHICategory.MRN, "action": "remove"},
    "(0010,0030)": {"name": "PatientBirthDate", "category": PHICategory.DATE, "action": "remove"},
    "(0010,0040)": {"name": "PatientSex", "category": PHICategory.UNIQUE_CODE, "action": "generalize"},
    "(0010,1000)": {"name": "OtherPatientIDs", "category": PHICategory.MRN, "action": "remove"},
    "(0010,1001)": {"name": "OtherPatientNames", "category": PHICategory.NAME, "action": "remove"},
    "(0008,0050)": {"name": "AccessionNumber", "category": PHICategory.ACCOUNT, "action": "remove"},
    "(0008,0080)": {"name": "InstitutionName", "category": PHICategory.GEOGRAPHIC, "action": "remove"},
    "(0008,0081)": {"name": "InstitutionAddress", "category": PHICategory.GEOGRAPHIC, "action": "remove"},
    "(0008,0090)": {"name": "ReferringPhysicianName", "category": PHICategory.NAME, "action": "remove"},
    "(0008,1050)": {"name": "PerformingPhysicianName", "category": PHICategory.NAME, "action": "remove"},
    "(0008,0020)": {"name": "StudyDate", "category": PHICategory.DATE, "action": "generalize"},
    "(0008,0030)": {"name": "StudyTime", "category": PHICategory.DATE, "action": "remove"},
    "(0020,0010)": {"name": "StudyID", "category": PHICategory.UNIQUE_CODE, "action": "remove"},
}


# =============================================================================
# SECTION 3: PHI DETECTOR
# =============================================================================


class PHIDetector:
    """
    Automated PHI/PII detection for clinical trial data.

    Scans structured data, clinical text, DICOM headers, and genomic
    metadata for HIPAA Safe Harbor identifiers. Produces detailed
    findings reports with confidence scores and risk assessments.

    PHI DETECTION STRATEGY:
    ----------------------
    1. Pattern-based detection (regex) for structured identifiers
    2. DICOM tag inspection for imaging data
    3. NLP-based detection for unstructured clinical text (via Presidio)
    4. Metadata analysis for genomic and telemetry data
    """

    def __init__(
        self,
        detection_mode: str = "comprehensive",
        data_sources: list[str] | None = None,
        custom_patterns: dict[str, str] | None = None,
        confidence_threshold: float = 0.7,
        use_presidio: bool = False,
    ):
        """
        Initialize PHI detector.

        Args:
            detection_mode: "quick" (regex only), "comprehensive" (all methods)
            data_sources: Data types to scan
            custom_patterns: Additional regex patterns {name: pattern}
            confidence_threshold: Minimum confidence to report a finding
            use_presidio: Enable Microsoft Presidio NLP-based detection
        """
        self.detection_mode = detection_mode
        self.data_sources = data_sources or ["clinical_notes", "dicom_headers"]
        self.confidence_threshold = confidence_threshold
        self.use_presidio = use_presidio
        self._custom_patterns = custom_patterns or {}
        self._presidio_analyzer = None

        if use_presidio:
            self._init_presidio()

        logger.info(
            f"PHIDetector initialized: mode={detection_mode}, sources={data_sources}, threshold={confidence_threshold}"
        )

    def _init_presidio(self):
        """Initialize Microsoft Presidio analyzer if available."""
        try:
            from presidio_analyzer import AnalyzerEngine

            self._presidio_analyzer = AnalyzerEngine()
            logger.info("Microsoft Presidio analyzer initialized")
        except ImportError:
            logger.warning(
                "presidio-analyzer not installed. Install with: pip install presidio-analyzer presidio-anonymizer"
            )
            self.use_presidio = False

    def scan_dataset(self, dataset_path: str, output_report: str | None = None, recursive: bool = True) -> ScanResult:
        """
        Scan a dataset directory for PHI/PII.

        Args:
            dataset_path: Path to dataset directory or file
            output_report: Path for JSON report output
            recursive: Scan subdirectories

        Returns:
            ScanResult with all findings
        """
        scan_id = f"PHI-SCAN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting PHI scan {scan_id}: {dataset_path}")

        result = ScanResult(scan_id=scan_id, timestamp=datetime.now().isoformat(), dataset_path=dataset_path)

        path = Path(dataset_path)

        if path.is_file():
            self._scan_file(path, result)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    self._scan_file(file_path, result)
        else:
            result.errors.append(f"Path not found: {dataset_path}")
            logger.error(f"Dataset path not found: {dataset_path}")

        # Calculate risk assessment
        result.total_findings = len(result.findings)
        result.risk_assessment = self._assess_overall_risk(result)

        # Write report if requested
        if output_report:
            self._write_report(result, output_report)

        logger.info(
            f"Scan {scan_id} complete: {result.total_findings} findings, "
            f"risk={result.risk_assessment}, files={result.files_scanned}"
        )

        return result

    def scan_text(self, text: str, source_label: str = "inline") -> list[PHIFinding]:
        """
        Scan a text string for PHI/PII.

        Args:
            text: Text to scan
            source_label: Label for the text source

        Returns:
            List of PHI findings
        """
        findings = []

        # Regex-based detection
        for category, patterns in PHI_PATTERNS.items():
            for pattern_def in patterns:
                matches = re.finditer(pattern_def["pattern"], text, re.IGNORECASE)
                for match in matches:
                    confidence = pattern_def["confidence"]
                    if confidence >= self.confidence_threshold:
                        # Redact the actual value for safe logging
                        value = match.group()
                        redacted = value[:2] + "***" + value[-1:] if len(value) > 3 else "***"

                        findings.append(
                            PHIFinding(
                                phi_type=category,
                                location=f"char {match.start()}-{match.end()}",
                                value_preview=redacted,
                                confidence=confidence,
                                risk_level=self._categorize_risk(category, confidence),
                                source_file=source_label,
                                context=pattern_def["description"],
                            )
                        )

        # Custom pattern detection
        for name, pattern in self._custom_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group()
                redacted = value[:2] + "***" + value[-1:] if len(value) > 3 else "***"
                findings.append(
                    PHIFinding(
                        phi_type=PHICategory.UNIQUE_CODE,
                        location=f"char {match.start()}-{match.end()}",
                        value_preview=redacted,
                        confidence=0.85,
                        risk_level=RiskLevel.MEDIUM,
                        source_file=source_label,
                        context=f"Custom pattern: {name}",
                    )
                )

        # Presidio NLP-based detection
        if self.use_presidio and self._presidio_analyzer:
            presidio_findings = self._run_presidio(text, source_label)
            findings.extend(presidio_findings)

        return findings

    def scan_dicom_headers(self, dicom_path: str) -> list[PHIFinding]:
        """
        Scan DICOM file headers for PHI.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            List of PHI findings from DICOM tags
        """
        findings = []

        try:
            import pydicom

            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

            for tag_str, tag_info in PHI_DICOM_TAGS.items():
                # Parse tag group and element
                group, element = tag_str.strip("()").split(",")
                tag = (int(group, 16), int(element, 16))

                if tag in ds:
                    value = str(ds[tag].value)
                    if value and value.strip():
                        redacted = value[:2] + "***" if len(value) > 2 else "***"
                        findings.append(
                            PHIFinding(
                                phi_type=tag_info["category"],
                                location=f"DICOM tag {tag_str} ({tag_info['name']})",
                                value_preview=redacted,
                                confidence=0.99,
                                risk_level=RiskLevel.HIGH,
                                source_file=dicom_path,
                                context=f"DICOM {tag_info['name']}: action={tag_info['action']}",
                            )
                        )

        except ImportError:
            logger.warning("pydicom not installed. Skipping DICOM header scan.")
        except Exception as e:
            logger.error(f"Error scanning DICOM {dicom_path}: {e}")

        return findings

    def _scan_file(self, file_path: Path, result: ScanResult):
        """Scan a single file for PHI."""
        result.files_scanned += 1
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".dcm" or suffix == ".dicom":
                findings = self.scan_dicom_headers(str(file_path))
                result.findings.extend(findings)

            elif suffix in (".txt", ".csv", ".json", ".xml", ".hl7", ".fhir"):
                text = file_path.read_text(encoding="utf-8", errors="replace")
                findings = self.scan_text(text, str(file_path))
                result.findings.extend(findings)

            elif suffix in (".tsv", ".vcf"):
                # Genomic data formats
                text = file_path.read_text(encoding="utf-8", errors="replace")
                findings = self.scan_text(text, str(file_path))
                result.findings.extend(findings)

        except Exception as e:
            result.errors.append(f"Error scanning {file_path}: {e}")
            logger.error(f"Error scanning {file_path}: {e}")

    def _run_presidio(self, text: str, source_label: str) -> list[PHIFinding]:
        """Run Microsoft Presidio NLP-based detection."""
        findings = []

        if not self._presidio_analyzer:
            return findings

        # Map Presidio entity types to PHI categories
        presidio_to_phi = {
            "PERSON": PHICategory.NAME,
            "PHONE_NUMBER": PHICategory.PHONE,
            "EMAIL_ADDRESS": PHICategory.EMAIL,
            "LOCATION": PHICategory.GEOGRAPHIC,
            "DATE_TIME": PHICategory.DATE,
            "IP_ADDRESS": PHICategory.IP_ADDRESS,
            "URL": PHICategory.URL,
            "US_SSN": PHICategory.SSN,
            "MEDICAL_LICENSE": PHICategory.LICENSE,
        }

        try:
            results = self._presidio_analyzer.analyze(
                text=text, language="en", score_threshold=self.confidence_threshold
            )

            for r in results:
                phi_category = presidio_to_phi.get(r.entity_type, PHICategory.UNIQUE_CODE)
                value = text[r.start : r.end]
                redacted = value[:2] + "***" if len(value) > 2 else "***"

                findings.append(
                    PHIFinding(
                        phi_type=phi_category,
                        location=f"char {r.start}-{r.end}",
                        value_preview=redacted,
                        confidence=r.score,
                        risk_level=self._categorize_risk(phi_category, r.score),
                        source_file=source_label,
                        context=f"Presidio: {r.entity_type}",
                    )
                )

        except Exception as e:
            logger.error(f"Presidio analysis error: {e}")

        return findings

    def _categorize_risk(self, category: PHICategory, confidence: float) -> RiskLevel:
        """Categorize risk level based on PHI type and confidence."""
        high_risk_categories = {
            PHICategory.SSN,
            PHICategory.NAME,
            PHICategory.MRN,
            PHICategory.HEALTH_PLAN,
            PHICategory.BIOMETRIC,
        }
        medium_risk_categories = {
            PHICategory.DATE,
            PHICategory.PHONE,
            PHICategory.EMAIL,
            PHICategory.GEOGRAPHIC,
            PHICategory.ACCOUNT,
        }

        if category in high_risk_categories and confidence > 0.8:
            return RiskLevel.CRITICAL
        elif category in high_risk_categories:
            return RiskLevel.HIGH
        elif category in medium_risk_categories and confidence > 0.8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _assess_overall_risk(self, result: ScanResult) -> str:
        """Assess overall risk level of scan results."""
        if not result.findings:
            return "clean"

        critical_count = len(result.get_findings_by_risk(RiskLevel.CRITICAL))
        high_count = len(result.get_findings_by_risk(RiskLevel.HIGH))

        if critical_count > 0:
            return "critical"
        elif high_count > 5:
            return "high"
        elif high_count > 0:
            return "medium"
        else:
            return "low"

    def _write_report(self, result: ScanResult, output_path: str):
        """Write scan results to JSON report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"PHI scan report written to {output_path}")


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================


def run_phi_detection_demo():
    """
    Demonstrate PHI detection capabilities.

    This function shows how to scan clinical trial data for
    PHI/PII using pattern-based and NLP-based detection.
    """
    logger.info("=" * 60)
    logger.info("PHI/PII DETECTION PIPELINE DEMO")
    logger.info("=" * 60)

    # Initialize detector
    detector = PHIDetector(
        detection_mode="comprehensive",
        confidence_threshold=0.7,
        custom_patterns={"mrn_alt": r"MRN[-:]?\s*\d{6,10}", "accession": r"ACC[-:]?\s*\d{8,12}"},
    )

    # Example clinical note with embedded PHI
    sample_note = """
    Patient: John Smith (MRN: 12345678)
    DOB: 03/15/1958
    SSN: 123-45-6789
    Phone: (555) 123-4567
    Email: john.smith@email.com

    Assessment: 62-year-old male presenting with Stage IIIA NSCLC.
    CT scan performed at Memorial Hospital, 123 Main St, New York, NY 10001.
    Referred by Dr. Jane Williams for clinical trial enrollment.

    Treatment plan discussed with patient at visit on 01/15/2026.
    Next follow-up scheduled for 02/15/2026.
    """

    # Scan the clinical note
    findings = detector.scan_text(sample_note, "sample_clinical_note")

    print("\nPHI Detection Results")
    print("-" * 60)
    print(f"Total findings: {len(findings)}")
    print()

    for finding in findings:
        print(
            f"  [{finding.risk_level.value.upper():8s}] "
            f"{finding.phi_type.value:30s} | "
            f"confidence={finding.confidence:.2f} | "
            f"{finding.value_preview}"
        )

    print()
    print(f"Categories detected: {len(set(f.phi_type for f in findings))}")
    print(f"Critical findings: {len([f for f in findings if f.risk_level == RiskLevel.CRITICAL])}")
    print(f"High-risk findings: {len([f for f in findings if f.risk_level == RiskLevel.HIGH])}")

    return {"findings_count": len(findings), "status": "demo_complete"}


if __name__ == "__main__":
    result = run_phi_detection_demo()
    print(f"\nDemo completed: {result}")
