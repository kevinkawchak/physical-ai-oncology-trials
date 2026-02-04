# PHI/PII Management for AI Oncology Trials

*Detection, classification, and handling of Protected Health Information in clinical AI workflows*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

Physical AI oncology trials process patient data across multiple modalities: clinical records, medical imaging (DICOM), genomic sequencing, robotic telemetry, and digital health technology (DHT) streams. This module provides automated PHI/PII detection and classification to prevent unauthorized exposure throughout AI model development and deployment.

**Key Capabilities**:
- Multi-modal PHI scanning (text, structured data, DICOM headers, genomic metadata)
- HIPAA 18-identifier classification with confidence scoring
- Continuous monitoring for PHI leakage in AI training pipelines
- Integration with Microsoft Presidio for NLP-based detection
- DICOM header sanitization for medical imaging workflows

---

## PHI Categories in Oncology AI

| Data Source | PHI Risk | Common Identifiers | Detection Approach |
|-------------|----------|-------------------|-------------------|
| EHR/FHIR records | High | Names, MRN, dates, SSN | NER + regex |
| DICOM imaging | High | Patient name, institution, dates in headers | DICOM tag parsing |
| Clinical notes | High | Names, locations, dates, providers | NLP + pattern matching |
| Genomic data | Medium | Sample IDs, institution codes, rare variants | Pattern matching + linkage analysis |
| Robotic telemetry | Low-Medium | Timestamps, facility IDs, operator IDs | Metadata parsing |
| Wearable/DHT data | Medium | Device IDs, IP addresses, GPS coordinates | Regex + geofencing |

---

## Quick Start

```python
from privacy.phi_pii_management.phi_detector import PHIDetector, PHICategory

# Initialize detector with oncology-specific patterns
detector = PHIDetector(
    detection_mode="comprehensive",
    data_sources=["clinical_notes", "dicom_headers", "genomic_metadata"],
    custom_patterns={
        "mrn": r"MRN[-:]?\s*\d{6,10}",
        "accession": r"ACC[-:]?\s*\d{8,12}"
    }
)

# Scan a clinical trial dataset
result = detector.scan_dataset(
    dataset_path="trial_data/enrollment_records/",
    output_report="phi_scan_report.json"
)

# Review findings by category
for category in PHICategory:
    findings = result.get_findings_by_category(category)
    if findings:
        print(f"{category.value}: {len(findings)} instances found")
```

---

## References

- [Microsoft Presidio](https://github.com/microsoft/presidio) - PII detection and anonymization SDK
- [HHS Guidance on PHI](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
- [DICOM Standard PS3.15](https://dicom.nema.org/medical/dicom/current/output/html/part15.html) - Security and system management profiles

---

*This module is part of the Privacy Framework for Physical AI Oncology Trials.*
