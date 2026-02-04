# Privacy Framework for Physical AI Oncology Trials

*HIPAA-compliant patient data protection for AI-enabled clinical trial workflows (February 2026)*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Executive Summary

Physical AI oncology clinical trials generate sensitive patient data across imaging, genomic, robotic telemetry, and clinical record systems. This directory provides production-ready tools and frameworks for protecting Protected Health Information (PHI) and Personally Identifiable Information (PII) throughout the trial lifecycle, from enrollment through data retention and destruction.

**Key Capabilities**:
- PHI/PII detection and classification across clinical data streams
- HIPAA-compliant de-identification (Safe Harbor and Expert Determination methods)
- Role-based access control with audit trails for 21 CFR Part 11
- Automated breach detection, response, and regulatory notification workflows
- Data Use Agreement generation for multi-site AI research collaborations

---

## Regulatory Context

| Regulation | Jurisdiction | Key Requirements | Status |
|------------|-------------|------------------|--------|
| HIPAA Privacy Rule | U.S. | PHI use/disclosure controls (45 CFR 164) | Active |
| HIPAA Security Rule | U.S. | ePHI safeguards, encryption, access control | NPRM Jan 2025 |
| 21 CFR Part 11 | U.S. (FDA) | Electronic records, audit trails, e-signatures | Active |
| EU GDPR | EU/EEA | Data subject rights, lawful processing basis | Active |
| EU AI Act | EU/EEA | High-risk AI system transparency requirements | Phased to Aug 2027 |
| NIST Privacy Framework 1.1 | U.S. | AI risk management for privacy | Draft Apr 2025 |
| ICH E6(R3) | International | Digital technology data governance provisions | Effective Sep 2025 |

---

## Directory Structure

```
privacy/
├── README.md                          # This file
│
├── phi-pii-management/                # PHI/PII detection and classification
│   ├── README.md                      # PHI/PII handling overview
│   └── phi_detector.py                # Automated PHI/PII detection pipeline
│
├── de-identification/                 # HIPAA-compliant de-identification
│   ├── README.md                      # De-identification methods overview
│   └── deidentification_pipeline.py   # Safe Harbor & Expert Determination tools
│
├── access-control/                    # Role-based access control framework
│   ├── README.md                      # Access control overview
│   └── access_control_manager.py      # RBAC with 21 CFR Part 11 audit trails
│
├── breach-response/                   # Breach detection and response
│   ├── README.md                      # Breach response protocol overview
│   └── breach_response_protocol.py    # Automated incident response workflows
│
└── dua-templates/                     # Data Use Agreement management
    ├── README.md                      # DUA framework overview
    └── dua_generator.py               # Multi-site DUA template generator
```

---

## Quick Start

### 1. Detect PHI/PII in Clinical Data

```python
from privacy.phi_pii_management.phi_detector import PHIDetector

detector = PHIDetector(
    detection_mode="comprehensive",
    data_sources=["clinical_notes", "dicom_headers", "genomic_metadata"]
)

# Scan a clinical dataset for PHI
scan_result = detector.scan_dataset(
    dataset_path="trial_data/enrollment_records/",
    output_report="phi_scan_report.json"
)

print(f"PHI instances found: {scan_result.total_findings}")
print(f"Risk level: {scan_result.risk_assessment}")
for finding in scan_result.findings[:5]:
    print(f"  {finding.phi_type}: {finding.location} (confidence: {finding.confidence:.2f})")
```

### 2. De-identify Patient Data

```python
from privacy.de_identification.deidentification_pipeline import DeidentificationPipeline

pipeline = DeidentificationPipeline(
    method="safe_harbor",       # or "expert_determination"
    hipaa_identifiers="all_18", # Remove all 18 HIPAA identifiers
    preserve_clinical_utility=True
)

# De-identify a clinical trial dataset
result = pipeline.deidentify(
    input_path="trial_data/raw_patient_records/",
    output_path="trial_data/deidentified/",
    data_types=["structured_ehr", "clinical_notes", "dicom_images"]
)

print(f"Records processed: {result.records_processed}")
print(f"Identifiers removed: {result.identifiers_removed}")
print(f"Re-identification risk: {result.residual_risk:.4f}")
```

### 3. Configure Access Controls

```python
from privacy.access_control.access_control_manager import AccessControlManager

acm = AccessControlManager(
    compliance_framework="21_cfr_part_11",
    audit_enabled=True
)

# Define trial-specific roles
acm.define_role("principal_investigator", permissions=[
    "read_phi", "write_clinical_data", "export_deidentified",
    "approve_enrollment", "view_audit_logs"
])

acm.define_role("data_analyst", permissions=[
    "read_deidentified", "run_queries", "export_aggregated"
])

acm.define_role("ai_system", permissions=[
    "read_deidentified", "write_model_outputs"
])

# Assign users and verify access
acm.assign_role(user_id="PI-001", role="principal_investigator")
access = acm.check_access(user_id="PI-001", resource="patient_records", action="read_phi")
print(f"Access granted: {access.granted}, Reason: {access.reason}")
```

### 4. Generate Data Use Agreements

```python
from privacy.dua_templates.dua_generator import DUAGenerator

generator = DUAGenerator(
    template="multi_site_ai_research",
    jurisdiction="us_hipaa"
)

dua = generator.generate(
    data_provider="Memorial Sloan Kettering Cancer Center",
    data_recipient="Physical AI Oncology Consortium",
    data_description="De-identified CT imaging and treatment outcomes for AI model training",
    permitted_uses=["model_training", "validation", "publication"],
    retention_period_years=7,
    security_requirements=["encryption_at_rest", "encryption_in_transit", "mfa"]
)

dua.export("agreements/msk_consortium_dua_2026.pdf")
print(f"DUA generated: {dua.agreement_id}")
```

---

## Privacy Architecture for AI Oncology Trials

### Data Flow with Privacy Controls

```
Patient Data Sources          Privacy Layer              AI/Research Systems
─────────────────          ──────────────              ───────────────────
                           ┌──────────────┐
 EHR/FHIR ──────────────→ │ PHI Detector │
                           │  & Scanner   │
 DICOM Imaging ──────────→ │              │──→ De-identification ──→ AI Model Training
                           │ Access       │         Pipeline
 Genomic Data ───────────→ │ Control      │                         Federated Learning
                           │ Manager      │──→ Limited Data Set ──→
 Robotic Telemetry ──────→ │              │                         Multi-site Analysis
                           │ Audit Trail  │──→ Aggregated Stats ──→
 Wearable/DHT Data ─────→ │              │                         Publication
                           └──────────────┘
                                 │
                           ┌──────────────┐
                           │ Breach       │
                           │ Response     │
                           │ Protocol     │
                           └──────────────┘
```

### Privacy-Preserving AI Approaches

| Approach | Description | Clinical Application |
|----------|-------------|---------------------|
| Federated Learning | Train models across sites without moving data | Multi-institutional tumor modeling |
| Differential Privacy | Add calibrated noise to protect individuals | Population-level outcome analysis |
| Secure Multi-Party Computation | Joint computation without revealing inputs | Cross-site treatment comparison |
| Homomorphic Encryption | Compute on encrypted data | Cloud-based model inference |
| Synthetic Data Generation | Create privacy-safe artificial datasets | Algorithm development and testing |

---

## HIPAA Identifier Reference

The 18 HIPAA Safe Harbor identifiers that must be removed for de-identification (45 CFR 164.514(b)(2)):

| # | Identifier | Detection Method | Oncology Context |
|---|-----------|-----------------|------------------|
| 1 | Names | NER, pattern matching | Patient, physician, caregiver names |
| 2 | Geographic data | Regex, geocoding | Treatment facility locations |
| 3 | Dates (except year) | Regex, NLP | Diagnosis, treatment, scan dates |
| 4 | Phone numbers | Regex | Patient contact information |
| 5 | Fax numbers | Regex | Facility communication |
| 6 | Email addresses | Regex | Patient/provider email |
| 7 | SSN | Regex | Insurance processing |
| 8 | Medical record numbers | Pattern matching | Hospital MRN |
| 9 | Health plan beneficiary # | Pattern matching | Insurance identifiers |
| 10 | Account numbers | Pattern matching | Billing accounts |
| 11 | Certificate/license # | Pattern matching | Provider credentials |
| 12 | Vehicle identifiers | Regex | Transportation records |
| 13 | Device identifiers | Pattern matching | Implant serial numbers |
| 14 | Web URLs | Regex | Patient portal links |
| 15 | IP addresses | Regex | Telehealth session data |
| 16 | Biometric identifiers | Specialized detection | Facial reconstruction from imaging |
| 17 | Full-face photographs | Image analysis | Clinical photography |
| 18 | Unique identifying codes | Pattern matching | Research subject IDs if linkable |

---

## Open Source Tools Referenced

| Tool | Purpose | License | Source |
|------|---------|---------|--------|
| Microsoft Presidio | PII detection and anonymization | MIT | [GitHub](https://github.com/microsoft/presidio) |
| ARX | Data anonymization (k-anonymity, l-diversity) | Apache 2.0 | [GitHub](https://github.com/arx-deidentifier/arx) |
| Medical Image De-ID | DICOM MRI/CT/WSI de-identification | MIT | [GitHub](https://github.com/TIO-IKIM/medical_image_deidentification) |
| PhysioNet DeID | Clinical text de-identification | GPL | [PhysioNet](https://physionet.org/content/deid/) |
| PyDICOM | DICOM file handling with de-ID support | MIT | [GitHub](https://github.com/pydicom/pydicom) |

---

## References

### Regulatory Sources
- [HHS HIPAA for Research](https://www.hhs.gov/hipaa/for-professionals/special-topics/research/index.html)
- [HHS De-Identification Guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) (reviewed Feb 2025)
- [HHS HIPAA Security Rule NPRM](https://www.federalregister.gov/documents/2025/01/06/2024-30983/hipaa-security-rule-to-strengthen-the-cybersecurity-of-electronic-protected-health-information) (Jan 2025)
- [FDA 21 CFR Part 11 Q&A Guidance](https://www.fda.gov/media/166215/download) (Oct 2024)
- [NIST Privacy Framework 1.1 Draft](https://www.nist.gov/privacy-framework) (Apr 2025)
- [HHS SACHRP: IRB Considerations on AI](https://www.hhs.gov/ohrp/sachrp-committee/recommendations/irb-considerations-use-artificial-intelligence-human-subjects-research/index.html)

### Research Publications
- [Federated Deep Learning for Cancer Subtyping](https://aacrjournals.org/cancerdiscovery/article/15/9/1803/764360/) - Cancer Discovery, 2025
- [FL for Breast, Lung, Prostate Cancer](https://www.nature.com/articles/s41746-025-01591-5) - npj Digital Medicine, 2025
- [FL with Differential Privacy for Breast Cancer](https://www.nature.com/articles/s41598-025-95858-2) - Scientific Reports, 2025
- [Medical Image De-Identification Tool](https://link.springer.com/article/10.1007/s00330-025-11695-x) - European Radiology, 2025

### Open Source Projects
- [Microsoft Presidio](https://github.com/microsoft/presidio) - PII detection/anonymization
- [ARX Data Anonymization](https://github.com/arx-deidentifier/arx) - Statistical anonymization
- [HIPAA Policies (Open Source)](https://github.com/globerhofer/HIPAA-policies) - Policy templates
- [GA4GH Framework](https://www.ga4gh.org/framework/) - Genomic data sharing standards

---

*This directory is part of the Physical AI Oncology Trials Unification Framework.*
