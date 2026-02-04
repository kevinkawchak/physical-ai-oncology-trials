# Regulatory Framework for Physical AI Oncology Trials

*FDA, IRB, and ICH-GCP compliance tools for AI-enabled clinical trial management (February 2026)*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Executive Summary

Physical AI oncology clinical trials operate at the intersection of drug/biologic regulation, medical device oversight, and emerging AI governance frameworks. This directory provides tools for navigating the regulatory landscape across FDA, IRB, ICH-GCP, and international jurisdictions, enabling trial teams to track submissions, verify compliance, and maintain audit-ready documentation.

**Key Capabilities**:
- FDA submission tracking for AI/ML-enabled medical devices (510(k), De Novo, PMA, Breakthrough)
- IRB protocol management with AI-specific review requirements
- ICH E6(R3) GCP compliance verification (effective Sep 2025 in the U.S.)
- Multi-jurisdiction regulatory intelligence monitoring (FDA, EMA, PMDA, TGA, Health Canada)
- Automated compliance checklists for 21 CFR Part 11, IEC 62304, ISO 13482

---

## Regulatory Landscape (2025-2026)

### Key Milestones

| Date | Event | Impact |
|------|-------|--------|
| Jan 2025 | FDA draft guidance: AI-enabled device software functions | Lifecycle management, TPLC approach for 510(k)/PMA/De Novo |
| Jan 2025 | FDA draft guidance: AI in drug development (CDER) | Risk-based credibility framework for AI models |
| Jan 2025 | ICH E6(R3) adopted (Step 4) | Comprehensive GCP overhaul with digital technology provisions |
| Jul 2025 | ICH E6(R3) effective in EU | GCP enforcement across European clinical sites |
| Aug 2025 | FDA PCCP guidance finalized | Predetermined Change Control Plans for adaptive AI devices |
| Aug 2025 | ArteraAI Prostate De Novo authorization | First AI prognostic tool for non-metastatic prostate cancer |
| Sep 2025 | ICH E6(R3) published by U.S. FDA | GCP requirements for U.S. clinical trial sites |
| Dec 2025 | FDA RWE guidance finalized | Real-World Evidence for device regulatory decisions |
| Feb 2026 | QMSR effective (ISO 13485 alignment) | U.S. quality management system regulation updated |
| Aug 2026 | EU AI Act high-risk requirements | Full transparency and conformity requirements |
| Aug 2027 | EU AI Act for CE-marked medical devices | AI systems in MDR/IVDR devices fully regulated |

### FDA AI/ML Device Statistics

As of December 2025, the FDA has authorized over **1,300 AI/ML-enabled medical devices**. In oncology:
- Cancer radiology: 54.9%
- Pathology: 19.7%
- Radiation oncology: 8.5%
- Gastroenterology: 8.5%
- Clinical oncology: 7.0%

---

## Directory Structure

```
regulatory/
├── README.md                          # This file
│
├── fda-compliance/                    # FDA regulatory pathway management
│   ├── README.md                      # FDA compliance overview
│   └── fda_submission_tracker.py      # Submission tracking and documentation
│
├── irb-management/                    # IRB protocol and review management
│   ├── README.md                      # IRB management overview
│   └── irb_protocol_manager.py        # IRB submission and tracking
│
├── ich-gcp/                           # ICH E6(R3) GCP compliance
│   ├── README.md                      # GCP compliance overview
│   └── gcp_compliance_checker.py      # GCP verification and audit tools
│
└── regulatory-intelligence/           # Regulatory landscape monitoring
    ├── README.md                      # Regulatory intelligence overview
    └── regulatory_tracker.py          # Multi-jurisdiction tracking
```

---

## Quick Start

### 1. Track FDA Submissions

```python
from regulatory.fda_compliance.fda_submission_tracker import FDASubmissionTracker

tracker = FDASubmissionTracker(
    sponsor="Physical AI Oncology Consortium",
    device_class="II"
)

# Register a new AI device submission
submission = tracker.create_submission(
    submission_type="de_novo",
    device_name="AI-Guided Surgical Planning System",
    intended_use="AI-assisted tumor resection planning using patient-specific digital twins",
    ai_ml_components=["tumor_segmentation_model", "surgical_path_optimizer"],
    breakthrough_designation=True
)

# Generate pre-submission package checklist
checklist = tracker.generate_presub_checklist(submission)
for item in checklist.items:
    print(f"  [{item.status}] {item.description}")
```

### 2. Manage IRB Protocols

```python
from regulatory.irb_management.irb_protocol_manager import IRBProtocolManager

irb = IRBProtocolManager(
    institution="Memorial Sloan Kettering Cancer Center",
    irb_type="central"
)

# Create AI-specific protocol submission
protocol = irb.create_protocol(
    title="Physical AI-Guided Surgical Resection for NSCLC",
    pi_name="Dr. Sarah Chen",
    ai_components=[
        "real-time tumor boundary detection",
        "robotic instrument guidance",
        "digital twin treatment simulation"
    ],
    participant_count=200,
    trial_phase="pivotal"
)

# Generate AI-specific review requirements
ai_review = irb.generate_ai_review_checklist(protocol)
print(f"AI-specific review items: {len(ai_review.items)}")
```

### 3. Verify ICH-GCP Compliance

```python
from regulatory.ich_gcp.gcp_compliance_checker import GCPComplianceChecker

checker = GCPComplianceChecker(
    guideline_version="E6_R3",
    jurisdiction="us_fda"
)

# Run compliance verification
report = checker.verify_compliance(
    protocol_path="protocols/NSCLC_AI_surgical_v3.0.pdf",
    trial_master_file="tmf/",
    check_categories=[
        "investigator_responsibilities",
        "sponsor_obligations",
        "digital_technology_provisions",
        "data_governance",
        "quality_management"
    ]
)

print(f"Compliance score: {report.overall_score:.1f}%")
print(f"Findings: {report.total_findings}")
```

### 4. Monitor Regulatory Changes

```python
from regulatory.regulatory_intelligence.regulatory_tracker import RegulatoryTracker

tracker = RegulatoryTracker(
    jurisdictions=["fda", "ema", "ich"],
    topics=["ai_ml_devices", "oncology", "clinical_trials", "digital_health"]
)

# Get current regulatory updates
updates = tracker.get_recent_updates(days=30)
for update in updates:
    print(f"  [{update.jurisdiction}] {update.title} ({update.date})")
```

---

## Regulatory Pathway Decision Tree

```
Is the AI/ML system a medical device?
│
├── YES: Does it have a predicate device?
│   │
│   ├── YES → 510(k) Pathway
│   │   └── Include: AI model description, training data,
│   │       performance metrics, human-AI workflow
│   │
│   └── NO: Is it high-risk (Class III)?
│       │
│       ├── YES → PMA Pathway
│       │   └── Include: Clinical evidence, safety data,
│       │       algorithm validation, post-market plan
│       │
│       └── NO → De Novo Pathway
│           └── Include: Risk assessment, performance testing,
│               special controls, labeling
│
├── MAYBE: Does it qualify for Breakthrough Device?
│   │
│   └── Apply for designation → Priority review,
│       interactive communication, data discussions
│
└── NO: Digital health technology exemption?
    │
    └── Review Jan 2026 FDA digital health guidance
        for exemption criteria
```

---

## FDA Submission Requirements for AI/ML Devices

### Required Documentation (per Jan 2025 Draft Guidance)

| Category | Requirement | Description |
|----------|------------|-------------|
| Model Description | Architecture, training approach | Algorithm type, model complexity, decision boundaries |
| Data Lineage | Training/validation/test data | Data sources, demographic representation, quality controls |
| Performance | Metrics tied to intended use | Sensitivity, specificity, AUC, subgroup analysis |
| Bias Analysis | Demographic performance | Age, sex, race/ethnicity stratification |
| Human-AI Workflow | Clinical integration | How clinicians interact with AI outputs |
| Cybersecurity | Device security | Threat modeling, vulnerability management |
| Labeling | User-facing information | Intended use, limitations, warnings |
| Post-Market Plan | Ongoing monitoring | Performance drift detection, adverse event tracking |
| PCCP (optional) | Change control plan | Pre-approved modification boundaries |

### Predetermined Change Control Plan (PCCP)

The August 2025 finalized PCCP guidance allows AI devices to be updated post-market without new submissions, provided modifications fall within a pre-approved plan:

- **Modification Protocol**: Describes what types of changes are permitted
- **Impact Assessment**: How each change type affects safety and effectiveness
- **Verification and Validation**: Testing required before implementing changes
- **Transparency**: Clear communication to users about device updates

---

## ICH E6(R3) Key Changes for AI-Enabled Trials

ICH E6(R3) was adopted January 2025 and is effective in the U.S. as of September 2025. Key changes relevant to physical AI oncology trials:

| Area | E6(R2) | E6(R3) |
|------|--------|--------|
| Risk Management | SDV-focused quality assurance | Risk-Based Quality Management (RBQM) |
| Digital Technology | Not addressed | Explicit provisions for digital tools |
| Data Governance | Investigator responsibility | Shared sponsor/investigator responsibility |
| AI/ML Validation | Not addressed | Justify use, document validation, supervise |
| Service Providers | CRO oversight | Broader service provider category |
| Decentralized Trials | Not addressed | Annex 2 (expected 2026) |

---

## Recent FDA AI Oncology Approvals and Designations

| Device | Company | Indication | Pathway | Date |
|--------|---------|------------|---------|------|
| ArteraAI Prostate | Artera | Prostate cancer prognosis | De Novo | Aug 2025 |
| Allix5 | Clairity | Breast cancer risk prediction | De Novo | May 2025 |
| Serial CTRS | Onc.AI | NSCLC mortality risk (CT) | Breakthrough | Feb 2025 |
| VENTANA TROP2 RxDx | Roche | NSCLC CDx (computational pathology) | Breakthrough | Apr 2025 |
| DAMO PANDA | DAMO | Pancreatic cancer detection | Breakthrough | Late 2025 |

---

## Open Source Tools Referenced

| Tool | Purpose | License | Source |
|------|---------|---------|--------|
| Phoenix CTMS | ICH GCP-compliant trial management | LGPL 2.1 | [GitHub](https://github.com/phoenixctms/ctsms) |
| OpenClinica | Electronic data capture (EDC/CDM) | LGPL | [GitHub](https://github.com/OpenClinica/OpenClinica) |
| ComplianceAsCode | Compliance control automation | Apache 2.0 | [GitHub](https://github.com/ComplianceAsCode) |

---

## References

### FDA Guidance Documents
- [FDA Draft Guidance: AI-Enabled Device Software Functions](https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices) (Jan 2025)
- [FDA Draft Guidance: AI in Drug Development](https://www.fda.gov/news-events/press-announcements/fda-proposes-framework-advance-credibility-ai-models-used-drug-and-biological-product-submissions) (Jan 2025)
- [FDA PCCP Guidance for AI Devices](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence) (Aug 2025)
- [FDA RWE Guidance Finalized](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-real-world-evidence-support-regulatory-decision-making-medical-devices) (Dec 2025)
- [FDA AI/ML Device List](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [FDA OCE Oncology AI Program](https://www.fda.gov/about-fda/oncology-center-excellence/oce-oncology-artificial-intelligence-program)
- [FDA 21 CFR Part 11 Q&A Guidance](https://www.fda.gov/media/166215/download) (Oct 2024)

### ICH Guidelines
- [ICH E6(R3) Final Guideline](https://database.ich.org/sites/default/files/ICH_E6(R3)_Step4_FinalGuideline_2025_0106.pdf) (Jan 2025)
- [FDA Adoption of ICH E6(R3)](https://acrpnet.org/2025/09/16/fda-publishes-ich-e6r3-what-it-means-for-u-s-clinical-trials) (Sep 2025)

### IRB Guidance
- [HHS SACHRP: IRB Considerations on AI](https://www.hhs.gov/ohrp/sachrp-committee/recommendations/irb-considerations-use-artificial-intelligence-human-subjects-research/index.html)
- [MRCT Center: Framework for Review of Clinical Research Involving AI](https://mrctcenter.org/resource/framework-for-review-of-clinical-research-involving-ai/) (Jul 2025)

### International Regulations
- [EU AI Act Timeline](https://gardner.law/news/eu-ai-act-compliance-timeline)
- [EU MDR Simplification Proposals](https://healthcarelifesciences.bakermckenzie.com/2025/12/19/the-eus-2025-proposal-to-simplify-the-medical-and-in-vitro-diagnostic-devices-regulations-mdr-ivdr/) (Dec 2025)
- [WHO: Regulatory Considerations on AI for Health](https://www.who.int/publications/i/item/9789240078871) (2023)

---

*This directory is part of the Physical AI Oncology Trials Unification Framework.*
