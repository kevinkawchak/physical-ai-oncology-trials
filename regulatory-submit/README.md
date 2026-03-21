# Regulatory Submission Automation for Physical AI Oncology Trials

*FDA Pre-Submission Package Generation, PCCP Template Engine, Classification Decision Support, IEC 62304 Documentation, Clinical Evidence Reporting, and 21 CFR Part 11 Audit Trail Generation*

**Version**: 2.7.1
**Status**: Active Development
**Last Updated**: March 2026

---

## Executive Summary

The `regulatory-submit/` directory provides automated document generation tools for FDA regulatory submissions of AI/ML-enabled oncology devices. Unlike the `regulatory/` directory (which tracks submission status, manages IRB protocols, verifies GCP compliance, and monitors regulatory intelligence), this module focuses on **generating** the structured documents required for submissions — Pre-Sub packages, PCCP plans, pathway analysis, lifecycle documentation, clinical evidence reports, and audit trails.

All output is generated as structured Markdown. No external APIs, FDA systems, or network connectivity required.

**Key Capabilities**:
- FDA Pre-Submission (Q-Sub) meeting request package generation
- Predetermined Change Control Plan (PCCP) document authoring per Aug 2025 guidance
- 510(k) / De Novo / PMA pathway classification decision support
- IEC 62304 software lifecycle documentation (SDP, SRS, SAD, Risk Analysis)
- Clinical evidence reports linking simulation benchmarks to clinical claims
- 21 CFR Part 11-compliant audit trail generation with hash chain integrity

---

## Directory Structure

```
regulatory-submit/
├── README.md                          # This file
├── presub_generator.py                # FDA Pre-Sub meeting package generation
├── pccp_engine.py                     # Predetermined Change Control Plan templates
├── classification_advisor.py          # 510(k)/De Novo pathway decision support
├── iec62304_generator.py              # IEC 62304 software lifecycle documentation
├── clinical_evidence.py               # Simulation-to-clinical-claims linking
├── audit_trail.py                     # 21 CFR Part 11 audit trail generation
│
└── examples-regulatory-submit/        # 6 progressive example scripts
    ├── README.md                      # Examples overview
    ├── 01_presub_package.py           # Basic Pre-Sub generation
    ├── 02_pccp_plan.py                # Change control plan creation
    ├── 03_classification.py           # Pathway decision support
    ├── 04_iec62304_docs.py            # Lifecycle documentation
    ├── 05_clinical_evidence.py        # Evidence report building
    └── 06_full_submission.py          # Complete regulatory strategy
```

---

## Module Overview

| Module | Purpose | Key Output |
|--------|---------|------------|
| `presub_generator.py` | FDA Pre-Sub meeting request packages | Device description, AI model docs, testing protocols, FDA questions |
| `pccp_engine.py` | PCCP documents per Aug 2025 guidance | Modification boundaries, V&V protocols, transparency plans |
| `classification_advisor.py` | Regulatory pathway decision support | Pathway recommendation, risk classification, decision factors |
| `iec62304_generator.py` | IEC 62304 lifecycle documentation | SDP, SRS, SAD, risk analysis matrices |
| `clinical_evidence.py` | Clinical evidence reports | Benchmark results with CIs, subgroup analysis, claims |
| `audit_trail.py` | 21 CFR Part 11 audit trails | Training runs, validations, config changes, hash chains |

---

## Quick Start

### 1. Generate a Pre-Sub Package

```python
from regulatory_submit.presub_generator import (
    PreSubGenerator, DeviceCategory, RegulatoryPathway, AIModelDescription,
)

generator = PreSubGenerator(sponsor="My Oncology Consortium")
package = generator.create_package(
    device_name="AI Surgical Planner",
    device_category=DeviceCategory.SURGICAL_PLANNING,
    intended_use="AI-assisted surgical planning for NSCLC tumor resection",
    proposed_pathway=RegulatoryPathway.DE_NOVO,
    oncology_indication="NSCLC stages I-IIIA",
)

generator.add_ai_model(package, AIModelDescription(
    model_name="TumorSeg-3D",
    model_type="segmentation",
    training_data_size=12500,
    performance_metrics={"dice": 0.912, "sensitivity": 0.945},
))

generator.populate_risk_considerations(package)
generator.auto_generate_questions(package)
markdown = generator.generate_markdown(package)
```

### 2. Create a PCCP

```python
from regulatory_submit.pccp_engine import PCCPEngine

engine = PCCPEngine(sponsor="My Oncology Consortium")
document = engine.create_pccp(
    device_name="AI Surgical Planner",
    submission_reference="DN-2026-0042",
    ai_model_names=["TumorSeg-3D", "PathOptimizer-RL"],
)

engine.populate_default_boundaries(document)
engine.populate_default_validation(document)
engine.set_transparency_plan(document)
markdown = engine.generate_markdown(document)
```

### 3. Classify Regulatory Pathway

```python
from regulatory_submit.classification_advisor import (
    ClassificationAdvisor, DeviceCharacteristics, AIDeviceType,
)

advisor = ClassificationAdvisor()
recommendation = advisor.analyze(DeviceCharacteristics(
    device_name="AI Surgical Planner",
    device_type=AIDeviceType.TREATMENT_PLANNING,
    intended_use="AI-assisted surgical planning",
    novel_algorithm=True,
    life_threatening_condition=True,
))
print(f"Pathway: {recommendation.recommended_pathway.value}")
```

### 4. Generate IEC 62304 Documents

```python
from regulatory_submit.iec62304_generator import IEC62304Generator, SafetyClass

generator = IEC62304Generator(sponsor="My Oncology Consortium")
doc_set = generator.create_document_set(
    project_name="OncoTwin-SW",
    device_name="AI Surgical Planner",
    safety_class=SafetyClass.CLASS_C,
)
generator.populate_default_requirements(doc_set)
generator.populate_sample_risks(doc_set)
docs = generator.generate_all(doc_set)
```

### 5. Build Clinical Evidence Report

```python
from regulatory_submit.clinical_evidence import (
    ClinicalEvidenceBuilder, EvidenceLevel, MetricType,
)

builder = ClinicalEvidenceBuilder(sponsor="My Oncology Consortium")
report = builder.create_report(
    device_name="AI Surgical Planner",
    intended_use="AI-assisted surgical planning for NSCLC",
)
builder.add_benchmark_result(
    report, "Validation Study", EvidenceLevel.RETROSPECTIVE,
    MetricType.ACCURACY, "Expert Agreement", 0.856, 200,
)
markdown = builder.generate_markdown(report)
```

### 6. Generate Audit Trail

```python
from regulatory_submit.audit_trail import AuditTrailGenerator, AuditEventType

generator = AuditTrailGenerator(system_name="OncoTwin Platform")
generator.record_training_run(
    operator="ml_engineer",
    model_name="TumorSeg-3D",
    model_version="1.0.0",
    dataset_id="DS-2026-Q1",
    dataset_size=12500,
    final_metrics={"dice": 0.912},
)
report = generator.generate_report()
```

---

## Relationship to `regulatory/` Directory

| Aspect | `regulatory/` (existing) | `regulatory-submit/` (this module) |
|--------|--------------------------|-----------------------------------|
| **Purpose** | Tracks and monitors regulatory status | Generates submission documents |
| **Function** | Status management, compliance checking | Document authoring, package assembly |
| **Output** | Status reports, compliance scores | Structured Markdown documents |
| **Scope** | FDA tracking, IRB management, GCP, intelligence | Pre-Sub, PCCP, classification, IEC 62304, evidence, audit |
| **Interaction** | Monitors ongoing regulatory obligations | Produces initial submission artifacts |

---

## Regulatory Standards Referenced

| Standard | Module(s) |
|----------|----------|
| FDA Draft Guidance: AI-Enabled Device Software Functions (Jan 2025) | presub_generator, classification_advisor |
| FDA PCCP Guidance for AI Devices (Aug 2025, finalized) | pccp_engine, presub_generator |
| FDA Q-Submission Program Guidance (Rev. 2025) | presub_generator |
| 21 CFR Part 11 — Electronic Records | audit_trail |
| 21 CFR Part 807 — Premarket Notification (510(k)) | classification_advisor |
| 21 CFR Part 860 — De Novo Classification | classification_advisor |
| IEC 62304:2015 — Software Lifecycle | iec62304_generator |
| ISO 14971:2019 — Risk Management | iec62304_generator |
| IEC 80601-2-77 — Surgical Robots | classification_advisor |
| ISO 13482 — Robot Safety | classification_advisor |
| SPIRIT-AI / CONSORT-AI | clinical_evidence |

---

## Dependencies

All modules require only Python 3.10+ standard library. No external packages needed.

---

## License

MIT — See repository root LICENSE for details.

---

*This directory is part of the Physical AI Oncology Trials Unification Framework.*
