# Human Oversight and Quality Management for CRF/AE Automation

## Purpose

This document defines the human oversight requirements and quality management procedures for AI-assisted automation of Case Report Form (CRF) data entry and Adverse Event (AE) reporting in oncology clinical trials. It is intended for engineers implementing these pipelines and for quality/regulatory teams reviewing their deployment.

## Regulatory Basis

| Regulation | Relevance |
|------------|-----------|
| ICH E6(R3) | GCP requirements for computerized systems, data governance, and oversight of technology-enabled processes |
| 21 CFR Part 11 | Electronic records and electronic signatures |
| FDA AI/ML Guidance (Jan 2025) | Predetermined Change Control Plans (PCCP) for adaptive AI |
| ICH E2B(R3) | Individual Case Safety Report (ICSR) format for AE reporting |
| HIPAA Security Rule | Access controls and audit trails for systems handling PHI |

---

## 1. CRF Auto-Fill Automation

### 1.1 Overview

AI agents can pre-populate CRF fields by extracting data from electronic medical records (EMR), lab systems, and imaging reports. This reduces manual transcription time by up to 73% (see `agentic-ai/results.md`) but introduces risk of propagating source-data errors.

### 1.2 Risk Classification of CRF Fields

| Risk Tier | Field Examples | Auto-Fill Policy | Human Review |
|-----------|---------------|-----------------|--------------|
| **Low** | Demographics (name, DOB, sex), site ID, visit date | Auto-fill permitted | Spot-check (10% sample audit) |
| **Medium** | Vitals, lab values, concomitant medications | Auto-fill with visual flag | Required before sign-off |
| **High** | Tumor response (RECIST), dose modifications, AE grading | Draft only; not auto-committed | Mandatory clinician review and e-signature |

### 1.3 Accuracy Thresholds

Based on empirical validation (see `agentic-ai/results.md`), the following accuracy thresholds must be met before a field category is eligible for auto-fill:

| Field Category | Minimum Accuracy | Current Measured | Status |
|----------------|-----------------|-----------------|--------|
| Demographics | 98% | 99.2% | Eligible |
| Vitals | 97% | 98.7% | Eligible |
| Medications | 95% | 96.4% | Eligible (with review) |
| Adverse events | 95% | 91.2% | **Not eligible for auto-commit** |
| Free-text narratives | 90% | 87.5% | **Not eligible for auto-commit** |

### 1.4 Required Controls

1. **Audit trail**: Every auto-filled field must be logged with timestamp, source system, extraction confidence score, and the identity of the reviewing human (21 CFR Part 11).
2. **Visual differentiation**: Auto-filled values must be visually distinct (e.g., highlighted background) from manually entered values until a human confirms them.
3. **Rejection workflow**: Reviewers must be able to reject, edit, or override any auto-filled value. Rejections are logged for quality trending.
4. **Confidence threshold**: Fields with extraction confidence below 0.85 must be flagged for mandatory review regardless of risk tier.
5. **Periodic revalidation**: Auto-fill accuracy must be re-measured quarterly against a manual gold-standard dataset. If accuracy drops below the threshold for any category, auto-fill for that category must be suspended until root cause is resolved.

---

## 2. Adverse Event (AE) Detection and Reporting

### 2.1 Overview

AI agents can assist with AE detection from clinical notes, grading per CTCAE v5.0, causality assessment, expectedness determination, and ICSR narrative drafting. These are safety-critical functions requiring robust human oversight.

### 2.2 Automation Levels

| AE Workflow Step | Permitted Automation Level | Human Role |
|-----------------|--------------------------|------------|
| **Detection** (flagging potential AEs in clinical notes) | Full automation with human notification | Review all flagged events; confirm or dismiss |
| **Grading** (CTCAE severity assignment) | Draft grade proposed to reviewer | Clinician confirms or corrects grade |
| **Causality assessment** (related / not related) | Advisory recommendation only | Investigator makes final determination and signs |
| **Expectedness** (expected vs. unexpected) | Lookup against reference safety information | Pharmacovigilance reviewer confirms |
| **Narrative drafting** | AI generates draft narrative | Medical writer reviews and edits before submission |
| **Regulatory submission** (ICSR to FDA/EMA) | Prohibited without human sign-off | Qualified person must review and submit |

### 2.3 Safety Gates

The following safety gates must be enforced in any AE automation pipeline:

1. **Serious Adverse Event (SAE) escalation**: Any event flagged as potentially serious (Grade 3+, hospitalization, life-threatening, or death) must immediately notify the principal investigator and sponsor safety desk. AI must not delay or filter SAE notifications.
2. **No autonomous submission**: AI-generated ICSRs must never be submitted to regulatory authorities without a qualified human reviewer's e-signature.
3. **Sensitivity over specificity**: AE detection models must be tuned for high sensitivity (target: >95%). False positives are reviewed and dismissed by humans; false negatives are unacceptable.
4. **Duplicate detection**: AI may flag potential duplicate AE reports, but a human must confirm before merging or suppressing.
5. **Expedited reporting timelines**: AI must surface SAEs within the system promptly so that human reviewers have sufficient time to meet 15-day (FDA) / 7-day (fatal/life-threatening) reporting deadlines.

### 2.4 Required Controls

1. **Blinding preservation**: In blinded trials, AE automation must not expose treatment assignment. Systems must operate on blinded data only; unblinding follows the trial's standard unblinding SOP.
2. **Inter-rater validation**: Before deployment, AI AE grading must be validated against a panel of >= 3 clinicians on a representative dataset. Cohen's kappa >= 0.75 is required for each CTCAE category.
3. **Override logging**: Every instance where a human overrides an AI recommendation (grade, causality, expectedness) must be logged with rationale for continuous learning and audit.
4. **Model versioning**: The AI model version used for each AE assessment must be recorded in the audit trail. Model updates require re-validation per the PCCP before clinical deployment.

---

## 3. Quality Management System (QMS) Integration

### 3.1 Document Control

| Document | Owner | Review Cycle |
|----------|-------|-------------|
| This human oversight SOP | Quality Assurance | Annual or after major model update |
| CRF auto-fill validation report | Data Management | Quarterly |
| AE detection validation report | Pharmacovigilance | Semi-annual |
| System access log review | IT Security | Monthly |

### 3.2 Training Requirements

| Role | Required Training |
|------|-------------------|
| Clinical Research Coordinator | CRF review procedures, auto-fill field identification, rejection workflow |
| Investigator / Sub-investigator | AE grading review, causality override, e-signature responsibilities |
| Data Manager | Audit trail review, accuracy monitoring, revalidation triggers |
| AI/ML Engineer | Model versioning, PCCP compliance, bias monitoring |

### 3.3 CAPA (Corrective and Preventive Action) Triggers

Initiate a CAPA investigation when any of the following occur:

- CRF auto-fill accuracy drops below threshold for any field category
- An AE is missed by the detection system and discovered during manual review or audit
- A regulatory submission contains data that was incorrectly auto-filled
- Audit trail gaps are identified (missing timestamps, unsigned records)
- A model update causes a measurable change in detection sensitivity or specificity

### 3.4 Key Performance Indicators (KPIs)

| KPI | Target | Measurement Frequency |
|-----|--------|----------------------|
| CRF auto-fill accuracy (per field tier) | See Section 1.3 | Quarterly |
| AE detection sensitivity | >= 95% | Semi-annual |
| AE grading inter-rater kappa | >= 0.75 | Per validation cycle |
| Human review completion rate | 100% for High-risk fields | Continuous |
| Time from AE detection to investigator notification | < 4 hours | Continuous |
| Audit trail completeness | 100% | Monthly |

---

## 4. Relationship to Repository Components

| Repository Module | Role in CRF/AE Automation |
|-------------------|--------------------------|
| `agentic-ai/` | Multi-agent CRF extraction and AE detection pipelines |
| `privacy/access-control/` | Role-based access and 21 CFR Part 11 audit trails |
| `privacy/phi-pii-management/` | PHI detection before data enters automation pipelines |
| `regulatory/fda-compliance/` | PCCP tracking for adaptive AI models |
| `regulatory/ich-gcp/` | E6(R3) compliance verification for computerized systems |
| `examples/04_agentic_clinical_workflow.py` | Reference implementation of multi-agent trial coordination |

---

## 5. Summary

Automation of CRF data entry and AE reporting can materially improve efficiency and reduce transcription errors in oncology trials. However, these are regulated clinical processes where errors have direct patient-safety consequences. The controls in this document ensure that:

1. **Humans remain accountable** for every clinical decision.
2. **AI outputs are transparent** (audit-trailed, versioned, visually flagged).
3. **Safety-critical functions** (SAE escalation, regulatory submission, causality assessment) always require human sign-off.
4. **Continuous monitoring** detects degradation before it affects trial integrity.

No AI system deployed under this framework operates autonomously on safety-critical clinical trial data without qualified human review.
