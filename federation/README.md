# Multi-Site Federated Oncology Trial Coordination Platform

**Version**: 2.7.1
**Status**: Active Development
**Last Updated**: March 2026

A federated learning and trial coordination platform for orchestrating multi-site oncology clinical trials without sharing raw patient data. Supports federated model training, differential privacy, secure aggregation, enrollment synchronization, cross-site data harmonization, and privacy-preserving analytics.

## Architecture

```
federation/
├── README.md                           # This file
├── __init__.py
├── federated_coordinator.py            # Core federated learning orchestration engine
├── differential_privacy.py             # Privacy budget management, noise mechanisms
├── secure_aggregation.py               # Simulated secure multi-party computation
├── site_enrollment.py                  # Enrollment sync, conflict resolution, stratification
├── data_harmonization.py               # DICOM/FHIR normalization, vocabulary mapping
├── consortium_reporting.py             # DSMB reports, enrollment dashboards, AE summaries
├── privacy_analytics.py                # Privacy-preserving survival analysis
└── examples-federation/                # Progressive example scripts
    ├── README.md
    ├── 01_basic_two_site.py            # Minimal 2-site federation
    ├── 02_differential_privacy.py      # Privacy budget demonstration
    ├── 03_secure_aggregation.py        # Secure weight aggregation
    ├── 04_enrollment_sync.py           # Multi-site enrollment coordination
    ├── 05_data_harmonization.py        # Cross-site data normalization
    └── 06_full_consortium.py           # 8-site multi-cancer coordination
```

## Core Components

### Federated Learning Coordinator (`federated_coordinator.py`)

Orchestrates model training across N simulated clinical sites without sharing raw patient data.

- **FedAvg** — Federated Averaging (McMahan et al., 2017): weighted average proportional to site patient counts
- **FedProx** — Proximal regularization (Li et al., 2020): handles heterogeneous site data distributions
- **SCAFFOLD** — Variance-reduced optimization (Karimireddy et al., 2020): corrects client drift via control variates

### Differential Privacy Engine (`differential_privacy.py`)

Configurable epsilon/delta privacy budgets with formal guarantees.

- **Gaussian mechanism** — (epsilon, delta)-DP for gradient updates
- **Laplacian mechanism** — Pure epsilon-DP for summary statistics
- **Gradient clipping** — Bounds per-sample contributions (L2 norm clipping)
- **Budget accounting** — Tracks cumulative epsilon consumption with exhaustion prevention
- **Histogram privatization** — Noised counting queries for enrollment/demographic statistics

### Secure Aggregation Protocol (`secure_aggregation.py`)

Simulated secure multi-party computation preventing any single site from reconstructing another's contributions.

- **Additive secret sharing** — Splits model weights into N shares summing to the original
- **Pairwise masking** — Complementary masks that cancel during aggregation
- **Commitment verification** — SHA-256 commitments prevent post-hoc input modification
- **Dropout tolerance** — Configurable resilience to site disconnections

### Site Enrollment Synchronizer (`site_enrollment.py`)

Cross-site enrollment tracking with conflict resolution and stratification.

- **Stratified block randomization** — Balanced arm assignment within strata
- **Duplicate enrollment detection** — Cross-site patient ID collision detection
- **Conflict resolution** — First-come, random assignment, and manual review strategies
- **Arm balance monitoring** — Detects treatment arm imbalances exceeding configurable thresholds
- **Withdrawal tracking** — Maintains accurate counts after patient withdrawals

### Cross-Site Data Harmonization (`data_harmonization.py`)

DICOM/FHIR interoperability across heterogeneous site systems.

- **DICOM normalization** — Modality codes, body part terminology, pixel spacing, patient position
- **ICD-10 → SNOMED CT** — Oncology diagnosis code mapping (6 cancer types)
- **LOINC coding** — Tumor marker standardization (CEA, PSA, CA125, CA19-9, AFP, HER2)
- **FHIR R4 resources** — Condition, Observation, MedicationStatement resource creation

### Consortium Reporting Engine (`consortium_reporting.py`)

Generates regulatory-ready reports and DSMB packages.

- **Enrollment dashboards** — Site-level and aggregate enrollment with projections
- **Adverse event summaries** — CTCAE v5.0 grading, SOC distribution, SAE rates
- **Site performance reports** — Risk-based monitoring with composite risk scoring
- **DSMB packages** — Combined enrollment, safety, and efficacy data with recommendations

### Privacy-Preserving Analytics (`privacy_analytics.py`)

Cross-site survival analysis without exposing individual patient records.

- **Federated Kaplan-Meier** — Product-limit estimator from aggregated at-risk/event counts
- **Federated Cox PH** — Pooled covariate analysis with Harrell's C-index
- **Response rate estimation** — Treatment arm response rates with confidence intervals
- **Cell suppression** — Automatic suppression of small cell sizes (configurable threshold)

## Examples (`examples-federation/`)

| # | Example | Description |
|---|---------|-------------|
| 01 | `01_basic_two_site.py` | Minimal 2-site federation with FedAvg |
| 02 | `02_differential_privacy.py` | Privacy budget demonstration |
| 03 | `03_secure_aggregation.py` | Secure weight aggregation |
| 04 | `04_enrollment_sync.py` | Multi-site enrollment coordination |
| 05 | `05_data_harmonization.py` | Cross-site data normalization |
| 06 | `06_full_consortium.py` | 8-site multi-cancer consortium |

## Quick Start

```bash
# Basic 2-site federation
python federation/examples-federation/01_basic_two_site.py

# Full 8-site consortium (all components)
python federation/examples-federation/06_full_consortium.py
```

## Dependencies

Core dependencies only (no external services required):

- Python 3.10+
- NumPy 1.24.0+
- SciPy 1.11.0+

All multi-site communication is simulated in-process — no networking, GPU, or external FHIR/DICOM servers needed.

## Clinical Trial Compliance

This platform addresses requirements from:

- **ICH E6(R3)** — Good Clinical Practice guidelines for multi-site trials
- **21 CFR Part 11** — Electronic records and audit trails
- **HIPAA** — De-identification and minimum necessary access
- **FDA AI/ML Guidance** — Transparency in federated model development
- **GDPR** — Differential privacy for cross-border data processing

## Roadmap Alignment

This module implements the Q2–Q3 2026 roadmap objectives documented in `unification/README.md`:

- Q2 2026: *Establish consortium data sharing infrastructure*
- Q3 2026: *Multi-site clinical trial coordination platform*

## Disclaimer

**RESEARCH USE ONLY.** This platform is not approved for clinical decision-making. All outputs must be reviewed by qualified biostatisticians, clinical investigators, and regulatory professionals before any clinical application.

## License

MIT — See repository root for full license terms.
