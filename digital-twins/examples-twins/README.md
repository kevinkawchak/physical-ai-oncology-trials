# Digital Twins Examples for Oncology Clinical Trials

*Comprehensive code examples for engineers building patient-specific digital twins in physical AI oncology trials (February 2026)*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

This directory contains **6 production-oriented code examples** focused on advanced digital twin capabilities for oncology clinical trials. These examples complement the core `digital-twins/` modules (patient modeling, treatment simulation, clinical integration) by providing end-to-end implementations of the most pressing engineering challenges in deploying digital twins for clinical use.

**Audience**: Engineers working directly on digital twin systems for oncology — ML engineers, biomedical engineers, clinical informaticists, and regulatory affairs specialists.

---

## Examples

| # | File | Use Case | Key Techniques | Difficulty |
|---|------|----------|---------------|------------|
| 01 | `01_realtime_dt_synchronization.py` | Live patient data fusion during treatment | EKF, particle filter, anomaly detection, CUSUM | Advanced |
| 02 | `02_multi_organ_toxicity_twin.py` | Systemic toxicity prediction across organs | PBPK compartmental model, CTCAE grading, dose modification | Advanced |
| 03 | `03_adaptive_radiation_therapy_dt.py` | Daily anatomical adaptation during RT | Deformable registration, dose accumulation, BED/EQD2 | Advanced |
| 04 | `04_tumor_microenvironment_immunotherapy_dt.py` | Checkpoint inhibitor response prediction | TME agent-based ODE, PD-1/PD-L1 dynamics, iRECIST | Advanced |
| 05 | `05_virtual_trial_cohort_dt.py` | In-silico trial design and virtual control arms | Virtual patient generation, Bayesian interim analysis, power simulation | Intermediate |
| 06 | `06_dt_validation_verification.py` | Regulatory V&V and model cards for FDA submission | C-index, calibration, AUC, subgroup analysis, robustness | Intermediate |

---

## Relationship to Existing Modules

These examples build on — but do not duplicate — the core digital twins modules:

| Core Module | What It Does | Examples-Twins Extension |
|-------------|-------------|-------------------------|
| `patient-modeling/` | Creates tumor DTs from imaging (growth models) | **01**: Real-time synchronization of DTs with live clinical data |
| `treatment-simulation/` | Predicts treatment response (chemo, RT, IO) | **02**: Multi-organ toxicity modeling; **03**: Adaptive RT dose accumulation; **04**: Deep TME immunotherapy modeling |
| `clinical-integration/` | FHIR/DICOM connectivity, clinical workflows | **05**: Population-level virtual trial DTs; **06**: Regulatory V&V framework |

---

## Quick Start

```bash
# All examples require only NumPy and SciPy (included in requirements.txt)
pip install numpy scipy

# Run any example
python digital-twins/examples-twins/01_realtime_dt_synchronization.py
python digital-twins/examples-twins/02_multi_organ_toxicity_twin.py
python digital-twins/examples-twins/03_adaptive_radiation_therapy_dt.py
python digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py
python digital-twins/examples-twins/05_virtual_trial_cohort_dt.py
python digital-twins/examples-twins/06_dt_validation_verification.py
```

---

## Example Details

### 01: Real-Time Digital Twin Synchronization

**Problem**: During active treatment (e.g., 6 cycles of FOLFOX over 12 weeks), the patient DT must stay synchronized with asynchronous clinical observations arriving at irregular intervals from multiple hospital systems.

**Solution**: Extended Kalman Filter (EKF) and Sequential Monte Carlo (particle filter) for Bayesian state estimation over an 8-dimensional patient state vector (tumor volume, growth rate, drug effect, ANC, creatinine, hemoglobin, weight, ECOG).

**Key capabilities**:
- Asynchronous multi-modal data fusion (vitals, labs, imaging, tumor markers)
- Anomaly detection via innovation monitoring (CUSUM) for early adverse event warning
- 21 CFR Part 11 compliant audit trail for every state update
- Missing data handling and observation gating

### 02: Multi-Organ Toxicity Digital Twin

**Problem**: Cytotoxic chemotherapy causes systemic toxicity across multiple organ systems. Predicting grade 3+ adverse events before clinical manifestation enables preemptive dose modification.

**Solution**: Physiologically-based pharmacokinetic (PBPK) model distributing drug across plasma, heart, kidney, liver, nerve, and marrow compartments, coupled with organ-specific toxicodynamic models.

**Key capabilities**:
- 6-compartment PBPK drug distribution (SciPy ODE integration)
- Organ-specific toxicity: cardiotoxicity (LVEF), nephrotoxicity (GFR), hepatotoxicity (bilirubin), neurotoxicity (TNSc), myelosuppression (ANC)
- CTCAE v5.0 grade prediction with dose modification recommendations
- Multi-cycle cumulative toxicity tracking
- Drug-specific parameter library (cisplatin, doxorubicin, oxaliplatin, paclitaxel)

### 03: Adaptive Radiation Therapy Digital Twin

**Problem**: During 30-fraction radiation therapy, tumor shrinks 30-50% and OAR anatomy shifts. Without adaptation, planned dose diverges from delivered dose.

**Solution**: B-spline deformable image registration between daily CBCT and planning CT, dose warping, cumulative dose accumulation on reference anatomy, and replanning trigger detection.

**Key capabilities**:
- Deformable image registration with Jacobian quality monitoring (AAPM TG-132)
- Dose accumulation on reference anatomy across fractions (AAPM TG-275)
- DVH metric computation (D95, D2, Dmax, Dmean, V20)
- BED and EQD2 computation for variable fraction sizes
- Automated replanning trigger detection

### 04: Tumor Microenvironment & Immunotherapy Digital Twin

**Problem**: Checkpoint inhibitors benefit only 20-40% of patients. Response depends on complex TME factors (TIL density, PD-L1, TMB, cytokine milieu) that vary between patients.

**Solution**: Coupled ODE model of tumor-immune dynamics with 9 state variables (tumor cells, CD8+ T-cells, Tregs, macrophages, DCs, PD-L1, IFN-gamma, TGF-beta, IL-10) and checkpoint inhibitor pharmacokinetics.

**Key capabilities**:
- PD-1/PD-L1 checkpoint axis dynamics with anti-PD-1 blockade
- CD8+ T-cell activation, exhaustion, and reinvigoration
- Cytokine feedback loops (IFN-gamma, TGF-beta, IL-10)
- Composite immunogenicity scoring from TMB, MSI, PD-L1, TIL
- iRECIST response classification with pseudoprogression detection
- Mono vs. combination therapy comparison (pembrolizumab ± ipilimumab)

### 05: Virtual Clinical Trial Cohort Digital Twin

**Problem**: Oncology trials are slow, expensive, and have >95% failure rate. Virtual cohort DTs can reduce enrollment needs, optimize sample sizes, and enable adaptive designs.

**Solution**: Population-level virtual patient generation with realistic covariate distributions, individual outcome simulation via parametric survival models, and Bayesian adaptive interim analysis.

**Key capabilities**:
- Correlated covariate generation from published population statistics
- Weibull survival model with patient-specific hazard modification
- Kaplan-Meier and log-rank analysis
- Bayesian posterior probability monitoring with futility/efficacy stopping
- Power analysis across sample sizes
- Virtual control arm construction with propensity score matching

### 06: Digital Twin Validation & Verification Framework

**Problem**: FDA requires rigorous V&V demonstrating that DT predictions are accurate, calibrated, and generalizable before clinical use.

**Solution**: Comprehensive V&V pipeline implementing accuracy metrics (MAE, RMSE, C-index), calibration analysis (Hosmer-Lemeshow), discrimination analysis (AUC, ROC), subgroup equity analysis, robustness testing, and regulatory documentation generation.

**Key capabilities**:
- Harrell's concordance index with bootstrap confidence intervals
- Hosmer-Lemeshow calibration test and expected calibration error
- Full discrimination analysis (AUC, sensitivity, specificity, PPV, NPV, F1)
- Subgroup analysis across demographics for equity
- Input sensitivity/robustness analysis
- Model card generation (per Mitchell et al. 2019)
- ASME V&V 40 report generation for FDA submission

---

## Regulatory Standards Referenced

| Standard | Reference | Applicable Examples |
|----------|-----------|-------------------|
| FDA 21 CFR Part 11 | Electronic records and signatures | 01, 02, 05, 06 |
| ASME V&V 40 | Computational model V&V | 06 |
| FDA AI/ML SaMD Guidance | AI/ML medical device framework | 06 |
| FDA PCCP Guidance (Aug 2025) | Predetermined change control | 06 |
| IEC 62304 | Medical device software lifecycle | All |
| AAPM TG-132 | Image registration in RT | 03 |
| AAPM TG-275 | Dose accumulation in RT | 03 |
| CTCAE v5.0 | Common Terminology Criteria for AEs | 02 |
| iRECIST | Immune-modified response criteria | 04 |
| ICH E6(R3) | Good Clinical Practice (digital) | 01, 05 |
| ICH E9(R1) | Estimands and sensitivity analysis | 05 |

---

## Dependencies

All examples require only the base repository dependencies:

```
numpy>=1.24.0
scipy>=1.11.0
```

Optional dependencies enhance specific examples:
- `monai>=1.4.0` — imaging-based registration (03) and TIL scoring (04)
- `pydicom>=2.4.0` — DICOM RT structure loading (03)
- `lifelines` — Kaplan-Meier and Cox regression (05)
- `scikit-learn` — full ROC/AUC computation (06)

---

*This directory is part of the Physical AI Oncology Trials Unification Framework.*
