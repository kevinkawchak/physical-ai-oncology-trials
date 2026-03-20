# v2.6.0 Development Prompts

## Prompt 1: Single-Patient Journey Orchestration

Build a complete single-patient journey system (v2.6.0) in the physical-ai-oncology-trials repository, tracing Patient PAT-2026-0042 (58F, Stage IIIB NSCLC, ECOG 1, PD-L1 65%, TMB 14 mut/Mb, SITE-003) through 10 stages of a Physical AI oncology trial.

### Stages

1. **Pre-Screening & Referral Intake** (Day -30 to Day -14): PHI detection, HIPAA Safe Harbor de-identification, ICD-10 to SNOMED harmonization, DICOM validation, patient record creation, access provisioning
2. **Enrollment & Informed Consent** (Day -14 to Day 0): ICH E6(R3) consent elements, eligibility criteria evaluation, duplicate enrollment check, protocol compliance verification, IRB review, stratified randomization
3. **Digital Twin Construction** (Day 0 to Day 7): ASME V&V 40 validation, tumor microenvironment modeling, adaptive radiation simulation, virtual cohort analysis, real-time sync establishment
4. **Robot Qualification** (Day 7 to Day 13): USL scoring (4 dimensions), cross-framework validation, cybersecurity assessment, hand-eye calibration, deployment readiness
5. **Robot-Assisted Surgery** (Day 14): ROS 2 deployment, pre-procedure safety matrix, shared autonomy, sensor fusion, sim-vs-real validation, specimen chain of custody
6. **Post-Operative Recovery** (Day 14 to Day 28): Digital twin sync transition, pathology integration (pT2aN2M0), adverse event tracking, Physical AI causality assessment
7. **Immunotherapy Treatment** (Day 28 to Day 763): 35 cycles pembrolizumab 200mg q3w, adaptive dosing, cumulative toxicity tracking, annual reporting
8. **Federated Learning** (Day 28 to Day 763): 70 rounds federated averaging, differential privacy (epsilon=1.0, delta=1e-5), secure aggregation, DSMB reporting
9. **Long-Term Surveillance** (Day 763 to Day 1858): Complete response, quarterly imaging, recurrence risk modeling (35% to 3%)
10. **Trial Closeout** (Day 1858+): HARD_LOCK, re-identification risk validation, GCP audit, regulatory package generation

### Regulatory Frameworks

- 21 CFR Part 312 Subpart J (sections 312.400-405)
- 21 CFR Part 50 Subpart C (sections 50.30-34)
- ICH E6(R3) (sections 1.2-1.5, 2.8-2.12)

### Deliverables

- 12 Python orchestrator modules in `patient-journey/`
- 30 ASCII progress diagrams (3 perspectives x 10 stages)
- 10 Plotly chart generators
- 6 text tables
- FDA cost-savings analysis
- 4 guidance documents (pharmaceutical industry, field observer, site activation, patient information)
- 262+ tests across 14 test modules
- Master generator script

### Repository Conventions

- Python 3.10+, 120 char line length, ruff linting
- np.random.seed(42) in test fixtures
- importlib.util.spec_from_file_location() for hyphenated directory imports
- No logos, no section symbols except for regulatory titles
- Em dashes only for regulatory titles
- DOI: [10.5281/zenodo.19119939](https://doi.org/10.5281/zenodo.19119939)

### Release Metadata

- Draft release
- Released on 20 March 2026
- CEO Kevin Kawchak, ChemicalQDevice
- Development by Claude Code Opus 4.6
