# Changelog

All notable changes to this repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.8.0] - 2026-02-07

### Added
- `tools/` directory: 5 standalone CLI utilities for physical AI oncology trial engineers
  - `tools/dicom-inspector/dicom_inspector.py`: DICOM file inspection, PHI audit across imaging directories, trial compliance validation (DICOM-BASE and DICOM-RT standards), and study-level summarization with modality distribution
  - `tools/dose-calculator/dose_calculator.py`: Radiotherapy dose calculations with BED, EQD2, TCP (Poisson and logistic models), NTCP (Lyman-Kutcher-Burman model with QUANTEC-derived organ presets), fractionation scheme comparison, and tissue alpha/beta reference tables
  - `tools/trial-site-monitor/trial_site_monitor.py`: Multi-site trial enrollment tracking, data quality scoring (completeness, query rates, protocol deviation rates, AE reporting delays), site status classification (green/yellow/red), and manifest template generation
  - `tools/sim-job-runner/sim_job_runner.py`: Cross-framework simulation job launcher supporting Isaac Lab, MuJoCo, PyBullet, and Gazebo with 6 oncology-relevant task definitions (needle insertion, tissue retraction, surgical reach, instrument handover, biopsy sampling, catheter navigation), framework auto-detection, and result comparison
  - `tools/deployment-readiness/deployment_readiness.py`: Pre-deployment AI model validation with ONNX compatibility checking, inference latency benchmarking (mean/P50/P95/P99), safety constraint verification, regulatory checklist generation (IEC 62304, FDA AI/ML PCCP, ISO 14971), and reference output validation
- `tools/README.md`: Documentation for all tools with usage examples, design principles, and dependency matrix

### Updated
- Main `README.md`: Added Command-Line Tools section with table and quick start; updated repository structure to include `tools/`

## [0.7.0] - 2026-02-06

### Added
- `digital-twins/examples-twins/` directory: 6 advanced digital twin engineering examples
  - `01_realtime_dt_synchronization.py`: Real-time DT synchronization via Extended Kalman Filter and particle filter (asynchronous multi-modal data fusion, anomaly detection via CUSUM, 21 CFR Part 11 audit trails)
  - `02_multi_organ_toxicity_twin.py`: Multi-organ toxicity digital twin with PBPK compartmental model (cardiac/renal/hepatic/neurological/hematologic toxicodynamics, CTCAE v5.0 grading, dose modification recommendations)
  - `03_adaptive_radiation_therapy_dt.py`: Adaptive radiation therapy DT with B-spline deformable image registration (dose accumulation on deforming anatomy, DVH metrics, BED/EQD2, replanning trigger detection per AAPM TG-132/TG-275)
  - `04_tumor_microenvironment_immunotherapy_dt.py`: Tumor microenvironment and immunotherapy response DT (9-variable ODE model of TME dynamics, PD-1/PD-L1 checkpoint axis, iRECIST classification, pseudoprogression detection, biomarker-driven response prediction)
  - `05_virtual_trial_cohort_dt.py`: Virtual clinical trial cohort DT (virtual patient generation, Weibull survival simulation, Bayesian adaptive interim analysis, power analysis, virtual control arm construction)
  - `06_dt_validation_verification.py`: Digital twin validation and verification framework (C-index, Hosmer-Lemeshow calibration, AUC discrimination, subgroup equity analysis, robustness testing, model card and V&V report generation per ASME V&V 40 and FDA AI/ML guidance)
- `digital-twins/examples-twins/README.md`: Documentation for all examples with regulatory standards cross-reference

### Updated
- `digital-twins/README.md`: Added examples-twins directory to structure and key capabilities
- Main `README.md`: Added Digital Twin Engineering Examples section with table and quick start
- Repository structure updated to reflect new directory

## [0.6.0] - 2026-02-06

### Added
- `examples-new/` directory: 6 comprehensive physical robot engineering examples
  - `01_realtime_safety_monitoring.py`: IEC 80601-2-77 compliant safety monitoring (force/torque limits, workspace boundaries, watchdog timers, force rate detection)
  - `02_sensor_fusion_intraoperative.py`: Multi-sensor perception pipeline (stereo/RGBD depth, instrument segmentation, tissue deformation tracking, temporal synchronization)
  - `03_ros2_surgical_deployment.py`: ROS 2 node architecture for surgical deployment (procedure state machine, policy inference, hardware interface for dVRK/Kinova/UR, real-time control loop)
  - `04_hand_eye_calibration_registration.py`: Spatial calibration (Tsai-Lenz hand-eye calibration, Arun SVD fiducial registration, ICP surface registration, verification with test points)
  - `05_shared_autonomy_teleoperation.py`: Surgeon-AI shared control (5 autonomy levels, virtual fixtures, command blending, haptic rendering, tremor filtering)
  - `06_robotic_sample_handling.py`: Laboratory automation for clinical trials (specimen pick-and-place, barcode verification, cold chain monitoring, 21 CFR Part 11 audit trails, batch processing)
- `examples-new/README.md`: Documentation for all new examples with hardware requirements, regulatory references, and usage instructions

### Updated
- Main `README.md`: Added `examples-new/` section with table of all new examples and quick start instructions
- Repository structure updated to reflect new directory

## [0.5.1] - 2026-02-04

### Added
- `.github/` directory with issue templates, PR template, and CI workflow
- `CITATION.cff` for machine-readable citation metadata
- `SECURITY.md`, `SUPPORT.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- `regulatory/human-oversight/` quality management document for CRF/AE automation
- Python lint/format CI via `ruff` and `yamllint`
- Illustrative-data disclaimers on all `results.md` benchmark tables

## [0.5.0] - 2026-02-04

### Added
- `privacy/` framework: PHI/PII detection, de-identification, access control, breach response, DUA templates
- `regulatory/` framework: FDA submission tracking, IRB management, ICH E6(R3) compliance, regulatory intelligence
- Privacy tooling covers all 18 HIPAA identifiers
- Regulatory tooling aligned with FDA AI/ML guidance (Jan 2025), ICH E6(R3) (Sep 2025), EU AI Act timelines

## [0.4.0] - 2026-02-02

### Added
- `digital-twins/` directory: patient modeling (TumorTwin), treatment simulation, clinical integration (FHIR/DICOM)
- `examples/` directory: 5 production-ready Python examples covering surgical training, digital twins, cross-framework validation, agentic workflows, and treatment prediction
- `q1-2026-standards/` directory: 3 unification objectives (bidirectional conversion, model repository, validation benchmarks)
- `configs/training_config.yaml` with domain randomization, safety limits, and deployment settings

### Updated
- Framework versions: Isaac Sim 5.0.0, Newton Physics Beta, MuJoCo Warp Beta, GR00T N1.6, Cosmos Predict 2.5, Cosmos Reason 2

## [0.3.1] - 2026-02-01

### Added
- Source citations across documentation to support framework/version claims

### Fixed
- Corrected outdated framework versions and related references (11 files modified; 140 insertions; 102 deletions)

## [0.3.0] - 2026-02-01

### Added
- `q1-2026-standards/` directory defining unification objectives:
  - Objective 1: Isaac <-> MuJoCo bidirectional conversion
  - Objective 2: Unified robot model repository (50+ models)
  - Objective 3: Validation benchmark suite v1.0

### Notes
- Includes an implementation guide with timeline and compliance checklist  
- Framework versions referenced: Isaac Lab 2.3.2, MuJoCo 3.4.0

## [0.2.0] - 2026-01-31

### Added
- Unification framework for framework-agnostic physical AI development for oncology clinical trials
- Multi-organization cooperation framing (release notes reference “February 2026” objectives)
- Adoption guidance spanning: (a) simulation physics, (b) agentic/generative AI, (c) surgical robots, (d) cross-platform tools

## [0.1.0] - 2026-01-31

### Added
- Initial repository structure
- `unification/` framework: Isaac-MuJoCo bridge, model converters, unified agent interface, cross-platform tools
- `frameworks/` integration guides: NVIDIA Isaac, MuJoCo, Gazebo, PyBullet
- Learning domain documentation: supervised, reinforcement, self-supervised, agentic, generative AI
- `scripts/verify_installation.py` for dependency checking
- `requirements.txt` with 30+ production dependencies
