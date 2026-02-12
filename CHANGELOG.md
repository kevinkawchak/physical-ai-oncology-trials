# Changelog

All notable changes to this repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.0.1] - 2026-02-12

### Added
- `DEVELOPMENT_PROPOSALS.md`: Three comprehensive prompt proposals for future Claude Code development — Proposal A (Comprehensive Test Suite), Proposal B (Multi-Site Federated Trial Coordination), Proposal C (Regulatory Submission Automation) — with feature-by-feature comparison tables, strategic impact matrix, and audience impact analysis
- **Comprehensive test suite**: 1,289+ tests across 54 test modules covering all 51 Python source modules, fully implementing Proposal A (Comprehensive Test Suite & Continuous Validation Infrastructure)
  - `tests/conftest.py`: Shared fixtures, mock data factories (synthetic tumor geometry, dose distributions, trial cohort config), and `importlib.util.spec_from_file_location()` loader for hyphenated directories; autouse RNG seeding (seed=42) for deterministic tests
  - **Root-level tests** (7 modules, 143 tests):
    - `tests/test_safety_monitoring.py` (15 tests): SafetyMonitor, ForceTorqueSensorProcessor, WorkspaceBoundaryGenerator
    - `tests/test_dose_calculator.py` (16 tests): BED, EQD2, TCP, NTCP, fractionation scheme parsing, tissue data
    - `tests/test_digital_twin_sync.py` (20 tests): EKF, particle filter, anomaly detection, synchronizer
    - `tests/test_mcp_server.py` (24 tests): MCP tool/resource handlers, audit trail, data models
    - `tests/test_calibration.py` (16 tests): Tsai-Lenz calibration, Arun SVD registration, transform math
    - `tests/test_sample_handling.py` (16 tests): Specimen model, barcode verification, cold chain
    - `tests/test_deidentification.py` (13 tests): Safe Harbor transforms, PHI detection, config
  - **`tests/test_digital_twins/`** (8 modules): Unit tests for all digital twin code
    - `test_tumor_twin_pipeline.py`: TumorType/ModelType/SolverType enums, PatientClinicalData, ModelParameters, LogisticGrowthModel, GompertzGrowthModel, MechanisticModel, PatientDigitalTwin
    - `test_treatment_simulator.py`: TreatmentType/ResponseType enums, TreatmentProtocol, TreatmentSimulator, LinearQuadraticModel, PharmacokineticModel, ImmunotherapyModel
    - `test_clinical_dt_interface.py`: ConnectionStatus, ComplianceRegulation, PatientRecord, ClinicalConnector, FHIRClient, ComplianceManager
    - `test_multi_organ_toxicity.py`: ChemoDrug, OrganSystem, CTCAEGrade, PBPKModel, CardiacToxicityModel, RenalToxicityModel, HepaticToxicityModel, MultiOrganToxicityTwin
    - `test_adaptive_radiation.py`: StructureType, DoseConstraint, BSplineRegistration, DoseAccumulator, AdaptiveRTDigitalTwin
    - `test_immunotherapy_dt.py`: ImmunePheno, CheckpointAgent, iRECISTResponse, TMEDynamicsModel, CheckpointPKModel, ImmunotherapyResponsePredictor
    - `test_virtual_trial_cohort.py`: TumorSite, TrialEndpoint, VirtualCohortGenerator, OutcomeSimulator, VirtualTrialSimulator, VirtualControlArmBuilder
    - `test_dt_validation.py`: ValidationLevel, RiskCategory, AccuracyMetrics, CalibrationAnalyzer, DiscriminationAnalyzer, SubgroupAnalyzer, RobustnessAnalyzer, VVReportGenerator
  - **`tests/test_agentic_ai/`** (5 modules): Unit tests for all agentic AI examples
    - `test_react_planner.py`: ProcedurePlanningTools, ReActProcedurePlanner, anatomy/instrument data models
    - `test_adaptive_treatment.py`: StreamBuffer, ForceTorqueProcessor, VitalsProcessor, CrossModalCorrelator, TreatmentDecisionEngine, AdaptiveTreatmentAgent
    - `test_simulation_orchestrator.py`: ExperimentDesigner, SimulationRunner, AnalysisEngine, SimulationOrchestrator
    - `test_safety_executor.py`: OncologyRoboticsConstraintLibrary, SafetyConstrainedExecutor, constraint checking
    - `test_rag_compliance.py`: RegulatoryKnowledgeBase, ComplianceVerifier, ProtocolRAGComplianceAgent
  - **`tests/test_tools/`** (4 modules): Unit tests for all CLI tools
    - `test_deployment_readiness.py`: ReadinessReport, deployment checks, regulatory checklists
    - `test_dicom_inspector.py`: InspectionResult, DICOM tag validation, PHI audit
    - `test_sim_job_runner.py`: JobResult, framework detection, task definitions
    - `test_trial_site_monitor.py`: SiteMetrics, enrollment tracking, quality scoring
  - **`tests/test_physical_robots/`** (6 modules): Unit tests for all robot examples
    - `test_sensor_fusion.py`: InstrumentSegmenter, TissueDeformationTracker, DepthToPointCloud, TemporalSynchronizer, SensorFusionPipeline
    - `test_ros2_deployment.py`: ProcedureStateMachine, PolicyInferenceEngine, RobotHardwareInterface, SurgicalControlLoop
    - `test_shared_autonomy.py`: VirtualFixtureEngine, CommandBlender, SharedAutonomyController, SurgeonInputProcessor
    - `test_surgical_training.py`: OncologySurgicalEnv, SurgicalPolicyNetwork, SurgicalPolicyTrainer, PolicyEvaluator
    - `test_surgical_planning.py`: SurgicalDigitalTwinBuilder, SurgicalDigitalTwin, VirtualSurgerySimulator
    - `test_treatment_prediction.py`: ExponentialGrowthModel, GompertzGrowthModel, TreatmentResponseModel, TreatmentOptimizer
  - **`tests/test_privacy/`** (4 modules): Unit tests for all privacy framework modules
    - `test_phi_detector.py`: PHICategory (18 HIPAA identifiers), PHIDetector scan/classification
    - `test_access_control.py`: Permission/UserType enums, AccessControlManager, audit trail copy guard
    - `test_breach_response.py`: IncidentType, RiskAssessment clamping, NotificationTimeline, BreachResponseManager
    - `test_dua_generator.py`: DUATemplate, DUAGenerator, jurisdiction handling
  - **`tests/test_regulatory/`** (4 modules): Unit tests for all regulatory framework modules
    - `test_fda_submission.py`: SubmissionType/Status/DeviceClass, FDASubmissionTracker, AI/ML component defaults
    - `test_irb_protocol.py`: ProtocolStatus, IRBProtocolManager, SubmissionChecklist completeness
    - `test_gcp_compliance.py`: GCPComplianceChecker, score excluding NOT_ASSESSED, ComplianceReport
    - `test_regulatory_tracker.py`: RegulatoryTracker, overdue/imminent status, cutoff date filtering
  - **`tests/test_unification/`** (5 modules): Unit tests for all unification framework modules
    - `test_isaac_mujoco_bridge.py`: PhysicsParameterMapper, StateConverter, IsaacMuJoCoBridge, PolicyTransferValidator
    - `test_urdf_converter.py`: URDFParser, MJCFGenerator, SDFGenerator, UnifiedModelConverter
    - `test_unified_agent.py`: UnifiedAgent, AgentTeam, OncologyToolkit, backend adapters
    - `test_framework_detector.py`: FrameworkDetector, FrameworkInfo, SystemInfo
    - `test_validation_suite.py`: MockEnvironment, PolicyLoader, CrossPlatformValidator
  - **`tests/test_standards/`** (3 modules): Unit tests for Q1 2026 standards
    - `test_isaac_to_mujoco.py`: PhysicsParameterConverter, URDFToMJCFConverter, IsaacToMuJoCoConverter
    - `test_benchmark_runner.py`: PhysicsBenchmark, PerformanceBenchmark, BenchmarkRunner
    - `test_model_validator.py`: FormatValidator, KinematicValidator, ModelValidator
  - **`tests/test_integration/`** (6 modules): Cross-module workflow tests
    - `test_dt_to_simulation.py`: Digital Twin → Treatment Simulation → Response Prediction flow
    - `test_agentic_to_regulatory.py`: Agentic AI decision → Regulatory audit trail → Compliance
    - `test_robot_to_safety.py`: Robot command → Safety monitoring → Emergency stop
    - `test_privacy_to_clinical.py`: Patient data → De-identification → Clinical utility preserved
    - `test_cross_framework.py`: Multi-framework simulation validation pipeline
    - `test_end_to_end_trial.py`: Full trial lifecycle: Patient → DT → Simulation → Regulatory
  - **`tests/test_regression/`** (2 modules): Comprehensive regression guards
    - `test_v092_guards.py`: 7 guards for critical v0.9.1/v0.9.2 bugs (EKF Jacobian, hazard ratio, division-by-zero, DoseResult truthiness)
    - `test_v092_comprehensive.py`: 28 additional guards for all remaining v0.9.2 fixes (bidirectional sync, bounded loops, overdue status, compliance scoring, format strings, audit log copy, date shift, weights_only, and more)
  - `tests/README.md`: Comprehensive testing strategy documentation with test organization tree, philosophy, coverage targets, and CI integration
  - `tests/__init__.py` and `__init__.py` in all 10 subdirectories

### Updated
- `.github/workflows/ci.yml`: Enhanced `test` job with pytest-cov dependency and test collection summary step
- `tests/conftest.py`: Added mock data factories (synthetic_tumor_geometry, synthetic_dose_distribution, trial_cohort_config)
- `tests/README.md`: Rewritten with full test tree, testing philosophy, coverage targets, and architecture docs

### Notes
- Fully implements Proposal A from `DEVELOPMENT_PROPOSALS.md` (Comprehensive Test Suite & Continuous Validation Infrastructure)
- 1,289+ tests pass with zero external dependencies beyond numpy, scipy, and pytest
- 2 tests skipped (optional framework availability checks)
- All tests pass `ruff format`, `ruff check`, and `py_compile` validation
- Tests use `importlib.util.spec_from_file_location()` to handle hyphenated directory names
- Mock-based isolation: all external dependencies (NVIDIA Isaac, MuJoCo, ROS 2, DICOM servers) mocked — tests run without GPU or hardware
- Deterministic RNG seeding (seed=42) ensures reproducible results across platforms and Python versions
- CI runs ruff format, ruff check, yamllint, py_compile, and pytest on Python 3.10–3.12
- Development by Claude Code Opus 4.6

## [1.0.0] - 2026-02-08

### Added
- `V1_RELEASE.md`: Comprehensive v1.0.0 release documentation covering community needs, technical achievements, version history, and v1.0.0 standards compliance
- Version badge (`v1.0.0`) added to README.md header
- v1.0.0 release summary block added to README.md with repository metrics (51 Python modules, 40,526 LOC, 69 docs, 28 examples, 5 CLI tools)
- `V1_RELEASE.md` added to repository structure in README.md

### Updated
- README.md: Added v1.0.0 designation, release summary, version badge, and updated citation block with `version = {1.0.0}`
- CHANGELOG.md: Consolidated all prior releases under v1.0.0 milestone

### Notes
- v1.0.0 designates the first stable release of the public API: directory structure, module interfaces, CLI tool contracts, and configuration formats
- Repository totals at v1.0.0: 66 commits, 12 merged pull requests, 65,287 insertions, 4,035 deletions, 160 project files across 61 directories
- Development primarily by Claude Code Opus 4.5/Opus 4.6; Claude Cowork Opus 4.5 for initial release; ChatGPT 5.2 Thinking/Agent for audit prompts and repo insights
- CI passes on Python 3.10, 3.11, and 3.12 (ruff lint, ruff format, yamllint, py_compile)
- Follows Semantic Versioning 2.0.0 and Keep a Changelog format

## [0.9.2] - 2026-02-08

### Fixed
- **Logic (CRITICAL)**: Fixed EKF Jacobian sign error in `digital-twins/examples-twins/01_realtime_dt_synchronization.py` (line 295: `1.0 + rate*dt` corrected to `1.0 - rate*dt`) causing divergent creatinine state estimates
- **Logic (CRITICAL)**: Fixed inverted hazard ratio calculation in `digital-twins/examples-twins/05_virtual_trial_cohort_dt.py` (line 743: `control/experimental` corrected to `experimental/control` per standard oncology convention where HR < 1 favors experimental arm)
- **Logic (CRITICAL)**: Fixed infinite `while not done: pass` loop in `unification/simulation_physics/isaac_mujoco_bridge.py` `_evaluate_policy()` that would hang indefinitely; replaced with bounded step loop
- **Logic (CRITICAL)**: Fixed `sync_state()` in `unification/simulation_physics/isaac_mujoco_bridge.py` only handling Isaac-to-MuJoCo direction; added MuJoCo-to-Isaac and MuJoCo-to-PyBullet branches and prevented false counter increment for unsupported frameworks
- **Logic**: Fixed unreachable "overdue" status branch in `regulatory/regulatory-intelligence/regulatory_tracker.py` where deadlines past due were mislabeled as "imminent" due to incorrect if/elif ordering
- **Logic**: Fixed GCP compliance score always reporting 0% in `regulatory/ich-gcp/gcp_compliance_checker.py` by excluding `NOT_ASSESSED` findings from the scoring denominator
- **Logic**: Fixed format string bug `%.1%%` in `digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py` (line 664) causing `TypeError` at runtime; corrected to `%.1f%%`
- **Logic**: Fixed division by zero in `digital-twins/patient-modeling/tumor_twin_pipeline.py` `LogisticGrowthModel.simulate()` when initial condition sums to zero (post-resection scenarios)
- **Logic**: Fixed division by zero in `tumor_twin_pipeline.py` `predict()` volume change calculation when baseline volume is zero
- **Logic**: Fixed floating-point equality comparison in `digital-twins/treatment-simulation/treatment_simulator.py` surgery day check (line 372) that could miss the surgery timepoint due to `np.linspace` precision
- **Logic**: Fixed MJCF parsing incorrectly falling back to URDF parser in `unification/simulation_physics/urdf_sdf_mjcf_converter.py`; now raises `NotImplementedError` with guidance to use dedicated conversion pipelines
- **Logic**: Fixed `sim_job_runner.py` `cmd_launch_all` iterating all frameworks including unavailable ones despite computing and displaying `target_frameworks`
- **Logic**: Fixed `dose_calculator.py` truthiness checks (`if self.bed_gy:`) that silently dropped valid zero-value results from `DoseResult.to_dict()`; changed to `is not None` checks
- **Logic**: Fixed `dose_calculator.py` CLI falsy-value check replacing explicit `--alpha-beta 0` and `--volume 0` inputs with defaults
- **Logic**: Fixed `validation_suite.py` success rate always reporting ~25% because threshold was computed as 75th percentile of the same rewards array; replaced with fixed task-appropriate threshold
- **Runtime (CRITICAL)**: Fixed `TypeError` crash in `privacy/access-control/access_control_manager.py` demo where `assign_role()` was called with unsupported `mfa_enrolled` keyword argument
- **Security**: Changed `torch.load()` to `torch.load(weights_only=True)` in `unification/cross_platform_tools/validation_suite.py` to prevent arbitrary code execution via pickle deserialization
- **Security**: Fixed `access_control_manager.py` `get_audit_log()` returning a reference to the internal audit log list; now returns a copy to prevent external mutation of audit trail
- **Security**: Fixed `access_control_manager.py` silently granting access when `access_expiration` date format is invalid; now logs error and denies access by default
- **Compliance**: Fixed `deidentification_pipeline.py` `DATE_SHIFT` handling silently falling through to date removal; added explicit `DATE_SHIFT` branch with appropriate logging
- **Compliance**: Fixed `fda_submission_tracker.py` defaulting all AI/ML components to `model_type="classification"`; changed to `"unspecified"` since component type should be explicitly specified
- **Compliance**: Fixed `deployment_readiness.py` safety constraints always reporting "passed" without checking actual model outputs; now reports `requires_runtime_verification` status
- **Compliance**: Fixed `deployment_readiness.py` identical ternary branches for multi-input model validation; both branches produced single-input feed dict
- **Compliance**: Added `RESEARCH USE ONLY` disclaimers to 11 modules: `deidentification_pipeline.py`, `phi_detector.py`, `access_control_manager.py`, `breach_response_protocol.py`, `dua_generator.py`, `fda_submission_tracker.py`, `irb_protocol_manager.py`, `gcp_compliance_checker.py`, `regulatory_tracker.py`, `tumor_twin_pipeline.py`, `treatment_simulator.py`, `dose_calculator.py`
- **Lint**: Added missing `import logging` and `logger` to `isaac_mujoco_bridge.py`; removed unused `Union` import
- **Format**: Auto-formatted `deidentification_pipeline.py` and `deployment_readiness.py` to pass `ruff format --check`

### Notes
- Comprehensive logic, context, and compliance audit of 51 Python files across all modules
- CI lint-and-format checks pass for Python 3.10, 3.11, and 3.12
- ChatGPT 5.2 Thinking Agent assisted with this audit prompt

## [0.9.1] - 2026-02-08

### Fixed
- **Security**: Replaced weak default pseudonymization salt (`"default_salt"`) in `privacy/de-identification/deidentification_pipeline.py` with cryptographically random salt generation via `os.urandom`; logs a warning when no explicit `hash_salt` is configured
- **Security**: Changed `numpy.load(allow_pickle=True)` to `allow_pickle=False` in `tools/deployment-readiness/deployment_readiness.py` to prevent arbitrary code execution from untrusted `.npz` files
- **Logic**: Fixed `RiskAssessment.calculate_risk()` in `privacy/breach-response/breach_response_protocol.py` to clamp out-of-range scores instead of silently returning and leaving the object in an inconsistent state
- **Logic**: Added missing `peak_cd8` and `peak_ifng` keys to `predict_response()` return dict in `digital-twins/examples-twins/04_tumor_microenvironment_immunotherapy_dt.py`, fixing a `KeyError` in the demo main block
- **Logic**: Fixed dead-code multiplication by `0.0` for renal elimination in `digital-twins/examples-twins/02_multi_organ_toxicity_twin.py` PBPK kidney compartment ODE
- **Logic**: Fixed `get_recent_updates()` in `regulatory/regulatory-intelligence/regulatory_tracker.py` to actually use the computed `cutoff` date for filtering
- **Logic**: Added whitespace stripping to comma-separated framework parsing in `unification/cross_platform_tools/validation_suite.py`
- **Type safety**: Added `from __future__ import annotations` to `regulatory/irb-management/irb_protocol_manager.py` to resolve forward reference of `SubmissionChecklist`
- **Type hint**: Added return type `-> int` to `main()` in `scripts/verify_installation.py`
- **Imports**: Removed unused `import re` from `unification/simulation_physics/urdf_sdf_mjcf_converter.py`
- **Imports**: Removed unused `from abc import ABC, abstractmethod` from `digital-twins/clinical-integration/clinical_dt_interface.py`
- **Imports**: Removed unused `import yaml` from `q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py`
- **Imports**: Removed unused `import yaml` and `import warnings` from `q1-2026-standards/objective-2-robot-model-repository/model_validator.py`
- **Formatting**: Fixed missing space in output string in `tools/deployment-readiness/deployment_readiness.py`
- **YAML**: Split long comment line in `unification/simulation_physics/physics_parameter_mapping.yaml` to resolve yamllint line-length warning

### Notes
- Full static analysis audit of 51 Python files, 5 YAML files, and 47+ Markdown files
- CI lint-and-format checks pass for Python 3.10, 3.11, and 3.12
- ChatGPT 5.2 Thinking Agent assisted with this audit prompt

## [0.9.0] - 2026-02-07

### Added
- `agentic-ai/examples-agentic-ai/` directory: 6 comprehensive agentic AI engineering examples for robotic oncology trials
  - `01_mcp_clinical_robotics_server.py`: Model Context Protocol (MCP) server exposing robot telemetry, DICOM imaging, patient vitals, and procedure management as structured tools and resources with 21 CFR Part 11 audit trails, keep-out zone enforcement, and WHO-adapted surgical safety checklist
  - `02_react_procedure_planner.py`: ReAct (Reasoning + Acting) agent for surgical procedure planning with chain-of-thought reasoning, patient-specific anatomy integration, instrument selection, approach safety evaluation, margin estimation, and contingency planning across lobectomy, nephrectomy, and prostatectomy protocols
  - `03_realtime_adaptive_treatment_agent.py`: Real-time adaptive treatment agent processing streaming multi-modal data (force/torque, patient vitals, intraoperative imaging) with cross-modal correlation engine detecting hemorrhage, hemodynamic instability, and resection boundary concerns, generating prioritized treatment recommendations
  - `04_autonomous_simulation_orchestrator.py`: Autonomous agent that designs, configures, runs, and analyzes simulation experiment campaigns across Isaac Lab, MuJoCo, PyBullet, and Gazebo with parameter sensitivity analysis, cross-framework consistency checks, hypothesis evaluation, and iterative refinement
  - `05_safety_constrained_agent_executor.py`: Formal safety constraint framework for agentic control of surgical robots with pre-condition/post-condition verification, runtime invariant monitoring, safety gate human-in-the-loop approval, constraint library aligned to IEC 80601-2-77 and ISO 14971, and rollback mechanisms
  - `06_protocol_rag_compliance_agent.py`: Retrieval-Augmented Generation (RAG) agent grounding clinical decisions in trial protocols, FDA guidance, ICH E6(R3), IEC standards, and institutional SOPs with keyword-based document retrieval, compliance verification, cited regulatory responses, and audit trail

### Updated
- `ruff.toml`: Added per-file ignore rules for `agentic-ai/**/*.py`
- Main `README.md`: Added Agentic AI Engineering Examples section with table and quick start
- Repository structure updated to include `agentic-ai/examples-agentic-ai/`

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
