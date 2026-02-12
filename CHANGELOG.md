# Changelog

All notable changes to this repository are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.0.1] - 2026-02-12

### Added
- `DEVELOPMENT_PROPOSALS.md`: Three comprehensive prompt proposals for future Claude Code development — Proposal A (Comprehensive Test Suite), Proposal B (Multi-Site Federated Trial Coordination), Proposal C (Regulatory Submission Automation) — with feature-by-feature comparison tables, strategic impact matrix, and audience impact analysis
- `tests/` directory: Comprehensive pytest-based test suite with **566 tests** across **11 subdirectories + 8 legacy modules**, all passing on Python 3.10–3.12
  - `tests/conftest.py`: Enhanced shared fixtures with `load_module()` helper using `importlib.util.spec_from_file_location()` for hyphenated directories; `PatientFactory`, `TrialFactory`, `RobotFactory` mock data factories; 40+ module-loading fixtures; autouse RNG seeding (seed=42) for deterministic tests
  - `tests/test_digital_twins/` (73 tests across 6 modules):
    - `test_tumor_twin_pipeline.py`: TumorType/ModelType/SolverType enums, PatientClinicalData/ModelParameters/TwinPrediction dataclasses, LogisticGrowthModel simulate/zero-volume/growth-bounds, GompertzGrowthModel, create_pipeline factory
    - `test_treatment_simulator.py`: TreatmentType/ResponseType enums, TreatmentProtocol/TreatmentResponse dataclasses, LinearQuadraticModel surviving_fraction/BED/TCP/tissue_params, PharmacokineticModel, TreatmentSimulator predict_response
    - `test_clinical_dt_interface.py`: ConnectionStatus/ComplianceRegulation enums, PatientRecord/ImagingStudy/AuditEntry dataclasses, ClinicalConnector, ComplianceManager
    - `test_multi_organ_toxicity.py`: ChemoDrug/OrganSystem/CTCAEGrade enums, DRUG_PBPK_LIBRARY, PBPKModel ODE system, MultiOrganToxicityTwin simulate_cycle/organ_states/CTCAE grading
    - `test_virtual_trial_cohort.py`: TumorSite/TrialEndpoint enums, VirtualPatient, PopulationParameters, VirtualCohortGenerator deterministic generation, VirtualTrialSimulator
    - `test_dt_validation.py`: ValidationLevel/RiskCategory enums, PredictionPair, ValidationDataset, AccuracyMetrics (MAE, RMSE, R², MAPE, C-index, compute_all)
  - `tests/test_agentic_ai/` (60 tests across 6 modules):
    - `test_mcp_server.py`: RobotMode/ProcedurePhase enums, RobotTelemetry, ClinicalRoboticsMCPTools/Resources definitions, async handle_tool_call (telemetry, commands, velocity clamping, out-of-bounds rejection), audit trail export
    - `test_react_planner.py`: TumorType/RiskLevel enums, AnatomicalStructure, PatientAnatomy, SurgicalInstrument, ProcedureStep, ProcedurePlanningTools instrument catalog
    - `test_adaptive_treatment.py`: StreamType/AlertSeverity ordering, DataSample, AnomalyEvent, TreatmentRecommendation, StreamBuffer
    - `test_simulation_orchestrator.py`: SimFramework/ExperimentStatus/SamplingStrategy enums, ParameterRange sample_values, SimulationConfig, SimulationResult, ExperimentDesign
    - `test_safety_executor.py`: ConstraintType/ConstraintSeverity/ViolationResponse/ActionCategory enums, SafetyConstraint check, ConstraintResult, AgentAction, SafetyVerdict
    - `test_protocol_rag.py`: DocumentType/ComplianceStatus enums, DocumentChunk get_citation, RetrievalResult, ComplianceCheck, RegulatoryKnowledgeBase retrieve
  - `tests/test_tools/` (50 tests across 5 modules):
    - `test_dose_calculator.py`: TISSUE_ALPHA_BETA constants, calc_bed/calc_eqd2/calc_tcp_poisson/calc_ntcp_lkb functions, DoseResult, _parse_scheme
    - `test_dicom_inspector.py`: PHI_TAGS/REQUIRED_TRIAL_TAGS/ONCOLOGY_MODALITIES constants, InspectionResult
    - `test_sim_job_runner.py`: TASK_REGISTRY/SUPPORTED_FRAMEWORKS constants, JobResult, _generate_job_id uniqueness
    - `test_trial_site_monitor.py`: DEFAULT_THRESHOLDS/SITE_STATUS constants, SiteMetrics, compute_site_metrics
    - `test_deployment_readiness.py`: REGULATORY_CHECKLIST/SAFETY_CATEGORIES constants, _generate_checklist IEC 62304/FDA AI/ML items
  - `tests/test_physical_robots/` (61 tests across 6 modules):
    - `test_safety_monitoring.py`: SafetyLevel/StopCategory enums, SafetyLimits defaults, SafetyMonitor check_and_filter (nominal/force/workspace/velocity), reset_stop, event_log, ForceTorqueSensorProcessor bias calibration, WorkspaceBoundaryGenerator compute_workspace_from_tumor
    - `test_sensor_fusion.py`: SensorType enum, StereoImagePair, DepthImage, PointCloud, TrackedPose
    - `test_ros2_deployment.py`: ProcedurePhase enum, PhaseConfig, PHASE_CONFIGS constants
    - `test_calibration.py`: make_transform, rotation_error_deg, translation_error_mm, HandEyeCalibrator add_pose/calibrate, PatientRegistration add_fiducial/register_fiducial, CalibrationVerifier add_test_point/verify
    - `test_shared_autonomy.py`: AutonomyLevel enum, SharedAutonomyConfig, SurgeonInputProcessor
    - `test_sample_handling.py`: SpecimenType/ContainerType/StorageCondition enums, Specimen record_event/audit_trail, BarcodeVerifier scan_and_verify, ColdChainMonitor check_temperature, SpecimenRobotController pick_specimen
  - `tests/test_integration/` (22 tests across 6 modules):
    - `test_twin_to_simulation.py`: Digital twin creation → growth simulation → treatment response pipeline
    - `test_agentic_decision_workflow.py`: Safety constraint check → async MCP handle_tool_call → audit trail chain
    - `test_clinical_data_pipeline.py`: DICOM constants → SafeHarborTransformer deidentification → PHI detection
    - `test_safety_monitoring_chain.py`: Nominal → critical transition, multiple sensor readings, reset_stop after emergency
    - `test_regulatory_audit_trail.py`: FDA submission lifecycle, GCP verify_compliance, access control define_role/assign_role/check_access audit
    - `test_cross_framework_validation.py`: Physics parameter round-trip, contact conversion, framework detection
  - `tests/test_regression/` (30 tests across 4 modules):
    - `test_v092_critical.py`: EKF Jacobian sign error, inverted hazard ratio convention
    - `test_v092_logic.py`: Division by zero guards, DoseResult truthiness, GCP scoring excludes NOT_ASSESSED, MJCF parsing, sim job runner
    - `test_v091_security.py`: DeidentificationPipeline salt randomization, audit log reference leak, access expiration denial, deployment safety
    - `test_v092_compliance.py`: DATE_SHIFT handling, AIMLComponent model_type default, FDA status values, regulatory tracker ordering, RESEARCH USE disclaimers
  - `tests/test_unification/` (32 tests across 4 modules):
    - `test_unified_agent_interface.py`: AgentBackend/AgentRole enums, Tool to_mcp/openai/anthropic format, AgentMessage, AgentConfig
    - `test_isaac_mujoco_bridge.py`: PhysicsParameters, ContactParameters to_isaac_physx/to_mujoco_solref/to_pybullet, PhysicsParameterMapper, StateConverter
    - `test_urdf_sdf_mjcf_converter.py`: Link, Joint, RobotModel dataclasses, URDFParser
    - `test_framework_detector.py`: FrameworkDetector.FRAMEWORKS, FrameworkInfo, SystemInfo, detect_all
  - `tests/test_privacy/` (39 tests across 4 modules):
    - `test_phi_detector.py`: PHICategory (18 members), RiskLevel, PHI_PATTERNS, PHIFinding, ScanResult, PHIDetector scan_text
    - `test_access_control.py`: Permission/UserType enums, Role has_permission, AccessDecision, AccessControlManager define_role/assign_role/check_access/audit_log
    - `test_breach_response.py`: IncidentType, RiskLevel, NotificationRecipient, Incident, RiskAssessment calculate_risk/clamping
    - `test_dua_generator.py`: DUATemplate, Jurisdiction, SecurityLevel, DUAParty, DataDescription, DUADocument
  - `tests/test_regulatory/` (31 tests across 4 modules):
    - `test_fda_submission_tracker.py`: SubmissionType, SubmissionStatus, DeviceClass, AIMLComponent model_type, Submission, ChecklistItem
    - `test_irb_protocol_manager.py`: ProtocolStatus, ReviewType, RiskCategory, AIComponent, ConsentElement, Protocol
    - `test_gcp_compliance_checker.py`: GuidelineVersion, ComplianceStatus, FindingSeverity, ComplianceFinding, GCP_REQUIREMENTS, GCPComplianceChecker verify_compliance
    - `test_regulatory_tracker.py`: RegulatoryJurisdiction, UpdateType, ImpactLevel, RegulatoryUpdate, REGULATORY_TIMELINE
  - `tests/test_q1_standards/` (20 tests across 3 modules):
    - `test_isaac_to_mujoco_pipeline.py`: PhysXContactParams, MuJoCoContactParams, ConversionConfig, PhysicsParameterConverter physx_to_mujoco/mujoco_to_physx
    - `test_model_validator.py`: ValidationResult, ModelValidationReport add_result/to_dict
    - `test_benchmark_runner.py`: SCENARIOS, BenchmarkConfig, BenchmarkResult, BenchmarkReport
  - `tests/test_scripts/` (5 tests):
    - `test_verify_installation.py`: REQUIREMENTS/OPTIONAL constants, check_version return tuple validation
  - Legacy flat test files retained (143 tests across 8 modules): `test_safety_monitoring.py`, `test_dose_calculator.py`, `test_digital_twin_sync.py`, `test_mcp_server.py`, `test_calibration.py`, `test_sample_handling.py`, `test_deidentification.py`
  - `tests/__init__.py` and per-subdirectory `__init__.py` package markers
  - `tests/README.md`: Comprehensive testing strategy documentation with directory structure, architecture notes, fixture hierarchy, coverage targets, and CI integration guide

### Updated
- `.github/workflows/ci.yml`: Added `needs: [lint-and-format]` dependency for test job, added `pytest-cov` to test dependencies, added test count verification step
- `ruff.toml`: Added `tests/**/*.py` ignore rules for `F401` and `F811` to accommodate test fixture imports

### Notes
- Implements Proposal A (Comprehensive Test Suite & Continuous Validation Infrastructure) from `DEVELOPMENT_PROPOSALS.md`
- All 566 tests pass with zero external dependencies beyond numpy, scipy, and pytest
- 50+ test modules across 11 subdirectories mirroring the source structure
- Unit tests cover all 51 Python modules in the repository
- 6 end-to-end integration test scenarios validating cross-module workflows
- 30+ regression tests guarding v0.9.1/v0.9.2 bug fixes
- Mock-based isolation — no hardware, GPU, or network required
- Tests use `importlib.util.spec_from_file_location()` to handle hyphenated directory names
- Deterministic RNG seeding (seed=42) ensures reproducible results across platforms
- CI now runs ruff format, ruff check, yamllint, py_compile, and pytest with coverage across Python 3.10–3.12
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
