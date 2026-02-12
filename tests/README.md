# Test Suite — Physical AI Oncology Trials

## Overview

Comprehensive pytest-based test suite providing unit test coverage for all 51
Python modules, integration tests for cross-module workflows, and regression
guards for 30+ bugs fixed in v0.9.1/v0.9.2. Tests are organized into
subdirectories mirroring the source structure.

**Key metrics:**
- 50+ test modules across 11 subdirectories
- Unit tests for every Python module in the repository
- 6 end-to-end integration test scenarios
- 30+ regression tests for v0.9.1/v0.9.2 bug fixes
- Mock-based isolation — no hardware, GPU, or network required
- Deterministic — NumPy RNG seeded to 42 for reproducible results
- CI-validated across Python 3.10, 3.11, 3.12

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific subdirectory
pytest tests/test_digital_twins/ -v
pytest tests/test_agentic_ai/ -v
pytest tests/test_tools/ -v
pytest tests/test_physical_robots/ -v
pytest tests/test_integration/ -v
pytest tests/test_regression/ -v

# Run a single test module
pytest tests/test_tools/test_dose_calculator.py -v

# Run tests matching a pattern
pytest tests/ -k "safety" -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# List all discovered tests
pytest tests/ --co -q
```

## Requirements

- Python 3.10+
- numpy, scipy (already in project requirements)
- pytest, pytest-cov

```bash
pip install pytest pytest-cov
```

## Directory Structure

```
tests/
├── conftest.py                    # Root conftest: load_module(), mock factories, fixtures
├── __init__.py
├── README.md                      # This file
│
├── test_digital_twins/            # 6 modules — digital twin framework
│   ├── test_tumor_twin_pipeline.py
│   ├── test_treatment_simulator.py
│   ├── test_clinical_dt_interface.py
│   ├── test_multi_organ_toxicity.py
│   ├── test_virtual_trial_cohort.py
│   └── test_dt_validation.py
│
├── test_agentic_ai/               # 6 modules — agentic AI examples
│   ├── test_mcp_server.py
│   ├── test_react_planner.py
│   ├── test_adaptive_treatment.py
│   ├── test_simulation_orchestrator.py
│   ├── test_safety_executor.py
│   └── test_protocol_rag.py
│
├── test_tools/                    # 5 modules — CLI tools
│   ├── test_dose_calculator.py
│   ├── test_dicom_inspector.py
│   ├── test_sim_job_runner.py
│   ├── test_trial_site_monitor.py
│   └── test_deployment_readiness.py
│
├── test_physical_robots/          # 6 modules — physical robot examples
│   ├── test_safety_monitoring.py
│   ├── test_sensor_fusion.py
│   ├── test_ros2_deployment.py
│   ├── test_calibration.py
│   ├── test_shared_autonomy.py
│   └── test_sample_handling.py
│
├── test_integration/              # 6 cross-module workflow tests
│   ├── test_twin_to_simulation.py
│   ├── test_agentic_decision_workflow.py
│   ├── test_clinical_data_pipeline.py
│   ├── test_safety_monitoring_chain.py
│   ├── test_regulatory_audit_trail.py
│   └── test_cross_framework_validation.py
│
├── test_regression/               # Regression guards for v0.9.1/v0.9.2
│   ├── test_v092_critical.py      # EKF Jacobian, hazard ratio, infinite loop
│   ├── test_v092_logic.py         # Division by zero, truthiness, format strings
│   ├── test_v091_security.py      # Salt hardcoding, pickle, audit log leak
│   └── test_v092_compliance.py    # DATE_SHIFT, model_type, RESEARCH disclaimers
│
├── test_unification/              # 4 modules — unification framework
│   ├── test_unified_agent_interface.py
│   ├── test_isaac_mujoco_bridge.py
│   ├── test_urdf_sdf_mjcf_converter.py
│   └── test_framework_detector.py
│
├── test_privacy/                  # 4 modules — privacy framework
│   ├── test_phi_detector.py
│   ├── test_access_control.py
│   ├── test_breach_response.py
│   └── test_dua_generator.py
│
├── test_regulatory/               # 4 modules — regulatory framework
│   ├── test_fda_submission_tracker.py
│   ├── test_irb_protocol_manager.py
│   ├── test_gcp_compliance_checker.py
│   └── test_regulatory_tracker.py
│
├── test_q1_standards/             # 3 modules — Q1 2026 standards
│   ├── test_isaac_to_mujoco_pipeline.py
│   ├── test_model_validator.py
│   └── test_benchmark_runner.py
│
├── test_scripts/                  # 1 module — utility scripts
│   └── test_verify_installation.py
│
└── (legacy flat test files)       # Original 8 test files retained
    ├── test_safety_monitoring.py
    ├── test_dose_calculator.py
    ├── test_digital_twin_sync.py
    ├── test_mcp_server.py
    ├── test_calibration.py
    ├── test_sample_handling.py
    ├── test_deidentification.py
    └── test_regression.py
```

## Architecture

### Import Strategy

Source modules live in hyphenated directories (`examples-new/`, `digital-twins/`, etc.)
that cannot be imported with standard Python `import`. The `conftest.py` provides
a `load_module()` helper that uses `importlib.util.spec_from_file_location()` to
load modules by file path.

### Fixture Hierarchy

```
conftest.py (root)
├── PatientFactory     — synthetic patient data, tumor volumes, dose distributions
├── TrialFactory       — trial designs, site manifests
├── RobotFactory       — robot state data, force/torque readings
├── Module fixtures    — load_module() wrappers for all 51 source modules
└── Data fixtures      — baseline_patient_state, standard_fractionation, etc.
```

### Mock-Based Isolation

All hardware and cloud dependencies (NVIDIA Isaac, MuJoCo, ROS 2, DICOM servers,
MCP SDK) are either optional imports in the source or not needed for the tested
code paths. Tests run without GPU, robot hardware, or network access.

### Clinical Safety Validation

Tests include boundary-condition checks for:
- Dose calculations (zero dose, single fraction, BED/EQD2 limits)
- Force/torque safety limits (emergency stop thresholds)
- Workspace boundary enforcement
- Emergency stop and reset cycles
- Hazard ratio directionality (oncology convention)
- Division-by-zero guards in post-resection scenarios

### Regression Testing Philosophy

Every bug fixed in v0.9.1 and v0.9.2 has a corresponding regression test.
Tests are named after the component and bug type, and reference the specific
CHANGELOG entry. This ensures:

1. The bug cannot recur without test failure
2. New contributors understand the historical context
3. CI catches regressions before merge

## Coverage Targets

| Area | Target | Notes |
|------|--------|-------|
| Digital twins | All 6 example modules + 3 core modules | Enums, dataclasses, growth models |
| Agentic AI | All 6 example modules | MCP, ReAct, safety constraints, RAG |
| CLI tools | All 5 tools | Constants, dataclasses, calculations |
| Physical robots | All 6 example modules | Safety, sensors, calibration, autonomy |
| Unification | All 4 modules | Physics bridge, converters, detector |
| Privacy | All 4 modules (+ deidentification) | PHI detection, access control, DUA |
| Regulatory | All 4 modules | FDA, IRB, GCP, regulatory intelligence |
| Q1 Standards | All 3 objective modules | Conversion, validation, benchmarks |
| Integration | 6 cross-module workflows | End-to-end pipelines |
| Regression | 30+ v0.9.1/v0.9.2 bugs | Critical, logic, security, compliance |

## Adding New Tests

1. Identify the source module's subdirectory category
2. Create or extend the corresponding `tests/test_<category>/test_<module>.py`
3. Use `load_module()` from `conftest.py` to import the source
4. Follow the class-per-feature pattern used in existing tests
5. Run `ruff format tests/` and `ruff check tests/` before committing
6. Verify with `pytest tests/ -v --tb=short`

## CI Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
1. **lint-and-format**: `ruff check` + `ruff format --check` across Python 3.10–3.12
2. **validate-scripts**: `py_compile` syntax checking
3. **test**: `pytest tests/ -v --tb=short` across Python 3.10–3.12
