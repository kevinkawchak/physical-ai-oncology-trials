# Test Suite -- Physical AI Oncology Trials

**Version**: 2.7.1
**Last Updated**: March 2026

## Overview

Comprehensive pytest-based test suite providing **1,289+ tests** across **54 test modules**
covering all 51 Python source modules in the repository. Tests include unit tests for
every module, integration tests for cross-module workflows, and regression guards for
all bugs fixed in v0.9.1/v0.9.2.

All tests run without GPU, robot hardware, external services, or network access through
mock-based isolation of all external dependencies.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test category
pytest tests/test_digital_twins/ -v
pytest tests/test_agentic_ai/ -v
pytest tests/test_integration/ -v

# Run a single test module
pytest tests/test_dose_calculator.py -v

# Run tests matching a pattern
pytest tests/ -k "safety" -v

# Run with coverage
pytest tests/ -v --cov --cov-report=term-missing
```

## Requirements

- Python 3.10, 3.11, or 3.12
- numpy, scipy (already in project requirements)
- pytest (>= 7.4.0)

```bash
pip install pytest numpy scipy
```

## Test Organization

```
tests/
├── conftest.py                     # Shared fixtures, mock factories, load_module()
├── __init__.py                     # Package marker
│
├── test_safety_monitoring.py       # examples-new/01 — SafetyMonitor, F/T, workspace
├── test_dose_calculator.py         # tools/dose-calculator — BED, EQD2, TCP, NTCP
├── test_digital_twin_sync.py       # digital-twins/examples-twins/01 — EKF, PF, anomaly
├── test_mcp_server.py              # agentic-ai/examples-agentic-ai/01 — MCP tools/resources
├── test_calibration.py             # examples-new/04 — Tsai-Lenz, Arun SVD, transforms
├── test_sample_handling.py         # examples-new/06 — specimen, barcode, cold chain
├── test_deidentification.py        # privacy/de-identification — Safe Harbor, PHI
│
├── test_digital_twins/             # 8 modules covering all digital twin code
│   ├── test_tumor_twin_pipeline.py
│   ├── test_treatment_simulator.py
│   ├── test_clinical_dt_interface.py
│   ├── test_multi_organ_toxicity.py
│   ├── test_adaptive_radiation.py
│   ├── test_immunotherapy_dt.py
│   ├── test_virtual_trial_cohort.py
│   └── test_dt_validation.py
│
├── test_agentic_ai/                # 5 modules covering all agentic AI examples
│   ├── test_react_planner.py
│   ├── test_adaptive_treatment.py
│   ├── test_simulation_orchestrator.py
│   ├── test_safety_executor.py
│   └── test_rag_compliance.py
│
├── test_tools/                     # 4 modules covering all CLI tools
│   ├── test_deployment_readiness.py
│   ├── test_dicom_inspector.py
│   ├── test_sim_job_runner.py
│   └── test_trial_site_monitor.py
│
├── test_physical_robots/           # 6 modules covering robot examples
│   ├── test_sensor_fusion.py
│   ├── test_ros2_deployment.py
│   ├── test_shared_autonomy.py
│   ├── test_surgical_training.py
│   ├── test_surgical_planning.py
│   └── test_treatment_prediction.py
│
├── test_privacy/                   # 4 modules covering privacy framework
│   ├── test_phi_detector.py
│   ├── test_access_control.py
│   ├── test_breach_response.py
│   └── test_dua_generator.py
│
├── test_regulatory/                # 4 modules covering regulatory framework
│   ├── test_fda_submission.py
│   ├── test_irb_protocol.py
│   ├── test_gcp_compliance.py
│   └── test_regulatory_tracker.py
│
├── test_unification/               # 5 modules covering unification framework
│   ├── test_isaac_mujoco_bridge.py
│   ├── test_urdf_converter.py
│   ├── test_unified_agent.py
│   ├── test_framework_detector.py
│   └── test_validation_suite.py
│
├── test_standards/                 # 3 modules covering Q1 2026 standards
│   ├── test_isaac_to_mujoco.py
│   ├── test_benchmark_runner.py
│   └── test_model_validator.py
│
├── test_integration/               # 6 cross-module workflow tests
│   ├── test_dt_to_simulation.py
│   ├── test_agentic_to_regulatory.py
│   ├── test_robot_to_safety.py
│   ├── test_privacy_to_clinical.py
│   ├── test_cross_framework.py
│   └── test_end_to_end_trial.py
│
└── test_regression/                # Regression guards for v0.9.1/v0.9.2 fixes
    ├── test_v092_guards.py
    └── test_v092_comprehensive.py
```

## Testing Philosophy

### 1. Mock-Based Isolation

All external dependencies (NVIDIA Isaac, MuJoCo, ROS 2, DICOM servers, MCP SDK,
torch, langchain, crewai) are either not needed or mocked. Tests exercise pure
Python logic: math, data structures, state machines, and constraint checking.

### 2. Import Strategy

Source modules live in hyphenated directories (`examples-new/`, `digital-twins/`,
`agentic-ai/`, etc.) that cannot be imported with standard Python `import`. The
`conftest.py` provides a `load_module()` helper that uses
`importlib.util.spec_from_file_location()` to load modules by file path.

### 3. Deterministic Results

An `autouse` fixture seeds `np.random.seed(42)` before every test to ensure
reproducible numerical results across platforms and Python versions.

### 4. Clinical Safety Validation

Tests specifically validate:
- Dose calculation boundary conditions (zero alpha/beta, zero volume)
- Safety constraint enforcement (force limits, workspace bounds, velocity)
- Emergency stop logic (stop categories, authorization)
- De-identification completeness (all 18 HIPAA identifiers)
- Regulatory compliance scoring correctness

### 5. Regression Guards

Dedicated regression tests reproduce all 30+ bugs fixed in v0.9.1 and v0.9.2:
- EKF Jacobian sign error
- Inverted hazard ratio
- Division-by-zero in dose calculations
- Infinite loop in policy evaluation
- Audit log returning mutable reference
- Compliance score including NOT_ASSESSED
- And many more

## Coverage Targets

| Category | Modules | Tests | Target |
|----------|---------|-------|--------|
| Digital Twins | 9 | 215+ | Core growth models, treatment sim, clinical interface |
| Agentic AI | 6 | 220+ | MCP server, ReAct planning, safety constraints |
| Physical Robots | 6 | 175+ | Safety monitoring, calibration, sensor fusion |
| CLI Tools | 5 | 110+ | Dose calc, DICOM inspection, deployment readiness |
| Privacy | 4 | 110+ | PHI detection, de-identification, access control |
| Regulatory | 4 | 100+ | FDA, IRB, GCP compliance, regulatory tracking |
| Unification | 5 | 100+ | Bridge, converter, agent interface, validation |
| Standards | 3 | 70+ | Benchmarks, model validation, format conversion |
| Integration | 6 | 65+ | Cross-module workflows, end-to-end trial |
| Regression | 2 | 40+ | v0.9.1/v0.9.2 bug fix guards |

## Adding New Tests

1. Create `tests/test_<category>/test_<module_name>.py`
2. Use `load_module()` from `conftest.py` to import the source module
3. Follow the class-per-feature pattern used in existing tests
4. Add shared fixtures to `conftest.py` if reused across modules
5. Run `ruff format tests/` and `ruff check tests/` before committing
6. Ensure tests pass on Python 3.10, 3.11, and 3.12

## CI Integration

Tests run automatically via GitHub Actions on every push and PR to `main`:
- Python 3.10, 3.11, 3.12 matrix
- `ruff check` and `ruff format --check` for code quality
- `pytest tests/ -v --tb=short` for the full test suite
