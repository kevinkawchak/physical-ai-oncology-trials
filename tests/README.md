# Test Suite — Physical AI Oncology Trials

## Overview

Comprehensive pytest-based test suite covering safety-critical modules,
dose calculations, digital twin synchronization, agentic AI servers,
spatial calibration, specimen handling, privacy de-identification, and
regression guards for bugs fixed in v0.9.1/v0.9.2.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a single test module
pytest tests/test_dose_calculator.py -v

# Run tests matching a pattern
pytest tests/ -k "safety" -v
```

## Requirements

- Python 3.10+
- numpy, scipy (already in project requirements)
- pytest

```bash
pip install pytest
```

## Test Modules

| Module | Source Under Test | Tests |
|--------|-------------------|-------|
| `test_safety_monitoring.py` | `examples-new/01_realtime_safety_monitoring.py` | SafetyMonitor, ForceTorqueSensorProcessor, WorkspaceBoundaryGenerator |
| `test_dose_calculator.py` | `tools/dose-calculator/dose_calculator.py` | BED, EQD2, TCP, NTCP, scheme parsing, tissue data |
| `test_digital_twin_sync.py` | `digital-twins/examples-twins/01_realtime_dt_synchronization.py` | EKF, particle filter, anomaly detection, synchronizer |
| `test_mcp_server.py` | `agentic-ai/examples-agentic-ai/01_mcp_clinical_robotics_server.py` | Tool/resource handlers, audit trail, data models |
| `test_calibration.py` | `examples-new/04_hand_eye_calibration_registration.py` | Transform math, Tsai-Lenz calibration, Arun SVD registration |
| `test_sample_handling.py` | `examples-new/06_robotic_sample_handling.py` | Specimen model, barcode verification, cold chain, batch processing |
| `test_deidentification.py` | `privacy/de-identification/deidentification_pipeline.py` | Safe Harbor transforms, PHI detection, config, results |
| `test_regression.py` | Multiple modules | v0.9.2 critical bugs: EKF Jacobian, hazard ratio, division-by-zero, DoseResult truthiness |

## Architecture

### Import Strategy

Source modules live in hyphenated directories (`examples-new/`, `digital-twins/`, etc.)
that cannot be imported with standard Python `import`. The `conftest.py` provides
a `load_module()` helper that uses `importlib.util.spec_from_file_location()` to
load modules by file path.

### Fixtures

Shared fixtures in `conftest.py` provide:

- **Deterministic RNG**: `autouse` fixture seeds NumPy to 42
- **Module loaders**: `safety_mod`, `dose_calc_mod`, `dt_sync_mod`, `mcp_mod`, `calibration_mod`, `deid_mod`
- **Test data**: `nominal_robot_state`, `default_safety_limits`, `zero_ft_reading`, `baseline_patient_state`, `standard_fractionation`, `sample_patient_record`

### No External Dependencies

All hardware and cloud dependencies (NVIDIA Isaac, MuJoCo, ROS 2, DICOM servers,
MCP SDK) are either optional imports in the source or not needed for the tested
code paths. Tests run without GPU, robot hardware, or network access.

## Adding New Tests

1. Create `tests/test_<module_name>.py`
2. Use `load_module()` from `conftest.py` to import the source
3. Follow the class-per-feature pattern used in existing tests
4. Run `ruff format tests/` and `ruff check tests/` before committing
