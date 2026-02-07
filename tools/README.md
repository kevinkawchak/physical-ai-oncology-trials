# Tools: Command-Line Helpers for Physical AI Oncology Trials

Command-line utilities for engineers working on physical AI systems in oncology clinical trials. Each tool operates as a standalone CLI script with no required external service dependencies.

## Directory Structure

```
tools/
├── README.md
├── dicom-inspector/
│   └── dicom_inspector.py          # DICOM file inspection and trial-readiness validation
├── dose-calculator/
│   └── dose_calculator.py          # Radiotherapy dose calculations (BED, EQD2, TCP, NTCP)
├── trial-site-monitor/
│   └── trial_site_monitor.py       # Multi-site enrollment tracking and data quality monitoring
├── sim-job-runner/
│   └── sim_job_runner.py           # Cross-framework simulation job launcher and comparator
└── deployment-readiness/
    └── deployment_readiness.py     # Pre-deployment AI model validation for clinical use
```

## Tool Summaries

### 1. DICOM Inspector (`dicom-inspector/dicom_inspector.py`)

Inspect, validate, and audit DICOM files for oncology trial data management. Checks de-identification status, validates required DICOM tags for trial compliance, and generates batch reports across imaging directories.

```bash
# Inspect a single DICOM file
python tools/dicom-inspector/dicom_inspector.py inspect /path/to/file.dcm

# Validate de-identification status of a directory
python tools/dicom-inspector/dicom_inspector.py audit-phi /path/to/dicom_dir/

# Check DICOM compliance for trial submission
python tools/dicom-inspector/dicom_inspector.py validate /path/to/dicom_dir/ --standard DICOM-RT

# Generate summary report across a study
python tools/dicom-inspector/dicom_inspector.py summarize /path/to/study_dir/ --output report.json
```

**Use cases**: Pre-submission data auditing, PHI leak detection in imaging pipelines, modality-specific tag validation, batch QA for multi-site imaging data.

### 2. Dose Calculator (`dose-calculator/dose_calculator.py`)

Radiotherapy dose calculations from the command line. Supports BED, EQD2, TCP (Poisson/logistic), and NTCP (Lyman-Kutcher-Burman) models with standard tissue parameters.

```bash
# Calculate BED for a fractionation scheme
python tools/dose-calculator/dose_calculator.py bed --dose 60 --fractions 30 --alpha-beta 10

# Compare fractionation schemes (conventional vs. hypofractionated)
python tools/dose-calculator/dose_calculator.py compare --schemes "60/30,42.56/16,34/10" --alpha-beta 10

# Calculate TCP for a tumor
python tools/dose-calculator/dose_calculator.py tcp --dose 60 --fractions 30 --model poisson --n0 1e9 --alpha 0.3

# Calculate NTCP using LKB model
python tools/dose-calculator/dose_calculator.py ntcp --dose 60 --fractions 30 --td50 50 --m 0.18 --n 0.12
```

**Use cases**: Quick dose-equivalence checks during protocol design, fractionation scheme comparison for trial arms, TCP/NTCP estimation for treatment planning QA.

### 3. Trial Site Monitor (`trial-site-monitor/trial_site_monitor.py`)

Monitor enrollment, data quality, and protocol adherence across multi-site oncology trials. Reads from a site data JSON manifest and flags sites needing intervention.

```bash
# Generate site enrollment dashboard
python tools/trial-site-monitor/trial_site_monitor.py enrollment /path/to/trial_manifest.json

# Run data quality checks
python tools/trial-site-monitor/trial_site_monitor.py quality /path/to/trial_manifest.json

# Detect protocol deviations
python tools/trial-site-monitor/trial_site_monitor.py deviations /path/to/trial_manifest.json

# Full site performance report
python tools/trial-site-monitor/trial_site_monitor.py report /path/to/trial_manifest.json --output site_report.json

# Generate a sample trial manifest template
python tools/trial-site-monitor/trial_site_monitor.py init-manifest --sites 5 --output trial_manifest.json
```

**Use cases**: CRO-level enrollment tracking, automated data quality scoring, early detection of underperforming sites, regulatory audit preparation.

### 4. Simulation Job Runner (`sim-job-runner/sim_job_runner.py`)

Launch, manage, and compare simulation jobs across Isaac Lab, MuJoCo, PyBullet, and Gazebo from a single CLI. Uses a YAML job specification to define parameters and target frameworks.

```bash
# Launch a simulation job on MuJoCo
python tools/sim-job-runner/sim_job_runner.py launch --framework mujoco --task needle_insertion --config configs/training_config.yaml

# Launch the same job across all available frameworks for comparison
python tools/sim-job-runner/sim_job_runner.py launch-all --task needle_insertion --config configs/training_config.yaml

# Compare results across frameworks
python tools/sim-job-runner/sim_job_runner.py compare --results-dir results/needle_insertion/

# List available task definitions
python tools/sim-job-runner/sim_job_runner.py list-tasks

# Generate a job configuration template
python tools/sim-job-runner/sim_job_runner.py init-config --task needle_insertion --output job_config.yaml
```

**Use cases**: Cross-framework policy validation, batch simulation runs for parameter sweeps, reproducible sim-to-real pipeline setup, framework performance benchmarking.

### 5. Deployment Readiness (`deployment-readiness/deployment_readiness.py`)

Pre-deployment validation for AI models entering clinical oncology workflows. Checks ONNX compatibility, measures inference latency, verifies safety constraints, and generates a regulatory readiness checklist.

```bash
# Run full deployment readiness check
python tools/deployment-readiness/deployment_readiness.py check --model model.onnx --config deployment_config.yaml

# Benchmark inference latency
python tools/deployment-readiness/deployment_readiness.py benchmark --model model.onnx --iterations 1000 --device cpu

# Verify safety constraints (force limits, workspace bounds)
python tools/deployment-readiness/deployment_readiness.py safety --model model.onnx --constraints safety_constraints.yaml

# Generate regulatory checklist (IEC 62304, FDA AI/ML)
python tools/deployment-readiness/deployment_readiness.py checklist --model model.onnx --output readiness_report.json

# Validate model against reference outputs
python tools/deployment-readiness/deployment_readiness.py validate --model model.onnx --reference reference_outputs.npz --tolerance 1e-4
```

**Use cases**: Pre-submission model qualification, go/no-go gate for clinical deployment, regulatory documentation generation, continuous model monitoring baseline.

## Dependencies

These tools use dependencies already listed in the project `requirements.txt`. Key packages:

| Tool | Primary Dependencies |
|------|---------------------|
| DICOM Inspector | `pydicom` |
| Dose Calculator | `numpy`, `scipy` |
| Trial Site Monitor | `numpy` (standard library otherwise) |
| Sim Job Runner | `pyyaml`, framework-specific SDKs (optional) |
| Deployment Readiness | `onnx`, `onnxruntime`, `numpy` |

## Design Principles

1. **Standalone operation** - Each tool works independently with no inter-tool dependencies
2. **No PHI in outputs** - Tools that handle patient data never log or print identifiable information
3. **JSON output** - All tools support `--output` for machine-readable JSON reports
4. **Offline-first** - No network calls required; all computation is local
5. **Exit codes** - Standard exit codes (0 = success, 1 = failure, 2 = warnings) for CI/CD integration

## Relationship to Other Directories

| Directory | Focus | How `tools/` Differs |
|-----------|-------|---------------------|
| `examples/` | AI/ML pipeline demonstrations | `tools/` provides reusable CLI utilities, not demos |
| `examples-new/` | Physical robot hardware examples | `tools/` targets data management and pre-deployment workflows |
| `scripts/` | Installation verification | `tools/` covers operational workflows beyond setup |
| `unification/` | Cross-platform framework bridging | `tools/` consumes unification outputs; does not duplicate bridging |
| `privacy/` | PHI detection and de-identification | `tools/dicom-inspector` checks DICOM-specific PHI; does not replace `privacy/` |
| `regulatory/` | FDA/IRB/ICH-GCP compliance tracking | `tools/deployment-readiness` generates checklists; does not replace `regulatory/` |

*Note: All tools contain illustrative parameters. Validate against your institution's protocols before clinical use.*
