# Commit 4: Iteration Design

This file specifies the nine files the future Claude Code Opus 4.7 1M Max session must author in its fourth commit. The session must author exactly the files listed and must not author additional files in this commit.

## Goal

Run 64 deterministic iterations across the 1-hour glioblastoma resection scenario, each with a different seed. Author the iteration design document, the iteration configuration, the Python orchestrator, the Rust high-throughput simulation engine, the per-iteration Parquet outputs (script-generated), the append-only manifest, the DuckDB analytical store, the exploratory notebook, and the plain-text execution log.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1hr-trial/docs/iteration_design.md` | Markdown | Hand-authored | 22 KB |
| 2 | `competitions/glioblastoma-1hr-trial/config/iterations.yaml` | YAML | Hand-authored | 18 KB |
| 3 | `competitions/glioblastoma-1hr-trial/src/simulation/iterate.py` | Python 3.10 | Hand-authored | 16 KB |
| 4 | `competitions/glioblastoma-1hr-trial/src/simulation/runner.rs` | Rust 2021 | Hand-authored | 20 KB |
| 5 | `competitions/glioblastoma-1hr-trial/src/simulation/Cargo.toml` | TOML | Hand-authored | 1 KB |
| 6 | `competitions/glioblastoma-1hr-trial/data/iterations/run_00001.parquet` through `run_00064.parquet` | Parquet | Script-generated | 90 MB each (5.7 GB total) |
| 7 | `competitions/glioblastoma-1hr-trial/data/iterations/index.jsonl` | JSON Lines | Script-generated | 32 KB |
| 8 | `competitions/glioblastoma-1hr-trial/data/iterations/aggregate.duckdb` | DuckDB | Script-generated | 200 MB |
| 9 | `competitions/glioblastoma-1hr-trial/notebooks/iteration_analysis.ipynb` | Jupyter | Hand-authored, outputs cleared | 30 KB |
| 10 | `competitions/glioblastoma-1hr-trial/logs/iteration_run.txt` | Plain text | Script-generated | 80 KB |

The original instruction list named 9 files. Splitting `Cargo.toml` from `runner.rs` adds one file because `cargo build` requires both. The future session must include both. Per-iteration Parquet files are listed as a single row even though they are 64 individual files; the future session generates them with a single orchestrator invocation.

## File 1: docs/iteration_design.md

Required sections:

1. Iteration count and rationale: 64 iterations to balance statistical power against compute time. 64 is the default tournament size for v3.9.0; later releases may scale to 128, 256, or 1,024.
2. Sweep dimensions (the only parameters that vary between iterations):
   - Seed: integer in [20260509, 20260572] inclusive, one per iteration.
   - Sensor noise sigma: 0.01 to 0.05 mm linearly across iterations.
   - Force feedback gain: 0.8 to 1.2 linearly.
   - Inverse kinematics solver tolerance: 1e-6 to 1e-3 logarithmically.
   - Random surgical adverse event injection probability: fixed at 0.05 per iteration.
3. Fixed parameters (never vary): patient identity, robot make and model, kinematic limits, safety limits, force limits, procedure phases.
4. Parameter sweep table reproducing the 64 (seed, noise, gain, tol) tuples.
5. Iteration runtime budget: 90 seconds wall-clock per iteration on the conventional high-end server profile from `competitions/instructions/runtime_environments.md`.
6. Total compute budget: 64 iterations times 90 seconds equals 96 minutes serial, or about 6 minutes with `--jobs 16`.
7. Storage budget: 64 iterations times 90 MB Parquet equals 5.7 GB. The future session must verify the parent repository's Git LFS quota before committing.
8. Failure handling: a failed iteration writes a record to `index.jsonl` with `status: "failed"` and a stack trace pointer. Failed iterations do not block subsequent iterations.
9. Reproducibility: each iteration's Parquet file embeds the seed in the `meta_seed` column and the iteration ID in the `meta_iteration_id` column.
10. Cross-references to the `runner.rs` engine and the `aggregate.duckdb` analytical store.

## File 2: config/iterations.yaml

Required keys:

```
---
iteration_set:
  name: gbm_v3_9_0_default
  count: 64
  description: "Default 64-iteration sweep for v3.9.0 release."

base_seed: 20260509

sweeps:
  seed:
    type: linear_int
    start: 20260509
    stop: 20260572
    inclusive: true
  sensor_noise_sigma_mm:
    type: linear
    start: 0.01
    stop: 0.05
    n: 64
  force_feedback_gain:
    type: linear
    start: 0.8
    stop: 1.2
    n: 64
  ik_solver_tolerance:
    type: log
    start: 1.0e-06
    stop: 1.0e-03
    n: 64

fixed:
  patient_id: PAT-GBM-0001
  robot_make: Medtronic
  robot_model: ROSA ONE Brain
  procedure_duration_seconds: 3600.000
  tick_seconds: 0.001
  ae_probability: 0.05

paths:
  output_dir: data/iterations
  index_file: data/iterations/index.jsonl
  duckdb_file: data/iterations/aggregate.duckdb
  log_file: logs/iteration_run.txt

execution:
  jobs: 16
  per_iteration_timeout_seconds: 600
```

The future session must list all 64 sweep tuples explicitly in the `iteration_design.md` document (File 1) but must keep `iterations.yaml` (File 2) compact by using the linear and log sweep specifications above. The orchestrator (File 3) materializes the 64 tuples deterministically from the YAML.

## File 3: src/simulation/iterate.py

Python 3.10 module orchestrating the 64 iterations. Required responsibilities:

- Read `config/iterations.yaml`.
- Materialize the 64 (seed, noise, gain, tol) tuples deterministically.
- For each tuple, invoke the Rust runner via `subprocess` with the parameter values.
- Capture per-iteration metrics into the `index.jsonl` manifest.
- Update the `aggregate.duckdb` analytical store after each completed iteration.
- Append a structured plain-text log to `logs/iteration_run.txt`.
- Support parallel execution via `--jobs` flag using `concurrent.futures.ProcessPoolExecutor`.

Required CLI signature using `click`:

```
@click.command()
@click.option("--seed", type=int, default=20260509)
@click.option("--iterations", type=int, default=64)
@click.option("--out", type=click.Path(), default="data/iterations")
@click.option("--jobs", type=int, default=1)
@click.option("--config", type=click.Path(exists=True), default="config/iterations.yaml")
def cli(seed: int, iterations: int, out: str, jobs: int, config: str) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean.

## File 4: src/simulation/runner.rs

Rust 2021 high-throughput simulation engine. Required responsibilities:

- Accept seed, noise, gain, tolerance, and output path as command-line arguments.
- Generate the 3,600,000 sensor records deterministically using the `rand_pcg` PRNG seeded by the given seed.
- Apply the same phase-conditioned mapping as `src/mapping/sensor_to_xyz.py` to produce the xyz command stream.
- Write the joined sensor + xyz record set to a single Parquet file at the requested output path.
- Inject random adverse events at the configured probability per iteration.
- Print structured progress to stderr at 10-percent intervals.

The Rust runner is approximately 100 times faster than the Python mapper. The Python mapper remains the reference implementation; the Rust runner is the production sweep engine. Both must produce bit-identical Parquet files for a fixed seed and parameter tuple.

The future session must include the following dependencies in `Cargo.toml`:

```
[package]
name = "gbm_runner"
version = "3.9.0"
edition = "2021"

[dependencies]
rand_pcg = "0.3"
rand = "0.8"
arrow = "53"
parquet = "53"
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
```

The future session must run `cargo fmt` and `cargo clippy --all-targets -- -D warnings` and must pass both before committing.

## File 5: src/simulation/Cargo.toml

The TOML file shown in File 4. Required for `cargo build`. The future session must verify `cargo build --release` completes on Linux, MacOS, and Windows.

## File 6: data/iterations/run_00001.parquet through run_00064.parquet

The future session must produce these files by running:

```
python -m src.simulation.iterate --iterations 64 --jobs 16
```

Each file contains the joined sensor + xyz record set for one iteration: approximately 3.6 million rows with 50 sensor columns plus 14 command columns plus the two metadata columns. Snappy compression. Approximate on-disk size: 90 MB per file, 5.7 GB total.

The future session must verify the parent repository's Git LFS quota and add a `.gitattributes` entry if needed:

```
data/iterations/*.parquet filter=lfs diff=lfs merge=lfs -text
```

If LFS is not available, the future session commits only `run_00001.parquet` (one file at 90 MB) and leaves the remaining 63 files in `.gitignore` with a regeneration recipe documented in the README. The future Commit 5 then runs against the regenerated set.

## File 7: data/iterations/index.jsonl

Append-only manifest. One JSON object per iteration. Required keys per record:

```
{
  "iteration_id": "run_00001",
  "seed": 20260509,
  "sensor_noise_sigma_mm": 0.010,
  "force_feedback_gain": 0.800,
  "ik_solver_tolerance": 1.0e-06,
  "status": "succeeded",
  "wall_clock_seconds": 87.4,
  "parquet_path": "data/iterations/run_00001.parquet",
  "parquet_sha256": "...",
  "ae_count": 0,
  "estop_count": 0,
  "force_violation_count": 0,
  "phase_durations_seconds": {
    "setup": 600.000,
    "dural_opening": 300.000,
    "tumor_resection_coarse": 1500.000,
    "tumor_resection_fine": 900.000,
    "hemostasis_and_closure_prep": 300.000
  }
}
```

## File 8: data/iterations/aggregate.duckdb

DuckDB analytical store created by the orchestrator after all iterations complete. Required tables:

- `iteration_index` mirroring `index.jsonl`.
- `sensor_per_second_mean` aggregating each iteration's sensor channels to per-second means (3,600 rows per iteration times 64 iterations equals 230,400 rows).
- `xyz_per_second_mean` aggregating each iteration's xyz commands to per-second means.
- `force_violations` listing every force limit violation across all iterations.

The future session must include a verification query in the README that returns the row count of each table.

## File 9: notebooks/iteration_analysis.ipynb

Jupyter notebook with the following cells (outputs cleared before commit):

1. Title and overview markdown cell.
2. DuckDB connection cell.
3. Per-iteration force violation count bar chart.
4. Per-iteration ae count bar chart.
5. Per-second mean xyz path overlay across all 64 iterations.
6. Per-iteration phase duration heatmap.
7. Markdown cell summarizing observations.

The notebook must be saved with cell outputs cleared so that the file size stays under 30 KB.

## File 10: logs/iteration_run.txt

Plain-text execution log. One event per line, ISO 8601 timestamp prefix. The orchestrator appends to this file across all 64 iterations. Required event types:

```
2026-05-09T12:34:56.789Z INFO  start iteration_id=run_00001 seed=20260509
2026-05-09T12:36:23.123Z INFO  end   iteration_id=run_00001 status=succeeded wall=87.4
2026-05-09T12:36:23.234Z WARN  force_violation iteration_id=run_00001 tick_ms=1234567 force_N=12.4
```

## Validation After Commit 4

- `python -m src.simulation.iterate --iterations 64 --jobs 16` completes in under 10 minutes.
- All 64 Parquet files exist and pass schema validation.
- `data/iterations/index.jsonl` contains exactly 64 lines.
- `data/iterations/aggregate.duckdb` contains the four required tables with the expected row counts.
- The notebook executes top-to-bottom without error in a fresh Python 3.10 venv with the project's dev extras.
- `ruff format --check .` passes.
- `ruff check .` passes.
- `yamllint -d relaxed competitions/glioblastoma-1hr-trial/config/` passes.

## Source Files Cited

- `competitions/instructions/competition_protocol.md`. Source for the 64-iteration default and the seed-snapshot pattern.
- `competitions/instructions/chunking_strategy.md`. Source for the per-commit budget and the script-then-Parquet pattern.
- `competitions/instructions/file_format_conventions.md`. Source for the iteration filename pattern (`run_<id>.parquet` with five-digit zero padding) and the Parquet column conventions.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules that File 3 and File 2 must satisfy.
- `competitions/instructions/runtime_environments.md`. Source for the conventional high-end server profile and the per-iteration runtime budget.
- `sponsor/final_paper/168_hours/`. Source for the existing repository pattern of running 168 hourly Python scripts in sequence; the 64-iteration sweep adapts that pattern to a single 1-hour scenario with parameter variation rather than time variation.
- `new-trial/national-24-7-trial/extra-hours/`. Source for the precedent of generating overflow simulation hours that exceed primary cloud token budgets.
