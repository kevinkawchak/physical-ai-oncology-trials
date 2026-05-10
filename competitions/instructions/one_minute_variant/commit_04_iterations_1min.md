# Commit 4 (1-Minute Variant): Iteration Design

This file specifies the files the future Claude Code Opus 4.7 1M Max session must author in its fourth commit for the 1-minute variant. The session must author exactly the files listed and must not author additional files in this commit. The parent `competitions/instructions/commit_04_iteration_design.md` lists 10 files for the v3.9.0 1-hour scenario. This 1-minute variant lists 14 files (each per-iteration file kind expands to 16 individual files for a total of 80 per-iteration files plus 4 release-aggregate files, with 5 per-iteration file kinds replacing the parent's single per-iteration Parquet kind).

## Goal

Run 16 deterministic iterations across the 1-minute glioblastoma resection scenario, each with a different seed. Author the iteration design document, the iteration configuration, the Python orchestrator, the Rust high-throughput simulation engine, the per-iteration L1 to L3 pyramid Parquet outputs (script-generated), the per-iteration L0 raw Zenodo pointer, the append-only manifest, the DuckDB analytical store, the exploratory notebook, and the plain-text execution log. Add the Mac M3 Ultra and A100 GPU recipes to the runtime environments inheritance.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1min-trial/docs/iteration_design.md` | Markdown | Hand-authored | 24 KB |
| 2 | `competitions/glioblastoma-1min-trial/config/iterations.yaml` | YAML | Hand-authored | 16 KB |
| 3 | `competitions/glioblastoma-1min-trial/src/simulation/iterate_1min.py` | Python 3.10 | Hand-authored | 18 KB |
| 4 | `competitions/glioblastoma-1min-trial/src/simulation/runner_1min.rs` | Rust 2021 | Hand-authored | 24 KB |
| 5 | `competitions/glioblastoma-1min-trial/src/simulation/Cargo.toml` | TOML | Hand-authored | 1 KB |
| 6 | `competitions/glioblastoma-1min-trial/data/iterations/run_NNNNN_L1_50ms.parquet` (16 files) | Parquet zstd-3 | Script-generated | 480 KB each (7.7 MB total) |
| 7 | `competitions/glioblastoma-1min-trial/data/iterations/run_NNNNN_L2_1s.parquet` (16 files) | Parquet zstd-3 | Script-generated | 24 KB each (384 KB total) |
| 8 | `competitions/glioblastoma-1min-trial/data/iterations/run_NNNNN_L3_phase.parquet` (16 files) | Parquet zstd-3 | Script-generated | under 4 KB each |
| 9 | `competitions/glioblastoma-1min-trial/data/iterations/run_NNNNN_events.parquet` (16 files) | Parquet zstd-3 | Script-generated | 8 KB each |
| 10 | `competitions/glioblastoma-1min-trial/data/iterations/run_NNNNN_L0_raw.zenodo_pointer.json` (16 files) | JSON | Hand-authored skeleton, populated at Commit 5 | 1 KB each |
| 11 | `competitions/glioblastoma-1min-trial/data/iterations/index.jsonl` | JSON Lines | Script-generated | 12 KB |
| 12 | `competitions/glioblastoma-1min-trial/data/iterations/aggregate.duckdb` | DuckDB | Script-generated | 4 MB (under 5 MB Parquet cap) |
| 13 | `competitions/glioblastoma-1min-trial/notebooks/iteration_analysis_1min.ipynb` | Jupyter | Hand-authored, outputs cleared | 30 KB |
| 14 | `competitions/glioblastoma-1min-trial/logs/iteration_run.txt` | Plain text | Script-generated | 40 KB |

The table above lists 14 file kinds. Five of those kinds (orders 6 through 10) are per-iteration file kinds and each expands to 16 individual files (one per iteration), for a total of 80 per-iteration files. The remaining 9 kinds are release-aggregate files. The future session generates the per-iteration files via a single orchestrator invocation. The total committed footprint across all iteration files is 8.2 MB.

## File 1: docs/iteration_design.md

Required sections:

1. Iteration count and rationale: 16 iterations to balance statistical power against the doubled per-iteration committed footprint of the 4-arm topology. 16 is the default iteration count for v3.9.1; later releases may scale to 32 or 64 if Zenodo bandwidth allows. The default tournament size of 4 (defined in `commit_05_competition_1min.md`) is independent of the iteration count.
2. Sweep dimensions (the only parameters that vary between iterations):
   - Seed: integer in [20260510, 20260525] inclusive, one per iteration.
   - Per-arm sensor noise sigma: 0.01 to 0.05 mm linearly across iterations.
   - Per-arm force feedback gain: 0.8 to 1.2 linearly.
   - Inverse kinematics solver tolerance: 1e-6 to 1e-3 logarithmically.
   - Random surgical adverse event injection probability: fixed at 0.05 per iteration.
   - Heartbeat jitter sigma: 0 to 50 microseconds linearly across iterations to test the 3 ms watchdog.
3. Fixed parameters (never vary): patient identity, robot make and model, kinematic limits per arm, safety limits per arm, cumulative force limit (12 N), procedure phases, per-arm tool assignment.
4. Parameter sweep table reproducing the 16 (seed, noise, gain, tol, jitter) tuples.
5. Iteration runtime budget: 30 seconds wall-clock per iteration on the Mac M3 Ultra recipe; 12 seconds wall-clock per iteration on the A100 GPU recipe; 60 seconds wall-clock on the conventional 32-core server profile.
6. Total compute budget: 16 iterations times 30 to 60 seconds equals 8 to 16 minutes serial.
7. Storage budget: 16 iterations times 510 KB committed equals 8.2 MB committed; plus 16 iterations times 26 MB Zenodo equals 416 MB Zenodo. Total committed under 9.7 MB; total Zenodo under 416 MB.
8. Failure handling: a failed iteration writes a record to `index.jsonl` with `status: "failed"` and a stack trace pointer. Failed iterations do not block subsequent iterations.
9. Reproducibility: each iteration's L1 to L3 Parquet files embed the seed in the `meta_seed` column and the iteration ID in the `meta_iteration_id` column.
10. Cross-references to the `runner_1min.rs` engine, the `aggregate.duckdb` analytical store, and the Zenodo pointer schema in `zenodo_archive_protocol.md`.

## File 2: config/iterations.yaml

Required keys:

```
---
iteration_set:
  name: gbm_v3_9_1_default
  count: 16
  description: "Default 16-iteration sweep for v3.9.1 1-minute variant."

base_seed: 20260510

sweeps:
  seed:
    type: linear_int
    start: 20260510
    stop: 20260525
    inclusive: true
  sensor_noise_sigma_mm:
    type: linear
    start: 0.01
    stop: 0.05
    n: 16
  force_feedback_gain:
    type: linear
    start: 0.8
    stop: 1.2
    n: 16
  ik_solver_tolerance:
    type: log
    start: 1.0e-06
    stop: 1.0e-03
    n: 16
  heartbeat_jitter_sigma_us:
    type: linear
    start: 0.0
    stop: 50.0
    n: 16

fixed:
  patient_id: PAT-GBM-0001
  robot_make: Medtronic
  robot_model: NeuroSpeed 1.0
  arms: 4
  procedure_duration_seconds: 60.000
  mixed_tick_us: 1000
  force_tick_us: 100
  ae_probability: 0.05
  cumulative_force_limit_N: 12.0
  per_arm_tip_force_limit_N: 5.0
  estop_latency_limit_ms: 5

paths:
  output_dir: data/iterations
  index_file: data/iterations/index.jsonl
  duckdb_file: data/iterations/aggregate.duckdb
  log_file: logs/iteration_run.txt

execution:
  jobs: 4
  per_iteration_timeout_seconds: 600
  l0_raw_local_cache: false
  l0_raw_upload_to_zenodo: true

zenodo:
  community: physical-ai-oncology-trials
  release_doi_pattern: "10.5281/zenodo.{record_id}"
```

## File 3: src/simulation/iterate_1min.py

Python 3.10 module orchestrating the 16 iterations. Required responsibilities:

- Read `config/iterations.yaml`.
- Materialize the 16 (seed, noise, gain, tol, jitter) tuples deterministically.
- For each tuple, invoke the Rust runner via `subprocess` with the parameter values.
- For each completed iteration, run the L0 to L1 to L2 to L3 plus events aggregation pipeline implemented in `runner_1min.rs`.
- Optionally upload the L0 raw to Zenodo and patch the per-iteration Zenodo pointer JSON.
- Capture per-iteration metrics into the `index.jsonl` manifest.
- Update the `aggregate.duckdb` analytical store after each completed iteration.
- Append a structured plain-text log to `logs/iteration_run.txt`.
- Support parallel execution via `--jobs` flag using `concurrent.futures.ProcessPoolExecutor`.

Required CLI signature using `click`:

```
@click.command()
@click.option("--seed", type=int, default=20260510)
@click.option("--iterations", type=int, default=16)
@click.option("--out", type=click.Path(), default="data/iterations")
@click.option("--jobs", type=int, default=1)
@click.option("--config", type=click.Path(exists=True), default="config/iterations.yaml")
@click.option("--upload-zenodo", is_flag=True, default=False)
def cli(seed: int, iterations: int, out: str, jobs: int, config: str, upload_zenodo: bool) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean.

## File 4: src/simulation/runner_1min.rs

Rust 2021 high-throughput simulation engine for the 4-arm 60-second scenario. Required responsibilities:

- Accept seed, noise, gain, tolerance, jitter, and output path as command-line arguments.
- Generate the 60,000 MIXED records and 540,000 FORCE_ONLY records per arm deterministically using the `rand_pcg` PRNG seeded by the given seed.
- Apply the same per-arm phase-conditioned mapping as `src/mapping/sensor_to_xyz_4arm.py`.
- Apply the cumulative 12 N force enforcement at every 100 microsecond tick.
- Inject random adverse events at the configured probability per iteration.
- Inject heartbeat jitter at the configured sigma.
- Write the per-arm joined sensor plus xyz record set to a single L0 raw Parquet file at the requested output path.
- Aggregate to L1 (20 Hz), L2 (1 Hz), L3 (per phase), and the event log; write each as a separate Parquet zstd-3 file.
- Print structured progress to stderr at 10 percent intervals.

The Rust runner is approximately 100 times faster than the Python reference. The Python reference remains available; the Rust runner is the production sweep engine. Both must produce bit-identical L1 to L3 Parquet files for a fixed seed and parameter tuple.

The future session must include the following dependencies in `Cargo.toml`:

```
[package]
name = "gbm_runner_1min"
version = "3.9.1"
edition = "2021"

[dependencies]
rand_pcg = "0.3"
rand = "0.8"
arrow = "53"
parquet = { version = "53", features = ["zstd"] }
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
crossbeam-channel = "0.5"
```

The future session must run `cargo fmt` and `cargo clippy --all-targets -- -D warnings` and must pass both before committing.

## File 5: src/simulation/Cargo.toml

The TOML file shown in File 4. Required for `cargo build`. The future session must verify `cargo build --release` completes on Linux, MacOS, and Windows.

## Files 6 to 9: Per-iteration L1 to L3 plus events Parquet files

The future session must produce these files by running:

```
python -m src.simulation.iterate_1min --iterations 16 --jobs 4
```

Each iteration produces 4 Parquet files (L1 50 ms, L2 1 s, L3 per phase, events) and 1 Zenodo pointer JSON file. The aggregation rules and per-arm column lists are fixed in `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`.

The future session must verify the parent repository's GitHub commit cap of 10 MB and add a `.gitattributes` entry if needed:

```
data/iterations/*.parquet -filter -diff -merge text
```

The 1-minute variant does not require Git LFS because all committed Parquet files are under 5 MB; the L1 50 ms aggregate at 480 KB across 4 arms per iteration is well inside the cap.

## File 10: data/iterations/run_NNNNN_L0_raw.zenodo_pointer.json

For each of the 16 iterations the future session creates one Zenodo pointer JSON with the following keys:

```
{
  "schema_version": "1.0",
  "release_version": "v3.9.1",
  "iteration_id": "run_NNNNN",
  "scope": "per_iteration",
  "zenodo_doi": "10.5281/zenodo.PLACEHOLDER",
  "zenodo_record_id": "PLACEHOLDER",
  "zenodo_filename": "run_NNNNN_L0_raw_4arm.parquet",
  "sha256": "PLACEHOLDER",
  "byte_size": 26000000,
  "compression": "zstd-3",
  "channel_count_per_arm": 50,
  "arm_count": 4,
  "channel_count_total": 200,
  "mixed_sample_rate_hz": 1000,
  "force_sample_rate_hz": 10000,
  "populated_at_commit": "Commit 5 of v3.9.1 PR"
}
```

The `zenodo_doi`, `zenodo_record_id`, and `sha256` fields are populated at Commit 5 after the Zenodo deposition completes. The pointer files are committed at Commit 4 with the placeholder values.

## File 11: data/iterations/index.jsonl

Append-only manifest. One JSON object per iteration. Required keys per record:

```
{
  "iteration_id": "run_00001",
  "seed": 20260510,
  "sensor_noise_sigma_mm": 0.010,
  "force_feedback_gain": 0.800,
  "ik_solver_tolerance": 1.0e-06,
  "heartbeat_jitter_sigma_us": 0.0,
  "status": "succeeded",
  "wall_clock_seconds": 28.4,
  "l1_path": "data/iterations/run_00001_L1_50ms.parquet",
  "l2_path": "data/iterations/run_00001_L2_1s.parquet",
  "l3_path": "data/iterations/run_00001_L3_phase.parquet",
  "events_path": "data/iterations/run_00001_events.parquet",
  "l0_zenodo_pointer_path": "data/iterations/run_00001_L0_raw.zenodo_pointer.json",
  "l1_sha256": "...",
  "l2_sha256": "...",
  "l3_sha256": "...",
  "events_sha256": "...",
  "ae_count": 0,
  "estop_count": 0,
  "force_violation_count": 0,
  "cumulative_force_violation_count": 0,
  "heartbeat_miss_count": 0,
  "phase_durations_seconds": {
    "phase_1_dural_opening_final": 5.000,
    "phase_2_bulk_resection": 40.000,
    "phase_3_margin_fine_resection": 10.000,
    "phase_4_hemostasis_withdrawal": 5.000
  }
}
```

## File 12: data/iterations/aggregate.duckdb

DuckDB analytical store created by the orchestrator after all iterations complete. Required tables:

- `iteration_index` mirroring `index.jsonl`.
- `l1_per_arm_50ms` aggregating each iteration's L1 Parquet into a single table (1,200 rows per arm per iteration times 4 arms times 16 iterations equals 76,800 rows).
- `l2_per_arm_1s` aggregating each iteration's L2 Parquet (60 rows per arm per iteration times 4 arms times 16 iterations equals 3,840 rows).
- `l3_per_arm_phase` aggregating each iteration's L3 Parquet (4 rows per arm per iteration times 4 arms times 16 iterations equals 256 rows).
- `events` aggregating each iteration's event log Parquet.
- `cumulative_force_violations` listing every cumulative 4-arm force limit violation across all iterations.

Approximate DuckDB file size: 4 MB (under the 5 MB committed Parquet cap; DuckDB files are committed under the same cap).

## File 13: notebooks/iteration_analysis_1min.ipynb

Jupyter notebook with the following cells (outputs cleared before commit):

1. Title and overview markdown cell.
2. DuckDB connection cell.
3. Per-iteration cumulative force violation count bar chart.
4. Per-iteration AE count bar chart.
5. Per-arm 1 second mean ee position overlay across all 16 iterations.
6. Per-iteration phase duration heatmap.
7. Per-iteration heartbeat miss count bar chart.
8. Markdown cell summarizing observations.

## File 14: logs/iteration_run.txt

Plain-text execution log. One event per line, ISO 8601 timestamp prefix. The orchestrator appends to this file across all 16 iterations.

## Mac M3 Ultra Runtime Recipe (added by 1-minute variant)

The 1-minute variant extends the parent runtime environments with a Mac M3 Ultra recipe optimized for the 4-arm Rust runner:

```
# 1. Homebrew packages (in addition to the parent recipe)
brew install rust llvm
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 2. Build the Rust runner with native CPU optimizations
RUSTFLAGS="-C target-cpu=apple-m3" cargo build --release --manifest-path src/simulation/Cargo.toml

# 3. Run a single iteration smoke test (under 30 s wall-clock on M3 Ultra)
python -m src.simulation.iterate_1min --seed 20260510 --iterations 1 --out data/iterations
```

Expected single-iteration wall-clock: 30 seconds. Expected 16-iteration wall-clock with `--jobs 4`: under 5 minutes.

## NVIDIA A100 GPU Runtime Recipe (added by 1-minute variant)

The 4-arm 10 kHz force physics simulator benefits from GPU acceleration on the per-arm finite-element tissue model. The future session adds an optional CUDA path in `runner_1min.rs` gated by the `cargo build --release --features cuda` flag:

```
# 1. CUDA toolkit
sudo apt-get install -y nvidia-cuda-toolkit
nvcc --version

# 2. Build the Rust runner with CUDA features
cargo build --release --features cuda --manifest-path src/simulation/Cargo.toml

# 3. Run a single iteration on A100
python -m src.simulation.iterate_1min --seed 20260510 --iterations 1 --jobs 1
```

Expected single-iteration wall-clock on A100: 12 seconds. Expected 16-iteration wall-clock: under 4 minutes serial or under 1 minute with `--jobs 4` on a 4-A100 host.

## Validation After Commit 4

- `python -m src.simulation.iterate_1min --iterations 16 --jobs 4` completes in under 10 minutes on the conventional server profile.
- All 80 Parquet files (16 iterations times 5 file kinds) exist and pass schema validation.
- Each L0 raw Zenodo pointer JSON exists with placeholder values.
- `data/iterations/index.jsonl` contains exactly 16 lines.
- `data/iterations/aggregate.duckdb` contains the six required tables with the expected row counts.
- The notebook executes top-to-bottom without error.
- All committed files are under 10 MB and all committed Parquet files are under 5 MB.
- `ruff format --check .` passes.
- `ruff check .` passes.
- `yamllint -d relaxed competitions/glioblastoma-1min-trial/config/` passes.

## Source Files Cited

- `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`. Source for the per-iteration L1 to L3 plus events file kinds and the 510 KB per-iteration committed total.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the per-arm tick counts and the 200-channel total.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the cumulative 12 N force enforcement that the simulator must execute at every 100 microsecond tick.
- `competitions/instructions/one_minute_variant/zenodo_archive_protocol.md`. Source for the per-iteration L0 Zenodo pointer schema used by File 10.
- `competitions/instructions/competition_protocol.md`. Source for the per-release snapshot pattern that `releases/v3.9.1/` mirrors.
- `competitions/instructions/chunking_strategy.md`. Source for the per-commit budget and the script-then-Parquet pattern.
- `competitions/instructions/file_format_conventions.md`. Source for the iteration filename pattern (`run_<id>_<level>.parquet` with five-digit zero padding) and the Parquet column conventions.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules that File 3 and File 2 must satisfy.
- `competitions/instructions/runtime_environments.md`. Source for the parent runtime recipes that this commit extends with the Mac M3 Ultra and A100 GPU recipes.
- `competitions/instructions/commit_04_iteration_design.md`. Source for the parent v3.9.0 10-file Commit 4 structure that this 1-minute variant extends to 14 files.
