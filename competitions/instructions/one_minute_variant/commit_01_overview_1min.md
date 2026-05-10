# Commit 1 (1-Minute Variant): Project Overview

This file specifies the files the future Claude Code Opus 4.7 1M Max session must author in its first commit for the 1-minute variant. The session must author exactly the files listed and must not author additional files in this commit. The parent `competitions/instructions/commit_01_project_overview.md` lists 7 files for the v3.9.0 1-hour scenario. This 1-minute variant lists 8 files because the 4-arm topology adds the multi-arm coordination overview document.

## Goal

Establish the project skeleton for the 1-minute variant: top-level README, system architecture (4-arm), dependency manifest, multi-service stack, global configuration, license, ASCII facility view, and the multi-arm coordination overview. The skeleton must allow a fresh clone to install dependencies and run a smoke test even though no Python or schema content exists yet.

## Files to Author

| Order | Path | Format | Approximate size |
|-------|------|--------|-------------------|
| 1 | `competitions/glioblastoma-1min-trial/README.md` | Markdown | 26 KB |
| 2 | `competitions/glioblastoma-1min-trial/docs/architecture.md` | Markdown with Mermaid | 22 KB |
| 3 | `competitions/glioblastoma-1min-trial/docs/multi_arm_coordination.md` | Markdown | 14 KB |
| 4 | `competitions/glioblastoma-1min-trial/pyproject.toml` | TOML | 4 KB |
| 5 | `competitions/glioblastoma-1min-trial/docker-compose.yml` | YAML | 4 KB |
| 6 | `competitions/glioblastoma-1min-trial/config/project.yaml` | YAML | 8 KB |
| 7 | `competitions/glioblastoma-1min-trial/LICENSE.txt` | Text | 1 KB |
| 8 | `competitions/glioblastoma-1min-trial/docs/architecture_overview_4arm.txt` | ASCII text | 7 KB |

## File 1: README.md

Sections required, in order:

1. Title block with v3.9.1 release badge, DOI badge, resolution badge, license badge, Python version badge, 1-minute variant badge.
2. Project narrative: the on-premises LLM thesis, the patient (PAT-GBM-0001), the procedure (60 seconds with pre-op precomputed), the robot (Medtronic NeuroSpeed 1.0 hypothetical 2030 with 4 cooperating arms), the resolution (mixed 1 kHz commands plus 10 kHz force), the duration (60 seconds).
3. Quick Start: the runtime recipes from `competitions/instructions/runtime_environments.md` plus the new Mac M3 Ultra and A100 GPU recipes added by the 1-minute variant in `commit_04_iterations_1min.md`.
4. Repository tree of `competitions/glioblastoma-1min-trial/` showing all files that will exist after Commit 5.
5. Per-commit roadmap: the seven commits described in one paragraph each.
6. Verification block from `commit_04_iterations_1min.md`.
7. Citation block: the v3.9.1 DOI plus a BibTeX snippet plus the Zenodo L0 archive DOI.
8. License pointer to `LICENSE.txt`.

The README must include single dashes only and black text only. The README must not include any em dashes, double dashes, or triple dashes.

## File 2: docs/architecture.md

Sections required, in order:

1. Architecture narrative covering the 4-arm topology and the on-prem LLM control loop.
2. Mermaid diagram extending the parent v3.9.0 `flowchart LR` template with 4 arm subgraphs and the 1 kHz heartbeat broadcast bus.
3. 4-phase 60-second procedure timeline table reproduced from `competitions/instructions/one_minute_variant/glioblastoma_context_1min.md`.
4. Per-arm sensor channel summary table reproduced from `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`.
5. Per-arm tool assignment table.
6. 4-arm coordination ASCII diagram reproduced verbatim from `multi_arm_coordination.md`.
7. Pyramid level table from `file_size_pyramid_1min.md`.
8. Pointer to `architecture_overview_4arm.txt` for the full ASCII facility view.

## File 3: docs/multi_arm_coordination.md

The future session must embed a verbatim copy of `competitions/instructions/one_minute_variant/multi_arm_coordination.md` into the output tree at `docs/multi_arm_coordination.md`. The embedded copy lets the simulation be self-standing once the future session ships it.

## File 4: pyproject.toml

Required content:

```
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "glioblastoma-1min-trial"
version = "3.9.1"
description = "On-prem LLM controlled 1-minute glioblastoma resection simulation with 4 cooperating arms at mixed 1 kHz / 10 kHz force resolution."
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE.txt"}
authors = [{name = "Kevin Kawchak"}]
dependencies = [
  "numpy>=1.24",
  "pandas>=2.0",
  "pyarrow>=14",
  "duckdb>=0.9",
  "pydantic>=2.5",
  "pyyaml>=6",
  "click>=8.1",
  "anthropic>=0.40",
  "matplotlib>=3.8",
  "plotly>=5.18",
  "zstandard>=0.22",
  "requests>=2.31",
]

[project.optional-dependencies]
dev = ["ruff>=0.5", "yamllint>=1.33", "pytest>=7.4"]
llm-local = ["ollama>=0.3"]
zenodo = ["zenodo-client>=0.3"]

[tool.setuptools.packages.find]
where = ["src"]
```

The `pyproject.toml` adds two dependencies relative to the parent v3.9.0: `zstandard` for the Parquet zstd-3 default override and `requests` for Zenodo deposition uploads. The `zenodo` optional extra wraps the Zenodo client used by the future Commit 5 to populate the L0 raw pointer files.

## File 5: docker-compose.yml

Required services:

- `llm`: on-prem language model service, default Ollama image with Anthropic API fallback.
- `ingest`: per-arm sensor stream consumer (4 instances multiplexed onto 1 process).
- `simulator`: Rust runner for the 4-arm physics simulation.
- `db`: DuckDB sidecar mounted on `data/iterations/` for the L1 to L3 aggregates.
- `zenodo`: optional one-shot service that uploads the L0 raw to Zenodo and patches the pointer files.

The `docker-compose.yml` must use named volumes for `data/`, `logs/`, and `releases/v3.9.1/`. The compose file must be `yamllint -d relaxed` clean.

## File 6: config/project.yaml

Required keys:

```
---
project:
  name: glioblastoma-1min-trial
  version: "3.9.1"
  release_date: "2026-05-10"
  variant: one_minute

units:
  time: microsecond
  position: millimeter
  rotation: radian
  force: newton
  torque: newton_meter

coordinate_frame:
  origin: mayfield_clamp_pin_midpoint
  x_positive: patient_left
  y_positive: patient_anterior
  z_positive: patient_superior
  quaternion_convention: scalar_first

trial:
  patient_id: PAT-GBM-0001
  duration_seconds: 60.000
  mixed_tick_us: 1000
  force_tick_us: 100
  total_mixed_ticks: 60000
  total_force_ticks: 600000

paths:
  data: data
  config: config
  logs: logs
  results: results
  releases: releases

robot:
  make: Medtronic
  model: NeuroSpeed 1.0
  hardware_revision: v1.0
  firmware: "1.0.0"
  arms: 4
  dof_per_arm: 7
  mixed_sample_rate_hz: 1000
  force_sample_rate_hz: 10000

phases:
  phase_1_dural_opening_final:
    start_s: 0.000
    end_s: 5.000
  phase_2_bulk_resection:
    start_s: 5.000
    end_s: 45.000
  phase_3_margin_fine_resection:
    start_s: 45.000
    end_s: 55.000
  phase_4_hemostasis_withdrawal:
    start_s: 55.000
    end_s: 60.000

zenodo:
  enabled: true
  community: physical-ai-oncology-trials
  l0_raw_size_per_iteration_mb: 26
  l0_raw_total_mb: 416
```

## File 7: LICENSE.txt

Verbatim MIT License text matching the parent repository's `LICENSE` file. The file is named `LICENSE.txt` to match the parent v3.9.0 convention. The future session must use the same MIT text used by the parent repository.

## File 8: docs/architecture_overview_4arm.txt

Single ASCII page using the 4-arm coordination template extension. Shows the operating suite snapshot with all 4 arms, the robot identity, the patient identity, and the per-arm sensor stream destinations. 60 lines maximum, 80 columns maximum. Embeds a small replica of the heartbeat diagram from `multi_arm_coordination.md`.

## Validation After Commit 1

The future session must verify the following after Commit 1 lands:

- `pip install -e .` succeeds inside a fresh Python 3.10 venv.
- `ruff format --check .` passes.
- `ruff check .` passes.
- `yamllint -d relaxed competitions/glioblastoma-1min-trial/config/` passes.
- `docker compose -f competitions/glioblastoma-1min-trial/docker-compose.yml config` parses without error.
- File 8 has 60 lines or fewer and 80 columns or fewer.

## Source Files Cited

- `competitions/instructions/one_minute_variant/README.md`. Source for the 1-minute variant inheritance map and the 12-file instruction set list.
- `competitions/instructions/one_minute_variant/glioblastoma_context_1min.md`. Source for the 4-phase 60-second timeline.
- `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md`. Source for the NeuroSpeed 1.0 4-arm 7-DOF specification.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the 1 kHz heartbeat protocol and the verbatim copy embedded in File 3.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the per-arm sensor channel inventory.
- `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`. Source for the L0 to L3 pyramid table.
- `competitions/instructions/one_minute_variant/zenodo_archive_protocol.md`. Source for the Zenodo deposition layout that the project.yaml zenodo block references.
- `competitions/instructions/runtime_environments.md`. Source for the parent runtime recipes that File 1 reproduces.
- `competitions/instructions/ascii_diagram_guide.md`. Source for the ASCII drawing rules used by File 8.
- `competitions/instructions/file_format_conventions.md`. Source for the TOML, YAML, and Markdown conventions.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules.
- `LICENSE`. Parent repository MIT license used verbatim by File 7.
- `requirements.txt`. Source for the package versions that `pyproject.toml` must remain compatible with.
- `competitions/instructions/commit_01_project_overview.md`. Source for the parent v3.9.0 7-file Commit 1 structure that this 1-minute variant extends to 8 files.
