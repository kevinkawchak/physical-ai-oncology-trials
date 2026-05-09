# Commit 1: Project Overview

This file specifies the seven files the future Claude Code Opus 4.7 1M Max session must author in its first commit. The session must author exactly the files listed and must not author additional files in this commit.

## Goal

Establish the project skeleton: top-level README, system architecture, dependency manifest, multi-service stack, global configuration, and license. The skeleton must allow a fresh clone to install dependencies and run a smoke test even though no Python or schema content exists yet.

## Files to Author

| Order | Path | Format | Approximate size |
|-------|------|--------|-------------------|
| 1 | `competitions/glioblastoma-1hr-trial/README.md` | Markdown | 25 KB |
| 2 | `competitions/glioblastoma-1hr-trial/docs/architecture.md` | Markdown with Mermaid | 18 KB |
| 3 | `competitions/glioblastoma-1hr-trial/pyproject.toml` | TOML | 4 KB |
| 4 | `competitions/glioblastoma-1hr-trial/docker-compose.yml` | YAML | 3 KB |
| 5 | `competitions/glioblastoma-1hr-trial/config/project.yaml` | YAML | 6 KB |
| 6 | `competitions/glioblastoma-1hr-trial/LICENSE.txt` | Text | 1 KB |
| 7 | `competitions/glioblastoma-1hr-trial/docs/architecture_overview.txt` | ASCII text | 6 KB |

The original instruction list named `docs/architecture.svg`. The future session must replace the SVG with the ASCII file `docs/architecture_overview.txt` plus a Mermaid block embedded in `docs/architecture.md`. SVG is rejected here because the architecture diagram embeds a 50-channel sensor manifold that cannot be rendered cleanly in static SVG and because the existing repository convention in `new-trial/national-24-7-trial/hour-XX/hour_XX_diagram_*.txt` uses ASCII for the same kind of diagram.

## File 1: README.md

Sections required, in order:

1. Title block with v3.9.0 release badge, DOI badge, resolution badge, license badge, Python version badge.
2. Project narrative: the on-premises LLM thesis, the patient (PAT-GBM-0001), the procedure, the robot (ROSA ONE Brain), the resolution (1 ms), the duration (1 hour).
3. Quick Start: the three runtime recipes from `competitions/instructions/runtime_environments.md` reproduced verbatim.
4. Repository tree of `competitions/glioblastoma-1hr-trial/` showing all files that will exist after Commit 5.
5. Per-commit roadmap: the seven commits described in one paragraph each.
6. Verification block from `runtime_environments.md`.
7. Citation block: the v3.9.0 DOI plus a BibTeX snippet.
8. License pointer to `LICENSE.txt`.

The README must include single dashes only and black text only. The README must not include any em dashes, double dashes, or triple dashes.

## File 2: docs/architecture.md

Sections required, in order:

1. Architecture narrative.
2. Mermaid diagram using the `flowchart LR` template from `competitions/instructions/ascii_diagram_guide.md` template 4.
3. Five-phase procedure timeline table reproduced from `competitions/instructions/glioblastoma_context.md`.
4. Sensor channel summary table reproduced from `competitions/instructions/robot_specification.md`.
5. Coordinate frame diagram in ASCII (template 1 from `ascii_diagram_guide.md`).
6. Pointer to `architecture_overview.txt` for the full ASCII facility view.

## File 3: pyproject.toml

Required content:

```
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "glioblastoma-1hr-trial"
version = "3.9.0"
description = "On-prem LLM controlled glioblastoma stereotactic resection simulation at 1 ms resolution."
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
]

[project.optional-dependencies]
dev = ["ruff>=0.5", "yamllint>=1.33", "pytest>=7.4"]
llm-local = ["ollama>=0.3"]

[tool.setuptools.packages.find]
where = ["src"]
```

The `pyproject.toml` is a subset of the parent repository's `requirements.txt`. The future session must verify the subset is compatible by running `pip install -e .` in a clean Python 3.10 venv.

## File 4: docker-compose.yml

Required services:

- `llm`: on-prem language model service, default Ollama image.
- `ingest`: sensor stream consumer, mounts `data/` volume.
- `simulator`: Rust runner, builds from `src/simulation/`.
- `db`: DuckDB sidecar, mounts `data/iterations/` volume.

The `docker-compose.yml` must use named volumes for `data/` and `logs/`. The compose file must be `yamllint -d relaxed` clean.

## File 5: config/project.yaml

Required keys:

```
---
project:
  name: glioblastoma-1hr-trial
  version: "3.9.0"
  release_date: "2026-05-09"

units:
  time: millisecond
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
  duration_seconds: 3600.000
  tick_seconds: 0.001
  total_ticks: 3600000

paths:
  data: data
  config: config
  logs: logs
  results: results
  releases: releases

robot:
  make: Medtronic
  model: ROSA ONE Brain
  hardware_revision: v3.0
  firmware: "3.1.4"
  sample_rate_hz: 1000
```

## File 6: LICENSE.txt

Verbatim MIT License text matching the parent repository's `LICENSE` file. The file is named `LICENSE.txt` to match the original instruction list. The future session must use the same MIT text used by the parent repository.

## File 7: docs/architecture_overview.txt

Single ASCII page using template 1 from `competitions/instructions/ascii_diagram_guide.md`. Shows the operating suite snapshot, the robot identity, the patient identity, and the sensor stream destinations. 60 lines maximum.

## Validation After Commit 1

The future session must verify the following after Commit 1 lands:

- `pip install -e .` succeeds inside a fresh Python 3.10 venv.
- `ruff format --check .` passes.
- `ruff check .` passes.
- `yamllint -d relaxed competitions/glioblastoma-1hr-trial/config/` passes.
- `docker compose -f competitions/glioblastoma-1hr-trial/docker-compose.yml config` parses without error.

## Source Files Cited

- `competitions/instructions/runtime_environments.md`. Source for the three runtime recipes that the README copies verbatim.
- `competitions/instructions/glioblastoma_context.md`. Source for the patient and procedure timeline.
- `competitions/instructions/robot_specification.md`. Source for the robot make and model and the sensor channel table.
- `competitions/instructions/ascii_diagram_guide.md`. Source for the templates that File 2 and File 7 use.
- `LICENSE`. Parent repository MIT license used verbatim by File 6.
- `requirements.txt`. Source for the package versions that `pyproject.toml` must remain compatible with.
- `new-trial/national-24-7-trial/README.md`. Source for the README structure including DOI badges and per-section narrative.
- `patients/paper/full-paper/README.md`. Source for the multi-badge release-block pattern that the v3.9.0 README mirrors.
