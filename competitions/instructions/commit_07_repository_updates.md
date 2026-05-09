# Commit 7: Repository Updates

This file specifies the three repository-wide updates that the future Claude Code Opus 4.7 1M Max session must author in its seventh and final commit. The session must touch only the three files listed and must not modify any file under `competitions/glioblastoma-1hr-trial/` in this commit.

## Goal

Surface the v3.9.0 release across the parent repository's main documentation entry points: the top-level README badge block, the top-level architecture diagram block, the `releases.md` entry, and the `CHANGELOG.md` entry. The future session also adds a v3.9.0 row to the parent repository's `tests/` markers if a test marker registry exists.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate diff size |
|-------|------|--------|--------------------|------------------------|
| 1 | `README.md` (parent repository root) | Markdown | Hand-authored patch | 60 lines added |
| 2 | `releases.md` (parent repository root) | Markdown | Hand-authored patch | 80 lines added |
| 3 | `CHANGELOG.md` (parent repository root) | Markdown | Hand-authored patch | 40 lines added |

## File 1: README.md

Required edits:

1. Replace the version badge from `v3.8.0` to `v3.9.0`.
2. Replace the "Last Updated" badge month if the release crosses a month boundary.
3. Add a new release block immediately above the existing v3.8.0 block:

```
**5/9: v3.9.0 (Glioblastoma Robotic Surgery Simulation)** *On-prem LLM
controlled glioblastoma stereotactic resection simulation at 1 ms
resolution for 1 hour with 64-iteration sweep* - Single-patient
PAT-GBM-0001 simulation against the Medtronic ROSA ONE Brain v3.0,
firmware 3.1.4, with on-prem LLM comparison agent ranking this
project's robot against prior versions, competitor robots, and hybrid
human-robot teams. [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445179-blue)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/competitions/instructions)
```

4. Add a v3.9.0 architecture diagram block immediately above the v3.8.0 architecture diagram block:

```
## Glioblastoma Robotic Surgery Simulation Instructions (v3.9.0)

  +---------------------------------------------------------------------+
  |  Sensor stream            xyz commands         Comparison agent     |
  |  (50 channels @ 1kHz) --> (1 kHz, phase-       (on-prem LLM,        |
  |                            conditioned)   -->   skill rating)       |
  |                                                                     |
  |  3.6M ticks/hr      -->  ~2.73M commands  -->  pairwise tournament  |
  |  Parquet 60 MB           Parquet 90 MB         Plotly dashboard     |
  +---------------------------------------------------------------------+
        |                          |                          |
        v                          v                          v
  +---------------------------------------------------------------------+
  | v3.9.0: 7-commit single-PR instruction set at                       |
  | competitions/instructions/ for a future Claude Code Opus 4.7 1M Max |
  | session to author the simulation across 7 future commits.           |
  +---------------------------------------------------------------------+
```

5. Update the Repository Structure tree to add the new `competitions/instructions/` block listing the 17 instruction files. The tree update is appended to the existing `competitions/` block; the existing tree content is preserved verbatim.

## File 2: releases.md

Required edits:

1. Insert the v3.9.0 entry as the first entry in the file (above the existing v3.8.0 entry), using the format below verbatim. The entry follows the same `## Summary` / `## Features` / `## Contributors` / `## Notes` structure as v3.8.0 and v3.7.0.

```
Glioblastoma Robotic Surgery Simulation Instructions (v3.9.0)
v3.9.0 - Glioblastoma Robotic Surgery Simulation Instructions

## Summary

- Added a 17-file instruction set at competitions/instructions/ for a
  future Claude Code Opus 4.7 1M Max session to author a complete
  end-to-end glioblastoma robotic surgery clinical trial simulation at
  millisecond (0.001 s) resolution across seven sequential commits in a
  single pull request.
- The instruction set fixes the patient (PAT-GBM-0001, 62F, IDH-wildtype
  glioblastoma WHO grade 4, right frontal lobe, 4.2 cm), the procedure
  (stereotactic-guided right frontal craniotomy with maximal safe
  resection, 1 hour, 5 phases), the surgical robot (Medtronic ROSA ONE
  Brain v3.0 firmware 3.1.4 with Stealth Autoguide for biopsy and
  Modus V from Synaptive Medical for visualization), the sensor suite
  (50 channels at 1 kHz including joint position / velocity / torque,
  end-effector pose and force / torque, navigation deviation, tool
  state flags, and safety zone enums), and the iteration count (64
  deterministic iterations covering seed, sensor noise sigma, force
  feedback gain, and IK solver tolerance sweeps).
- The instruction set explicitly replaces SVG diagrams with ASCII text
  inside .txt files and Mermaid blocks inside .md files for any
  high-frequency time series; a 1-hour 1 kHz path of 3.6 million points
  cannot be rendered cleanly as static SVG.
- The instruction set inherits the multi-chunk processing pattern from
  competitions/inputs/paper-a/ (CodeClash) and competitions/inputs/paper-b/
  (FAERS) and inherits the Gaussian N(mu, sigma squared) skill rating
  from competitions/inputs/site-1/ (Orbit Wars Kaggle competition).
- The instruction set inherits the 21 CFR 50.30 task-order lifecycle,
  IEC 80601-2-77 force limits (15.0 N tip, 5.0 N lateral), 50 ms E-stop
  latency budget, and 1 kHz runtime safety monitoring cadence from
  patient-journey/stage_05_surgery.py and patient-journey/patient_state.py.
- The competition protocol supports three competitor categories: prior
  versions of this project (snapshot per release), competitor robots
  (Renishaw NeuroMate, Brainlab Cirq, Synaptive Modus V, Mazor X, manual
  surgery), and hybrid human-robot teams (any non-zero
  human_intervention_seconds value in the metric record).

## Features

- competitions/instructions/README.md ties the 16 sibling instruction
  files together and lists the future output tree under
  competitions/glioblastoma-1hr-trial/.
- competitions/instructions/glioblastoma_context.md fixes the patient,
  disease, procedure, and 5-phase timeline (setup 0-600s, dural opening
  600-900s, tumor resection coarse 900-2400s, tumor resection fine
  2400-3300s, hemostasis and closure prep 3300-3600s).
- competitions/instructions/robot_specification.md fixes the Medtronic
  ROSA ONE Brain v3.0 kinematic limits (6 DOF, 0.5 mm RMS accuracy,
  50 mm/s max linear velocity, 200 mm/s squared max linear acceleration),
  the 50-channel sensor suite at 1 kHz, and the safety limits inherited
  from IEC 80601-2-77 and 21 CFR 50.30.
- competitions/instructions/chunking_strategy.md defines the three-layer
  strategy (generators not data, per-commit file budgets, within-file
  chunking) so that a future LLM session does not exceed working memory
  while authoring the millisecond resolution simulation.
- competitions/instructions/file_format_conventions.md fixes the
  defaults: .md narrative, .pdf reports, .json structured outputs, .jsonl
  streaming, .yaml configuration, .toml Python packaging, .parquet
  high-volume data, .csv human-review samples, .schema.json / .proto /
  .avsc schemas, .py / .cpp / .rs source code, .txt ASCII diagrams,
  .png static charts, .html interactive dashboards, .duckdb analytical
  store.
- competitions/instructions/ascii_diagram_guide.md gives concrete ASCII
  templates inheriting from new-trial/national-24-7-trial/hour-00/
  hour_00_diagram_*.txt and Mermaid templates for box-and-arrow
  architecture diagrams.
- competitions/instructions/runtime_environments.md gives the MacOS
  (Apple Silicon, macOS 14 Sonoma), Windows (Windows 11, PowerShell 7),
  Linux (Ubuntu 22.04 LTS), Docker, and conventional high-end server
  recipes for running the simulation locally.
- competitions/instructions/competition_protocol.md defines the
  three-category competitor model, the five comparison dimensions
  (Quality 0.40, Time 0.25, Cost 0.20, Safety 0.10, Patient Experience
  0.05), the Gaussian skill rating with mu_0 = 600 and sigma_0 = 200,
  the multi-round tournament structure (default size 8 per release),
  the on-premise LLM constraint, and the per-release snapshot pattern.
- competitions/instructions/ci_compliance_checklist.md lists the
  pre-commit ruff format, ruff check, and yamllint commands that the
  future session must run before each push to avoid the lint-and-format
  CI failures observed in v3.7.0 and v3.8.0.
- competitions/instructions/pr_workflow.md defines the seven-commit
  single-PR pattern, the branch naming convention, the per-commit
  procedure, and the autonomy rule (the future session must not stall,
  ask questions, or enter plan mode between commits).
- competitions/instructions/commit_01_project_overview.md specifies the
  seven Commit 1 files (README, architecture.md with Mermaid block,
  pyproject.toml, docker-compose.yml, project.yaml, LICENSE.txt, and
  the architecture_overview.txt that replaces the original
  architecture.svg).
- competitions/instructions/commit_02_sensor_specifications.md specifies
  the eight Commit 2 files (sensor_spec.md, sensor_record.schema.json,
  .proto, .avsc, sensor_sample.jsonl, sensor_1hr.parquet,
  sensor_sample.csv, ingest.py).
- competitions/instructions/commit_03_xyz_mapping.md specifies the nine
  Commit 3 files (coordinate_mapping.md, xyz_command.schema.json,
  .proto, kinematics.yaml, sensor_to_xyz.py, robot_loop.cpp,
  xyz_trace_1hr.parquet, xyz_trace_sample.csv, xyz_path.txt that
  replaces the original xyz_path.svg).
- competitions/instructions/commit_04_iteration_design.md specifies the
  ten Commit 4 files (iteration_design.md, iterations.yaml, iterate.py,
  runner.rs with its Cargo.toml, run_NNNNN.parquet times 64,
  index.jsonl, aggregate.duckdb, iteration_analysis.ipynb,
  iteration_run.txt).
- competitions/instructions/commit_05_comparison_competition.md
  specifies the twelve Commit 5 files (comparison_methodology.md,
  metrics.schema.json, human_surgeon_baseline.csv, robot_outcomes.parquet,
  compute.py, compare_agent.py, comparison_prompt.md, comparison.json,
  comparison_report.md, comparison_report.pdf, metrics_dashboard.html,
  metrics_summary.png) plus the immutable v3.9.0 release snapshot at
  competitions/glioblastoma-1hr-trial/releases/v3.9.0/.
- competitions/instructions/commit_06_error_fixes.md specifies the
  seven-check pre-commit error scan and the eight common error
  categories with their patches (mismatched channel count, oversize SVG,
  ruff E501, yamllint trailing whitespace, yamllint missing
  document-start, Cargo.toml version mismatch, forward reference
  triggering F821, cross-document path drift after the SVG-to-ASCII
  replacement).
- competitions/instructions/commit_07_repository_updates.md specifies
  the parent README, releases.md, and CHANGELOG.md edits that surface
  the v3.9.0 release across the repository's main documentation entry
  points.
- All instruction files use single dashes only and black text only. The
  instruction set adds no Python, YAML, or other CI-checked files; the
  lint-and-format CI workflow (ruff format check, ruff check, yamllint)
  on Python 3.10, 3.11, and 3.12 remains green.

## Contributors
@kevinkawchak
@claude
@openai
@google-gemini

## Notes

The v3.9.0 release does not produce simulation files. The release
delivers an instruction set that a future Claude Code Opus 4.7 1M Max
session reads to author the simulation across seven sequential commits
in a single pull request. The instruction set inherits the chunking,
SVG-replacement, and CI-compliance patterns proven in v3.6.0 through
v3.8.0. The future simulation pass will populate
competitions/glioblastoma-1hr-trial/ with approximately 65 simulation
files plus 64 iteration Parquet files, ready for ranking against prior
project versions, competitor robots, and hybrid human-robot teams via
the on-prem LLM comparison agent.
```

## File 3: CHANGELOG.md

Required edits:

1. Move the existing `## [Unreleased]` block contents (if any) into a new `## [3.9.0] - 2026-05-09` block.
2. Use the same format as the existing `## [3.8.0] - 2026-05-06` block.
3. Required content under `### Added`:

```
- competitions/instructions/README.md - Top-level instruction set
  overview that ties the 16 sibling instruction files together and
  lists the future output tree under
  competitions/glioblastoma-1hr-trial/
- competitions/instructions/glioblastoma_context.md - Patient, disease,
  procedure, and 5-phase timeline
- competitions/instructions/robot_specification.md - Medtronic ROSA ONE
  Brain v3.0 specifications, 50-channel sensor suite at 1 kHz, and
  IEC 80601-2-77 plus 21 CFR 50.30 safety limits
- competitions/instructions/chunking_strategy.md - Three-layer chunking
  strategy to keep a future LLM session within working memory
- competitions/instructions/file_format_conventions.md - Repository-wide
  file format defaults
- competitions/instructions/ascii_diagram_guide.md - ASCII and Mermaid
  templates that replace SVG for high-frequency series
- competitions/instructions/runtime_environments.md - MacOS, Windows,
  Linux, Docker, and conventional high-end server recipes
- competitions/instructions/competition_protocol.md - Three-category
  competitor model, five comparison dimensions, Gaussian skill rating,
  multi-round tournament structure, on-premise LLM constraint,
  per-release snapshot pattern
- competitions/instructions/ci_compliance_checklist.md - Pre-commit
  ruff and yamllint checklist
- competitions/instructions/pr_workflow.md - Seven-commit single-PR
  workflow definition
- competitions/instructions/commit_01_project_overview.md through
  commit_07_repository_updates.md - Per-commit file lists and authoring
  instructions for each of the seven future commits
```

4. Required content under `### Changed`:

```
- README.md - Updated v3.8.0 to v3.9.0 badge and added v3.9.0 release
  block above the v3.8.0 block
- releases.md - Added v3.9.0 release entry above v3.8.0
```

## Validation After Commit 7

- `git diff origin/main...HEAD --stat` shows changes only to `README.md`, `releases.md`, and `CHANGELOG.md`.
- The CI workflow runs to completion green for Python 3.10, 3.11, and 3.12.
- `grep "v3.9.0" README.md` returns at least 5 matches.
- `grep "v3.9.0" releases.md` returns at least 3 matches.
- `grep "3.9.0" CHANGELOG.md` returns at least 2 matches.

## Source Files Cited

- `releases.md`. Source for the entry format that the new v3.9.0 entry mirrors verbatim.
- `CHANGELOG.md`. Source for the `## [3.x.0] - YYYY-MM-DD` block format that the new v3.9.0 block mirrors.
- `README.md`. Source for the badge block, release block, architecture diagram block, and Repository Structure tree formats that the v3.9.0 patches augment.
- `competitions/instructions/pr_workflow.md`. Source for the rule that Commit 7 modifies only the three repository-wide documentation files and does not modify any file under `competitions/glioblastoma-1hr-trial/`.
