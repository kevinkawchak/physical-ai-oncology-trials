# Glioblastoma Robotic Surgery Simulation Instructions (v3.9.0)

Released on 9 May 2026
CEO Kevin Kawchak, ChemicalQDevice

[![Release](https://img.shields.io/badge/Release-v3.9.0-brightgreen.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445179-blue)](https://doi.org/10.5281/zenodo.18445179)
[![Resolution](https://img.shields.io/badge/Resolution-1ms-blue.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

This directory contains the v3.9.0 instruction set that a future Claude Code Opus 4.7 1M Max session will execute to produce a complete, end-to-end glioblastoma robotic surgery clinical trial simulation at millisecond (0.001 s) resolution. No simulation files are produced by this PR. The future session reads these instruction files and authors the simulation across seven sequential commits within a single pull request.

## Thesis

On-premises repository based LLMs provide commands to standard oncology surgical robots based on real-time sensor data and controlled via x, y, z coordinates to administer patient treatment. This workflow minimizes single robot error potential. The instruction set in this directory operationalizes that thesis for a 1-hour glioblastoma resection simulation with millisecond resolution sensor and command streams.

## Scope

- One simulated patient: PAT-GBM-0001 (62-year-old, IDH-wildtype glioblastoma WHO grade 4, right frontal lobe, 4.2 cm maximum diameter).
- One surgical procedure: stereotactic-guided open craniotomy with maximal safe resection.
- One simulation duration: 1 hour (3,600 seconds, 3,600,000 ms ticks).
- One primary surgical robot: ROSA ONE Brain (Medtronic) v3.0; firmware 3.1.4.
- Sensor sampling rate: 1 kHz (1 sample per millisecond per channel).
- Iteration count: 64 deterministic iterations per benchmarked configuration.
- Competition: this project's robot vs. prior version snapshots vs. competitor robots vs. hybrid human-robot teams.

## Why a Future Pass

A 1-hour millisecond resolution surgical simulation contains 3.6 million records per sensor channel. Authoring that volume of data inline as markdown would exceed the working memory of any single LLM session. The instruction set therefore directs the future Claude Code session to (a) author small generator scripts that produce the full data files at runtime and (b) author small human-review samples directly. This mirrors the chunking pattern used by the existing competition input papers under `competitions/inputs/paper-a/` and `competitions/inputs/paper-b/`, which were prepared as instructions for the same kind of downstream LLM processing pass.

## Instruction Files

| File | Purpose |
|------|---------|
| `README.md` | This file. Top-level orientation and table of contents. |
| `glioblastoma_context.md` | Patient, disease, and procedure context. |
| `robot_specification.md` | ROSA ONE Brain make, model, sensors, and limits. |
| `chunking_strategy.md` | How to chunk millisecond data so a future LLM does not exceed memory. |
| `file_format_conventions.md` | Repository-wide file format defaults. |
| `ascii_diagram_guide.md` | ASCII and Mermaid diagram replacements for SVG. |
| `runtime_environments.md` | MacOS, Windows, and Linux execution recipes. |
| `competition_protocol.md` | How to compare runs against prior versions and competitor robots. |
| `ci_compliance_checklist.md` | Pre-commit ruff and yamllint checklist. |
| `pr_workflow.md` | Seven-commit single-PR workflow definition. |
| `commit_01_project_overview.md` | Future Commit 1 file list and authoring instructions. |
| `commit_02_sensor_specifications.md` | Future Commit 2 file list and authoring instructions. |
| `commit_03_xyz_mapping.md` | Future Commit 3 file list and authoring instructions. |
| `commit_04_iteration_design.md` | Future Commit 4 file list and authoring instructions. |
| `commit_05_comparison_competition.md` | Future Commit 5 file list and authoring instructions. |
| `commit_06_error_fixes.md` | Future Commit 6 error-review and patch instructions. |
| `commit_07_repository_updates.md` | Future Commit 7 README, CHANGELOG, releases.md instructions. |

## Future Output Tree (Reference)

```
competitions/glioblastoma-1hr-trial/
  README.md
  docs/
    architecture.md
    sensor_spec.md
    coordinate_mapping.md
    iteration_design.md
    comparison_methodology.md
  config/
    project.yaml
    kinematics.yaml
    iterations.yaml
  schemas/
    sensor_record.schema.json
    sensor_record.proto
    sensor_record.avsc
    xyz_command.schema.json
    xyz_command.proto
    metrics.schema.json
  src/
    sensors/ingest.py
    mapping/sensor_to_xyz.py
    control/robot_loop.cpp
    simulation/iterate.py
    simulation/runner.rs
    metrics/compute.py
    llm/compare_agent.py
  data/
    sensor_sample.jsonl
    sensor_sample.csv
    sensor_1hr.parquet
    xyz_trace_sample.csv
    xyz_trace_1hr.parquet
    iterations/
      run_<id>.parquet
      index.jsonl
      aggregate.duckdb
    human_surgeon_baseline.csv
    robot_outcomes.parquet
  prompts/
    comparison_prompt.md
  results/
    comparison.json
    comparison_report.md
    comparison_report.pdf
  viz/
    xyz_path.txt
    metrics_dashboard.html
    metrics_summary.png
  notebooks/
    iteration_analysis.ipynb
  logs/
    iteration_run.txt
  pyproject.toml
  docker-compose.yml
  LICENSE.txt
```

The output tree intentionally lives at `competitions/glioblastoma-1hr-trial/` so that this `instructions/` directory remains a pure specification and the generated simulation lives next to its peers in `competitions/`.

## Source Citations

The instruction set draws on existing repository content. Each instruction file cites its sources inline with relative paths. Primary anchors are listed below for orientation.

- A. `new-trial/` and subdirectories. Source for the 7-files-per-hour pattern, the ASCII diagram convention, the PSL framework, and the 10 robot type taxonomy used for facility-level visualization. Cited under `commit_01_project_overview.md` and `ascii_diagram_guide.md`.
- B. `new-trial/national-24-7-trial/paper/full-paper/final-paper/`. Source for the polished LaTeX paper structure and the references.bib pattern that the future Commit 5 PDF report mirrors. Cited under `commit_05_comparison_competition.md`.
- C. `patient-journey/` and subdirectories. Source for the per-stage Python orchestrator pattern (`stage_05_surgery.py` in particular: 21 CFR 50.30 task-order lifecycle, IEC 80601-2-77 force limits, 1 kHz runtime safety monitoring) that the future Commit 3 control loop honors. Cited under `commit_03_xyz_mapping.md` and `robot_specification.md`.
- D. `competitions/inputs/`. Source for the chunking pattern (paper-a, paper-b) and the competition format (site-1, Orbit Wars Kaggle competition) that the future Commit 5 comparison protocol mirrors. Cited under `chunking_strategy.md`, `competition_protocol.md`, and `commit_05_comparison_competition.md`.

## Conventions Inherited Repository-Wide

- All instruction files use single dashes only. No em dashes, no double dashes, no triple dashes.
- All instruction files use black text only. No color overrides, no inline color spans.
- All instruction files use plain GitHub Flavored Markdown.
- All future generated diagrams use ASCII text inside `.txt` files or Mermaid blocks inside `.md` files. SVG files are not produced for high-frequency time series because a 3.6 million point path would exceed practical SVG size budgets and trigger browser slowdowns. SVG remains acceptable for static low-density schematics.

## Single-PR Single-Prompt Workflow

This directory was produced by a single prompt across seven commits within one pull request, using the same 7-commit pattern that the future simulation pass will follow. The seven commits in this PR are:

1. README, shared reference files, and `commit_01_project_overview.md`.
2. `commit_02_sensor_specifications.md`.
3. `commit_03_xyz_mapping.md`.
4. `commit_04_iteration_design.md`.
5. `commit_05_comparison_competition.md`.
6. `commit_06_error_fixes.md` and `commit_07_repository_updates.md` (meta files), plus any cross-file fixes.
7. Repository-wide updates to `README.md`, `releases.md` (v3.9.0), and `CHANGELOG.md`.
