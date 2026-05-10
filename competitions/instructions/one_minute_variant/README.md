# Glioblastoma Robotic Surgery Simulation Instructions: 1-Minute Variant (v3.9.1)

Released on 10 May 2026
CEO Kevin Kawchak, ChemicalQDevice

[![Release](https://img.shields.io/badge/Release-v3.9.1-brightgreen.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445179-blue)](https://doi.org/10.5281/zenodo.18445179)
[![Resolution](https://img.shields.io/badge/Resolution-1ms-blue.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Variant](https://img.shields.io/badge/Variant-1%20Minute-orange.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/competitions/instructions/one_minute_variant)

This directory contains the v3.9.1 1-minute variant of the v3.9.0 instruction set. A future Claude Code Opus 4.7 1M Max session executes both directories together to produce a complete, end-to-end 1-minute glioblastoma robotic surgery clinical trial simulation at millisecond (0.001 s) resolution. No simulation files are produced by this PR. The future session reads the parent instructions in `competitions/instructions/` together with the variant overrides here, and authors the simulation across seven sequential commits within a single pull request.

## Thesis

On-premises repository based LLMs provide commands to standard oncology surgical robots based on real-time sensor data and controlled via x, y, z coordinates to administer patient treatment. This workflow minimizes single robot error potential. The 1-minute variant operationalizes that thesis at the upper edge of feasible robotic speed: the entire glioblastoma resection completes in 60 seconds across four cooperating arms that share a single deterministic real-time bus.

## Inheritance from the v3.9.0 Parent Instruction Set

This 1-minute variant is a sibling layer that depends on the parent instructions at `competitions/instructions/`. The future session reads the parent files for shared context and reads the variant files for 1-minute-specific overrides. The full inheritance map is reproduced below so that this README is self-standing.

| Parent file (read for shared context) | Variant override (read for 1-minute scenario) |
|---------------------------------------|-----------------------------------------------|
| `competitions/instructions/README.md` | `competitions/instructions/one_minute_variant/README.md` (this file) |
| `competitions/instructions/glioblastoma_context.md` | `competitions/instructions/one_minute_variant/glioblastoma_context_1min.md` |
| `competitions/instructions/robot_specification.md` | `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md` |
| `competitions/instructions/chunking_strategy.md` | `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md` (Layer 4 addendum) |
| `competitions/instructions/file_format_conventions.md` | inherited; variant adds zstd-3 default for Parquet |
| `competitions/instructions/ascii_diagram_guide.md` | inherited; variant adds the 4-arm coordination template |
| `competitions/instructions/runtime_environments.md` | inherited; variant adds the Mac M3 Ultra and A100 GPU recipes documented in `commit_04_iterations_1min.md` |
| `competitions/instructions/competition_protocol.md` | inherited; variant changes the per-round time budget from 1 hour to 1 minute via `commit_05_competition_1min.md` |
| `competitions/instructions/ci_compliance_checklist.md` | inherited; variant adds the 10 MB committed-file cap and the 5 MB committed-Parquet cap |
| `competitions/instructions/pr_workflow.md` | inherited; variant uses the same 7-commit single-PR pattern with different file lists |
| `competitions/instructions/commit_01_project_overview.md` | `competitions/instructions/one_minute_variant/commit_01_overview_1min.md` |
| `competitions/instructions/commit_02_sensor_specifications.md` | `competitions/instructions/one_minute_variant/commit_02_sensors_1min.md` |
| `competitions/instructions/commit_03_xyz_mapping.md` | `competitions/instructions/one_minute_variant/commit_03_xyz_4arm.md` |
| `competitions/instructions/commit_04_iteration_design.md` | `competitions/instructions/one_minute_variant/commit_04_iterations_1min.md` |
| `competitions/instructions/commit_05_comparison_competition.md` | `competitions/instructions/one_minute_variant/commit_05_competition_1min.md` |
| `competitions/instructions/commit_06_error_fixes.md` | inherited; variant runs the same seven checks against the 1-minute output tree |
| `competitions/instructions/commit_07_repository_updates.md` | inherited; variant uses the v3.9.1 release notes block defined in `commit_05_competition_1min.md` |

## Scope

- One simulated patient: PAT-GBM-0001 (62-year-old, IDH-wildtype glioblastoma WHO grade 4, right frontal lobe, 4.2 cm maximum diameter), shared with the parent v3.9.0 instructions.
- One surgical procedure: stereotactic-guided open craniotomy with maximal safe resection. The pre-op anesthesia, registration, dural opening, and multi-arm setup are precomputed and frozen at simulation start; the simulation begins at the moment the four arms are docked and ready.
- One simulation duration: 60 seconds (60,000 ms ticks for 1 kHz channels and 600,000 ticks for 10 kHz force channels).
- One primary surgical robot: hypothetical 2030 Medtronic NeuroSpeed 1.0 multi-arm parallel stereotactic neurosurgical robot. The current SOTA Medtronic ROSA ONE Brain v3.0 cannot perform a 1-minute glioblastoma resection because its tissue removal rate, end-effector velocity, joint angular velocity, E-stop latency, positioning accuracy, and force resolution are each 5 to 200 times short of the requirement.
- Sensor sampling rate: mixed 10 kHz force channels and 1 kHz other channels per arm.
- Channel schema: 50 channels per arm times 4 arms equals 200 total channels.
- Iteration count: 16 deterministic iterations per benchmarked configuration. Reduced from the v3.9.0 64 iterations because the 1-minute scenario has fewer free parameters and the per-iteration committed data is doubled by the 4-arm telemetry.
- Competition: this project's 1-minute robot run versus the parent v3.9.0 1-hour ROSA ONE Brain run versus prior 1-minute releases.

## New Procedure Phase Timeline (4 phases, 60 seconds total)

The 1-minute scenario requires its own phase boundaries. The parent `glioblastoma_context.md` 5-phase 1-hour timeline does not apply to this variant. The variant timeline is fixed in `glioblastoma_context_1min.md` and is reproduced here for orientation.

| Phase | Start (s) | End (s) | Duration (s) | Description |
|-------|-----------|---------|--------------|-------------|
| Pre-op (precomputed, not in committed simulation) | T-1800 | T+0 | 30 minutes | Anesthesia, registration, dural opening, multi-arm setup; frozen at simulation start |
| Phase 1 dural opening final and exposure | 0.000 | 5.000 | 5 s | Final dural opening, ultrasound rapid mapping, 5-ALA UV on |
| Phase 2 bulk tumor resection | 5.000 | 45.000 | 40 s | All four arms active; arm 1 cuts at 800 mm cubed per second; arm 2 coagulates; arm 3 suctions; arm 4 images |
| Phase 3 margin assessment and fine resection | 45.000 | 55.000 | 10 s | Arm 1 reduces removal rate to 200 mm cubed per second; arm 4 increases imaging to 100 fps |
| Phase 4 hemostasis verification and arm withdrawal | 55.000 | 60.000 | 5 s | Arms 1 and 3 retract; arm 2 final hemostasis pass; arm 4 records final margin scan |

## Why a Future Pass

The parent README explains the chunking rationale at length. For the 1-minute variant the per-iteration committed data is approximately 510 KB across pyramid levels L1, L2, L3, and the event log. The committed total across 16 iterations is approximately 8.2 MB plus 1.5 MB of fixed overhead, for a total of 9.7 MB. This sits inside the GitHub 10 MB single-file cap with 0.3 MB headroom. The full L0 raw at 1 kHz mixed with 10 kHz force is 26 MB per iteration and 416 MB across 16 iterations; the L0 raw is archived to Zenodo per `zenodo_archive_protocol.md` and is never committed to Git.

## Instruction Files (this directory)

| File | Purpose |
|------|---------|
| `README.md` | This file. Top-level orientation and table of contents for the 1-minute variant. |
| `glioblastoma_context_1min.md` | 4-phase 1-minute procedure timeline; pre-op precomputed; same patient PAT-GBM-0001. |
| `robot_specification_neurospeed.md` | Medtronic NeuroSpeed 1.0 specification; 4 arms; 1,000 mm/s; 10 kHz force; 800 mm cubed per second. |
| `sensor_specification_10khz.md` | 10 kHz force sensors per arm with 1 kHz command sensors; total 200 channels (50 per arm times 4 arms). |
| `multi_arm_coordination.md` | Inter-arm collision avoidance protocol; 1 kHz heartbeat; 100 microsecond emergency arm-park trigger. |
| `file_size_pyramid_1min.md` | Pyramid 4 budget table; per-iteration committed budget of 510 KB across L1 plus L2 plus L3 plus events. |
| `commit_01_overview_1min.md` | Future Commit 1 file list for the 1-minute variant; reuses parent commit_01 plus the 4 new shared files. |
| `commit_02_sensors_1min.md` | Sensor specs covering 200 channels at mixed 10 kHz / 1 kHz rates; pyramid output schema. |
| `commit_03_xyz_4arm.md` | Coordinate mapping for 4 arms; per-arm safety zone gating; cross-arm coordination. |
| `commit_04_iterations_1min.md` | 16-iteration sweep at 1 minute; 20 Hz committed L1; 1 kHz mixed 10 kHz force Zenodo L0. |
| `commit_05_competition_1min.md` | Comparison vs ROSA ONE 1-hour baseline; vs prior 1-minute releases. |
| `zenodo_archive_protocol.md` | DOI assignment, deposition layout, SHA-256 manifest contract for the 416 MB L0 archive. |

## Future Output Tree (Reference)

The output tree intentionally lives at `competitions/glioblastoma-1min-trial/` so that this variant's instructions remain a pure specification and the generated simulation lives next to its peers in `competitions/`. The output tree is parallel to the parent v3.9.0 output tree at `competitions/glioblastoma-1hr-trial/`; nothing in the parent tree is modified.

```
competitions/glioblastoma-1min-trial/
  README.md
  docs/
    architecture.md
    sensor_spec.md
    coordinate_mapping.md
    iteration_design.md
    comparison_methodology.md
    multi_arm_coordination.md
  config/
    project.yaml
    kinematics_4arm.yaml
    iterations.yaml
  schemas/
    sensor_record_4arm.schema.json
    sensor_record_4arm.proto
    sensor_record_4arm.avsc
    xyz_command_4arm.schema.json
    xyz_command_4arm.proto
    metrics.schema.json
  src/
    sensors/ingest_4arm.py
    mapping/sensor_to_xyz_4arm.py
    control/robot_loop_4arm.cpp
    coordination/arm_heartbeat.cpp
    simulation/iterate_1min.py
    simulation/runner_1min.rs
    metrics/compute_1min.py
    llm/compare_agent_1min.py
  data/
    sensor_sample_4arm.jsonl
    sensor_sample_4arm.csv
    iterations/
      run_NNNNN_L1_50ms.parquet
      run_NNNNN_L2_1s.parquet
      run_NNNNN_L3_phase.parquet
      run_NNNNN_events.parquet
      run_NNNNN_L0_raw.zenodo_pointer.json
      index.jsonl
      aggregate.duckdb
    human_surgeon_baseline.csv
    robot_outcomes_1min.parquet
  prompts/
    comparison_prompt_1min.md
  results/
    comparison.json
    comparison_report.md
    comparison_report.pdf
  viz/
    xyz_path_4arm.txt
    metrics_dashboard.html
    metrics_summary.png
  notebooks/
    iteration_analysis_1min.ipynb
  logs/
    iteration_run.txt
  releases/
    v3.9.1/
      manifest.json
      metrics.json
      iterations_index.jsonl
      sample_seeds.txt
      zenodo_doi.txt
  pyproject.toml
  docker-compose.yml
  LICENSE.txt
```

## Conventions Inherited Repository-Wide

- All instruction files in this directory use single dashes only. No em dashes, no double dashes, no triple dashes.
- All instruction files use black text only. No color overrides, no inline color spans.
- All instruction files use plain GitHub Flavored Markdown.
- All future generated diagrams use ASCII text inside `.txt` files or Mermaid blocks inside `.md` files. SVG files are not produced for high-frequency time series. SVG remains acceptable for static low-density schematics under 100 KB.
- All committed files in the 1-minute variant must remain under 10 MB; all committed Parquet files must remain under 5 MB. The CI compliance addendum in `commit_06_error_fixes.md` enforces this.

## Single-PR Single-Prompt Workflow

This 1-minute variant directory was produced by a single prompt across seven commits within one pull request, using the same 7-commit pattern that the future simulation pass will follow. The seven commits in this PR are:

1. README, `glioblastoma_context_1min.md`, and `robot_specification_neurospeed.md`.
2. `sensor_specification_10khz.md` and `file_size_pyramid_1min.md`.
3. `multi_arm_coordination.md` and `commit_03_xyz_4arm.md`.
4. `commit_01_overview_1min.md`, `commit_02_sensors_1min.md`, and `commit_04_iterations_1min.md`.
5. `commit_05_competition_1min.md` and `zenodo_archive_protocol.md`.
6. Error fixes across all 12 variant files; addresses any lint, format, or cross-reference issues that would cause the GitHub `Cl / lint-and-format (3.10) (pull...)`, `(3.11)`, and `(3.12)` checks to fail.
7. Repository-wide updates to `README.md`, `releases.md` (v3.9.1 release notes block), and `CHANGELOG.md`.

## Source Citations

The 1-minute variant draws on existing repository content. Each instruction file cites its sources inline with relative paths. Primary anchors are listed below for orientation.

- A. `competitions/instructions/`. Source for the chunking strategy, file format conventions, ASCII diagram guide, runtime environments, competition protocol, CI compliance checklist, and 7-commit single-PR workflow that this variant inherits verbatim.
- B. `patient-journey/stage_05_surgery.py`. Source for the IEC 80601-2-77 force limits and 21 CFR 50.30 task-order lifecycle that the 4-arm control loop honors per arm.
- C. `new-trial/national-24-7-trial/hour-00/`. Source for the ASCII diagram template family that the new 4-arm coordination template extends.
- D. `competitions/inputs/`. Source for the chunking pattern (paper-a, paper-b) and the competition format (site-1, Orbit Wars Kaggle competition) that the 1-minute competition protocol mirrors.
