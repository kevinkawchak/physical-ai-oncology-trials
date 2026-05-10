# Commit 5 (1-Minute Variant): Comparison and Competition

This file specifies the files the future Claude Code Opus 4.7 1M Max session must author in its fifth commit for the 1-minute variant. The session must author exactly the files listed and must not author additional files in this commit. The parent `competitions/instructions/commit_05_comparison_competition.md` lists 12 files for the v3.9.0 1-hour scenario. This 1-minute variant lists 13 files because the 4-arm topology adds a per-arm contribution chart.

## Goal

Define quality, time, and cost metrics for the 1-minute scenario. Author the metric schema, the human surgeon baseline dataset (carried forward from the parent), the aggregated robot outcomes Parquet, the metric computation script, the on-prem LLM comparison agent, the versioned LLM prompt for the 1-minute variant, the structured comparison results, the markdown narrative report, the formal PDF report, the interactive HTML dashboard, the static summary chart, and the per-arm contribution analysis chart. Snapshot the v3.9.1 release at the end of the commit and patch every Zenodo pointer with the real DOI and SHA-256 values.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1min-trial/docs/comparison_methodology.md` | Markdown | Hand-authored | 28 KB |
| 2 | `competitions/glioblastoma-1min-trial/schemas/metrics.schema.json` | JSON Schema 2020-12 | Hand-authored | 12 KB |
| 3 | `competitions/glioblastoma-1min-trial/data/human_surgeon_baseline.csv` | CSV | Hand-authored | 14 KB |
| 4 | `competitions/glioblastoma-1min-trial/data/robot_outcomes_1min.parquet` | Parquet zstd-3 | Script-generated | 1 MB (under 5 MB cap) |
| 5 | `competitions/glioblastoma-1min-trial/src/metrics/compute_1min.py` | Python 3.10 | Hand-authored | 22 KB |
| 6 | `competitions/glioblastoma-1min-trial/src/llm/compare_agent_1min.py` | Python 3.10 | Hand-authored | 24 KB |
| 7 | `competitions/glioblastoma-1min-trial/prompts/comparison_prompt_1min.md` | Markdown | Hand-authored | 14 KB |
| 8 | `competitions/glioblastoma-1min-trial/results/comparison.json` | JSON | Script-generated | 80 KB |
| 9 | `competitions/glioblastoma-1min-trial/results/comparison_report.md` | Markdown | Script-generated | 32 KB |
| 10 | `competitions/glioblastoma-1min-trial/results/comparison_report.pdf` | PDF | Script-generated via pandoc | 4 MB (under 5 MB cap) |
| 11 | `competitions/glioblastoma-1min-trial/viz/metrics_dashboard.html` | HTML (Plotly) | Script-generated | 4 MB (under 5 MB cap) |
| 12 | `competitions/glioblastoma-1min-trial/viz/metrics_summary.png` | PNG | Script-generated | 400 KB |
| 13 | `competitions/glioblastoma-1min-trial/viz/per_arm_contribution.png` | PNG | Script-generated | 350 KB |

The future session must also create the v3.9.1 release snapshot at `competitions/glioblastoma-1min-trial/releases/v3.9.1/` containing five files: `manifest.json`, `metrics.json`, `iterations_index.jsonl`, `sample_seeds.txt`, and `zenodo_doi.txt`. The snapshot directory is created at the end of Commit 5.

The future session must also patch every Zenodo pointer JSON file authored by Commits 2, 3, and 4 with the real DOI, record_id, and SHA-256 values after the Zenodo deposition completes.

## File 1: docs/comparison_methodology.md

Required sections:

1. Comparison goal: rank this project's 1-minute robot run against the parent v3.9.0 1-hour ROSA ONE Brain run, against prior v3.9.1 release snapshots, and against published manual surgical baselines across quality, time, cost, safety, and patient experience.
2. Metric definitions reproduced from `competitions/instructions/competition_protocol.md`:
   - Quality (40 percent weight): composite of resection completeness percentage, eloquent cortex preservation score, and PSL Omniscient and Omnipresent dimensions.
   - Time (25 percent weight): total seconds from procedure start to final E-stop or COMPLETE state. The 1-minute variant trivially beats the 1-hour parent on this dimension; the comparison report must call out that this is structural and not a fair pairwise comparison.
   - Cost (20 percent weight): consumables, robot depreciation amortized over expected lifetime use, OR time, anesthesia time. Liquid nitrogen cooling consumables for the NeuroSpeed 1.0 contribute additional cost.
   - Safety (10 percent weight): inverse of (per-arm force violation count plus cumulative force violation count plus E-stop count plus AE count plus heartbeat miss count).
   - Patient experience (5 percent weight): predicted post-operative KPS at 30 days from a fixed regression model.
3. Composite score formula and weights, frozen for v3.9.1 (same weights as the parent v3.9.0).
4. Skill rating model (Gaussian N(mu, sigma squared)) with mu_0 = 600 and sigma_0 = 200, mirroring the Orbit Wars Kaggle competition and the parent v3.9.0.
5. Multi-round tournament structure (default size 4 for the 1-minute variant due to the smaller iteration count of 16; scalable to 8 if iterations scale to 32).
6. Per-arm contribution analysis: which of the 4 arms contributed the most tissue removal, the most coagulation work, the most suction volume, the most imaging frames, and which had the highest force violation rate.
7. Statistical methods: bootstrap 95 percent confidence intervals across 16 iterations, Mann-Whitney U for pairwise comparisons.
8. Cross-references to `prompts/comparison_prompt_1min.md` and `src/metrics/compute_1min.py`.

## File 2: schemas/metrics.schema.json

JSON Schema 2020-12 with the following structure. Adds three per-arm fields relative to the parent v3.9.0 metrics schema:

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kevinkawchak.github.io/physical-ai-oncology-trials/v3.9.1/metrics.schema.json",
  "title": "GBM 1min Metrics Record",
  "type": "object",
  "required": ["entity_id", "entity_kind", "iteration_id", "release_version",
               "quality_score", "total_seconds", "cost_usd", "safety_score",
               "patient_experience_score", "composite_score",
               "human_intervention_seconds",
               "force_violation_count_per_arm", "cumulative_force_violation_count",
               "estop_count", "ae_count", "heartbeat_miss_count",
               "resection_completeness_pct",
               "eloquent_preservation_score", "predicted_kps_day_30",
               "skill_mu", "skill_sigma",
               "per_arm_tissue_removed_mm3", "per_arm_force_peak_N", "per_arm_active_seconds"],
  "properties": {
    "entity_id": {"type": "string"},
    "entity_kind": {"type": "string", "enum": ["this_project_1min", "this_project_1hr", "prior_version_1min", "competitor_robot", "hybrid_team", "manual_human"]},
    "iteration_id": {"type": "string", "pattern": "^run_[0-9]{5}$"},
    "release_version": {"type": "string", "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$"},
    "quality_score": {"type": "number", "minimum": 0, "maximum": 100},
    "total_seconds": {"type": "number", "minimum": 0, "maximum": 14400},
    "cost_usd": {"type": "number", "minimum": 0},
    "safety_score": {"type": "number", "minimum": 0, "maximum": 100},
    "patient_experience_score": {"type": "number", "minimum": 0, "maximum": 100},
    "composite_score": {"type": "number", "minimum": 0, "maximum": 100},
    "human_intervention_seconds": {"type": "integer", "minimum": 0},
    "force_violation_count_per_arm": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 4, "maxItems": 4},
    "cumulative_force_violation_count": {"type": "integer", "minimum": 0},
    "estop_count": {"type": "integer", "minimum": 0},
    "ae_count": {"type": "integer", "minimum": 0},
    "heartbeat_miss_count": {"type": "integer", "minimum": 0},
    "resection_completeness_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "eloquent_preservation_score": {"type": "number", "minimum": 0, "maximum": 100},
    "predicted_kps_day_30": {"type": "number", "minimum": 0, "maximum": 100},
    "skill_mu": {"type": "number"},
    "skill_sigma": {"type": "number", "minimum": 0},
    "per_arm_tissue_removed_mm3": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 4, "maxItems": 4},
    "per_arm_force_peak_N": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 4, "maxItems": 4},
    "per_arm_active_seconds": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 60}, "minItems": 4, "maxItems": 4}
  },
  "additionalProperties": false
}
```

## File 3: data/human_surgeon_baseline.csv

The future session must reuse the parent v3.9.0 `competitions/glioblastoma-1hr-trial/data/human_surgeon_baseline.csv` verbatim and copy it into the 1-minute output tree. The 30 baseline records from 6 published surgical centers remain authoritative. The 1-minute scenario does not have a published 1-minute manual surgical baseline because no human surgeon can perform a glioblastoma resection in 60 seconds; the comparison report must call this out.

## File 4: data/robot_outcomes_1min.parquet

The future session must produce this file by running `src/metrics/compute_1min.py --aggregate-iterations`. The file aggregates the 16 iteration L1 to L3 plus events Parquet files into per-iteration metric records conforming to `metrics.schema.json`. Each iteration produces one row.

## File 5: src/metrics/compute_1min.py

Python 3.10 module with the following responsibilities:

- Read the 16 iteration L3 per-phase Parquet files and the events Parquet files.
- Compute per-iteration quality, time, cost, safety, and patient experience scores.
- Compute the per-arm tissue removed, per-arm force peak, and per-arm active seconds aggregates.
- Compute the composite score using the v3.9.1 weights (same as v3.9.0).
- Compute the skill mu and sigma using TrueSkill-style update.
- Read `data/human_surgeon_baseline.csv` and concatenate.
- Read the parent v3.9.0 `competitions/glioblastoma-1hr-trial/results/comparison.json` if it exists and concatenate the per-iteration rows from the parent.
- Write `data/robot_outcomes_1min.parquet`.
- Print per-entity summary table to stdout.

Required CLI signature using `click`:

```
@click.command()
@click.option("--iterations-dir", type=click.Path(exists=True), default="data/iterations")
@click.option("--baseline", type=click.Path(exists=True), default="data/human_surgeon_baseline.csv")
@click.option("--parent-comparison", type=click.Path(), default="../glioblastoma-1hr-trial/results/comparison.json")
@click.option("--out", type=click.Path(), default="data/robot_outcomes_1min.parquet")
@click.option("--aggregate-iterations", is_flag=True)
def cli(iterations_dir: str, baseline: str, parent_comparison: str, out: str, aggregate_iterations: bool) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean.

## File 6: src/llm/compare_agent_1min.py

Python 3.10 module implementing the on-prem LLM comparison agent for the 1-minute scenario. Required responsibilities:

- Read `data/robot_outcomes_1min.parquet`.
- Read the v3.9.1 prompt template from `prompts/comparison_prompt_1min.md`.
- Run a multi-round tournament (default size 4) of pairwise comparisons across this project's 1-minute robot, the parent v3.9.0 1-hour robot, and the manual human baseline.
- Call the configured LLM backend (Anthropic API by default with the `claude-opus-4-7` model, Ollama optional).
- Write `results/comparison.json` with structured per-round results.
- Write `results/comparison_report.md` with the narrative findings, including a structural-vs-fair-comparison call-out for the time dimension.
- Render `results/comparison_report.pdf` via pandoc.
- Render `viz/metrics_dashboard.html` via Plotly.
- Render `viz/metrics_summary.png` via matplotlib.
- Render `viz/per_arm_contribution.png` via matplotlib.
- Append to `results/tournament_log.jsonl`.

Required CLI signature using `click`:

```
@click.command()
@click.option("--outcomes", type=click.Path(exists=True), default="data/robot_outcomes_1min.parquet")
@click.option("--prompt", type=click.Path(exists=True), default="prompts/comparison_prompt_1min.md")
@click.option("--backend", type=click.Choice(["anthropic", "ollama"]), default="anthropic")
@click.option("--model", type=str, default="claude-opus-4-7")
@click.option("--tournament-size", type=int, default=4)
@click.option("--results-dir", type=click.Path(), default="results")
def cli(outcomes: str, prompt: str, backend: str, model: str, tournament_size: int, results_dir: str) -> None:
    ...
```

The script must read the API key from `os.environ["ANTHROPIC_API_KEY"]` and must never write the key to any committed file.

## File 7: prompts/comparison_prompt_1min.md

Versioned LLM prompt template for the 1-minute variant. Required sections:

1. Role: senior physical AI oncology trial reviewer with explicit awareness that a 1-minute glioblastoma resection is a hypothetical 2030 capability.
2. Context: 1-minute glioblastoma resection comparison across this project's 1-minute robot, the parent v3.9.0 1-hour robot, and manual human baselines.
3. Inputs: per-round pair of metric records.
4. Comparison weights (frozen at v3.9.1 and matching v3.9.0): Quality 0.40, Time 0.25, Cost 0.20, Safety 0.10, Patient experience 0.05.
5. Per-arm analysis section: the prompt asks the LLM to comment on whether the 4 arms are well balanced (each contributing within 30 percent of the others' tissue removal volume) or whether one arm is overworked.
6. Output schema: JSON object with `winner_entity_id`, `confidence`, `rationale_short`, `rationale_long`, `quality_delta`, `time_delta`, `cost_delta`, `safety_delta`, `patient_experience_delta`, `per_arm_balance_comment`.
7. Tone and length constraints.
8. Versioning: header `# Comparison Prompt v3.9.1`. The v3.9.1 prompt is immutable after the snapshot.

## File 8: results/comparison.json

Structured machine-readable results produced by the comparison agent. Required keys:

```
{
  "release_version": "v3.9.1",
  "tournament_size": 4,
  "round_count": 6,
  "weights": {"quality": 0.40, "time": 0.25, "cost": 0.20, "safety": 0.10, "patient_experience": 0.05},
  "leaderboard": [
    {"entity_id": "this_project_v3_9_1_1min", "skill_mu": 678.4, "skill_sigma": 92.1, "rank": 1},
    {"entity_id": "this_project_v3_9_0_1hr", "skill_mu": 642.1, "skill_sigma": 87.3, "rank": 2}
  ],
  "rounds": [],
  "per_arm_summary": {
    "arm_1_hyb_resection_mm3_mean": 32400,
    "arm_2_bipolar_coagulation_seconds_mean": 47.2,
    "arm_3_suction_ml_mean": 28.4,
    "arm_4_imaging_frames_mean": 4280
  },
  "structural_caveat_time_dimension": "The 1-minute scenario trivially beats the 1-hour scenario on the time dimension; this advantage is structural and not a fair pairwise comparison.",
  "generated_at": "2026-05-10T18:00:00Z"
}
```

## File 9: results/comparison_report.md

Narrative findings with embedded tables. Required sections: executive summary, leaderboard, per-dimension breakdown including the structural-vs-fair-comparison call-out, per-arm contribution analysis, statistical confidence, limitations including the lack of a published 1-minute manual baseline, methodology pointer to `docs/comparison_methodology.md`, citation block.

## File 10: results/comparison_report.pdf

Generated from `results/comparison_report.md` via pandoc with xelatex backend. The PDF must remain under the 5 MB committed Parquet cap (PDFs are not Parquet but the variant cap covers all binary committed files).

## File 11: viz/metrics_dashboard.html

Interactive Plotly dashboard. Required panels:

1. Leaderboard bar chart with skill mu and sigma error bars.
2. Per-dimension violin plot (Quality, Time, Cost, Safety, Patient Experience).
3. Per-iteration composite score box plot per entity.
4. Per-arm tissue removed bar chart.
5. Cumulative force violation rate scatter (this project 1-minute vs this project 1-hour vs manual).
6. Resection completeness vs total seconds scatter (with the structural caveat annotated).

The HTML file is self-contained (Plotly bundled). Approximate size: 4 MB.

## File 12: viz/metrics_summary.png

Static matplotlib summary chart for inclusion in the PDF report. Single figure with 2 by 2 subplots: leaderboard bar chart, composite score box plot, cumulative force violation rate, resection completeness vs total seconds. 1920 by 1080 pixels.

## File 13: viz/per_arm_contribution.png

Static matplotlib chart specific to the 4-arm topology. Single figure with 4 panels (one per arm) showing the per-iteration distribution of per-arm tissue removed (arm 1), per-arm coagulation seconds (arm 2), per-arm suction volume (arm 3), and per-arm imaging frames (arm 4). 1920 by 1080 pixels.

## v3.9.1 Release Snapshot

After Files 1 through 13 are committed and after the Zenodo deposition completes, the future session creates `competitions/glioblastoma-1min-trial/releases/v3.9.1/` and writes:

- `manifest.json`: SHA-256 hashes of every file under `competitions/glioblastoma-1min-trial/`.
- `metrics.json`: copy of the v3.9.1 leaderboard from `results/comparison.json`.
- `iterations_index.jsonl`: copy of `data/iterations/index.jsonl`.
- `sample_seeds.txt`: the 16 seeds, one per line.
- `zenodo_doi.txt`: the Zenodo DOI for the v3.9.1 L0 raw archive plus the per-iteration record IDs.

The snapshot is immutable after Commit 5.

## Zenodo Pointer Patching

After the Zenodo deposition completes, the future session must patch the placeholder values in every Zenodo pointer JSON file authored by Commits 2, 3, and 4. The patching script lives in `src/llm/compare_agent_1min.py` (or in a small helper at `src/zenodo/patch_pointers.py` if the future session prefers separation):

- `data/sensor_l0_raw_4arm.zenodo_pointer.json` from Commit 2.
- `data/xyz_trace_4arm.zenodo_pointer.json` from Commit 3.
- `data/iterations/run_NNNNN_L0_raw.zenodo_pointer.json` from Commit 4 (16 files).

Each patch updates the `zenodo_doi`, `zenodo_record_id`, and `sha256` fields with the real values returned by the Zenodo API.

## Validation After Commit 5

- `python -m src.metrics.compute_1min --aggregate-iterations` produces `data/robot_outcomes_1min.parquet` with the expected row count (16 robot 1-minute rows plus 16 robot 1-hour rows from the parent plus 30 baseline rows equals 62 rows).
- `python -m src.llm.compare_agent_1min --tournament-size 4` produces all four results files plus the three viz files.
- `pandoc` builds `results/comparison_report.pdf` cleanly under 5 MB.
- The Plotly dashboard opens in a modern browser without errors and is under 5 MB.
- The release snapshot directory contains the five required files.
- Every Zenodo pointer JSON has had its placeholder values patched with real DOI and SHA-256 values.
- All committed files are under 10 MB.
- `ruff format --check .` passes.
- `ruff check .` passes.

## Source Files Cited

- `competitions/instructions/competition_protocol.md`. Source for the five comparison dimensions, the composite score weights, the Gaussian skill rating model with mu_0 = 600 and sigma_0 = 200, and the on-premise LLM constraint.
- `competitions/instructions/one_minute_variant/glioblastoma_context_1min.md`. Source for the patient and procedure boundaries that the metric scoring respects.
- `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md`. Source for the per-arm safety limits whose violation count feeds the safety score.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the cumulative 12 N force limit whose violation count is the primary new safety signal.
- `competitions/instructions/one_minute_variant/zenodo_archive_protocol.md`. Source for the Zenodo deposition layout that the snapshot's `zenodo_doi.txt` references.
- `competitions/instructions/file_format_conventions.md`. Source for the PNG, HTML, and PDF size budgets.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff rules that File 5 and File 6 must satisfy.
- `competitions/instructions/commit_05_comparison_competition.md`. Source for the parent v3.9.0 12-file Commit 5 structure that this 1-minute variant extends to 13 files plus the Zenodo patching step.
- `competitions/inputs/site-1/chunk_1_site_text.md`. Source for the Gaussian skill rating model and the validation episode pattern.
- `competitions/inputs/paper-a/chunk_03_experiments.md`. Source for the multi-round tournament structure that the v3.9.1 default size of 4 inherits.
- `national-platform/usl_standard/`. Source for the USL scoring contribution to the quality score.
- `new-trial/psl_framework.md`. Source for the PSL Omniscient and Omnipresent dimensions.
- `patient-journey/master_journey.py`. Source for the per-stage outcome aggregation pattern.
