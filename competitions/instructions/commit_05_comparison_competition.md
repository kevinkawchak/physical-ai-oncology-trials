# Commit 5: Comparison and Competition

This file specifies the twelve files the future Claude Code Opus 4.7 1M Max session must author in its fifth commit. The session must author exactly the files listed and must not author additional files in this commit.

## Goal

Define quality, time, and cost metrics. Author the metric schema, the human surgeon baseline dataset, the aggregated robot outcomes Parquet, the metric computation script, the on-prem LLM comparison agent, the versioned LLM prompt, the structured comparison results, the markdown narrative report, the formal PDF report, the interactive HTML dashboard, and the static summary chart. Snapshot the v3.9.0 release at the end of the commit.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1hr-trial/docs/comparison_methodology.md` | Markdown | Hand-authored | 26 KB |
| 2 | `competitions/glioblastoma-1hr-trial/schemas/metrics.schema.json` | JSON Schema 2020-12 | Hand-authored | 9 KB |
| 3 | `competitions/glioblastoma-1hr-trial/data/human_surgeon_baseline.csv` | CSV | Hand-authored | 14 KB |
| 4 | `competitions/glioblastoma-1hr-trial/data/robot_outcomes.parquet` | Parquet | Script-generated | 50 MB |
| 5 | `competitions/glioblastoma-1hr-trial/src/metrics/compute.py` | Python 3.10 | Hand-authored | 18 KB |
| 6 | `competitions/glioblastoma-1hr-trial/src/llm/compare_agent.py` | Python 3.10 | Hand-authored | 22 KB |
| 7 | `competitions/glioblastoma-1hr-trial/prompts/comparison_prompt.md` | Markdown | Hand-authored | 12 KB |
| 8 | `competitions/glioblastoma-1hr-trial/results/comparison.json` | JSON | Script-generated | 60 KB |
| 9 | `competitions/glioblastoma-1hr-trial/results/comparison_report.md` | Markdown | Script-generated | 30 KB |
| 10 | `competitions/glioblastoma-1hr-trial/results/comparison_report.pdf` | PDF | Script-generated via pandoc | 5 MB |
| 11 | `competitions/glioblastoma-1hr-trial/viz/metrics_dashboard.html` | HTML (Plotly) | Script-generated | 4 MB |
| 12 | `competitions/glioblastoma-1hr-trial/viz/metrics_summary.png` | PNG | Script-generated | 400 KB |

The future session must also create the v3.9.0 release snapshot at `competitions/glioblastoma-1hr-trial/releases/v3.9.0/` containing the four files listed in `competitions/instructions/competition_protocol.md` (manifest.json, metrics.json, iterations_index.jsonl, sample_seeds.txt). The snapshot directory is created at the end of Commit 5.

## File 1: docs/comparison_methodology.md

Required sections:

1. Comparison goal: rank this project's robot against prior version snapshots, competitor robots, and hybrid human-robot teams across quality, time, cost, safety, and patient experience.
2. Metric definitions reproduced from `competitions/instructions/competition_protocol.md`:
   - Quality (40 percent weight): composite of resection completeness percentage, eloquent cortex preservation score, and PSL Omniscient and Omnipresent dimensions.
   - Time (25 percent weight): total seconds from procedure start to final E-stop or COMPLETE state.
   - Cost (20 percent weight): consumables, robot depreciation, OR time, anesthesia time.
   - Safety (10 percent weight): inverse of (force violation count plus E-stop count plus adverse event count).
   - Patient experience (5 percent weight): predicted post-operative KPS at 30 days from a fixed regression model.
3. Composite score formula and weights, frozen for v3.9.0.
4. Skill rating model (Gaussian N(mu, sigma squared)) with mu_0 = 600 and sigma_0 = 200, mirroring the Orbit Wars Kaggle competition.
5. Multi-round tournament structure (default size 8), mirroring the CodeClash paper.
6. Statistical methods: bootstrap 95 percent confidence intervals across 64 iterations, Mann-Whitney U for pairwise comparisons.
7. Cross-references to `prompts/comparison_prompt.md` and `src/metrics/compute.py`.

## File 2: schemas/metrics.schema.json

JSON Schema 2020-12 with the following structure:

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kevinkawchak.github.io/physical-ai-oncology-trials/v3.9.0/metrics.schema.json",
  "title": "GBM 1hr Metrics Record",
  "type": "object",
  "required": ["entity_id", "entity_kind", "iteration_id", "release_version",
               "quality_score", "total_seconds", "cost_usd", "safety_score",
               "patient_experience_score", "composite_score",
               "human_intervention_seconds", "force_violation_count",
               "estop_count", "ae_count", "resection_completeness_pct",
               "eloquent_preservation_score", "predicted_kps_day_30",
               "skill_mu", "skill_sigma"],
  "properties": {
    "entity_id": {"type": "string"},
    "entity_kind": {"type": "string", "enum": ["this_project", "prior_version", "competitor_robot", "hybrid_team", "manual_human"]},
    "iteration_id": {"type": "string", "pattern": "^run_[0-9]{5}$"},
    "release_version": {"type": "string", "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$"},
    "quality_score": {"type": "number", "minimum": 0, "maximum": 100},
    "total_seconds": {"type": "number", "minimum": 0, "maximum": 14400},
    "cost_usd": {"type": "number", "minimum": 0},
    "safety_score": {"type": "number", "minimum": 0, "maximum": 100},
    "patient_experience_score": {"type": "number", "minimum": 0, "maximum": 100},
    "composite_score": {"type": "number", "minimum": 0, "maximum": 100},
    "human_intervention_seconds": {"type": "integer", "minimum": 0},
    "force_violation_count": {"type": "integer", "minimum": 0},
    "estop_count": {"type": "integer", "minimum": 0},
    "ae_count": {"type": "integer", "minimum": 0},
    "resection_completeness_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "eloquent_preservation_score": {"type": "number", "minimum": 0, "maximum": 100},
    "predicted_kps_day_30": {"type": "number", "minimum": 0, "maximum": 100},
    "skill_mu": {"type": "number"},
    "skill_sigma": {"type": "number", "minimum": 0}
  },
  "additionalProperties": false
}
```

## File 3: data/human_surgeon_baseline.csv

Hand-authored reference dataset of human surgeon outcomes for glioblastoma resection, drawn from published literature. Columns:

```
entity_id,entity_kind,iteration_id,release_version,quality_score,total_seconds,cost_usd,safety_score,patient_experience_score,composite_score,human_intervention_seconds,force_violation_count,estop_count,ae_count,resection_completeness_pct,eloquent_preservation_score,predicted_kps_day_30,skill_mu,skill_sigma,source_doi
```

Required rows: 30 baseline records, 5 records each from 6 published surgical centers. Each row's `entity_kind` is `manual_human`. The `human_intervention_seconds` value equals `total_seconds` for manual procedures. The `source_doi` column points to the published outcome study.

The future session must use real surgical outcome ranges (resection completeness 60 to 95 percent, total surgical time 4,500 to 14,400 seconds, predicted KPS day 30 of 60 to 90). The CSV is hand-authored from literature and is not invented at random.

## File 4: data/robot_outcomes.parquet

The future session must produce this file by running `src/metrics/compute.py --aggregate-iterations`. The file aggregates the 64 iteration Parquet files from `data/iterations/` into per-iteration metric records conforming to `metrics.schema.json`. Each iteration produces one row.

## File 5: src/metrics/compute.py

Python 3.10 module with the following responsibilities:

- Read the 64 iteration Parquet files.
- Compute per-iteration quality, time, cost, safety, and patient experience scores.
- Compute the composite score using the v3.9.0 weights.
- Compute the skill mu and sigma using TrueSkill-style update.
- Read `data/human_surgeon_baseline.csv` and concatenate.
- Write `data/robot_outcomes.parquet`.
- Print per-entity summary table to stdout.

Required CLI signature using `click`:

```
@click.command()
@click.option("--iterations-dir", type=click.Path(exists=True), default="data/iterations")
@click.option("--baseline", type=click.Path(exists=True), default="data/human_surgeon_baseline.csv")
@click.option("--out", type=click.Path(), default="data/robot_outcomes.parquet")
@click.option("--aggregate-iterations", is_flag=True)
def cli(iterations_dir: str, baseline: str, out: str, aggregate_iterations: bool) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean.

## File 6: src/llm/compare_agent.py

Python 3.10 module implementing the on-prem LLM comparison agent. Required responsibilities:

- Read `data/robot_outcomes.parquet`.
- Read the v3.9.0 prompt template from `prompts/comparison_prompt.md`.
- Run a multi-round tournament (default size 8) of pairwise comparisons across entity categories.
- Call the configured LLM backend (Anthropic API by default, Ollama optional) to judge each round.
- Write `results/comparison.json` with structured per-round results.
- Write `results/comparison_report.md` with the narrative findings.
- Render `results/comparison_report.pdf` via pandoc.
- Render `viz/metrics_dashboard.html` via Plotly.
- Render `viz/metrics_summary.png` via matplotlib.
- Append to `results/tournament_log.jsonl`.

Required CLI signature using `click`:

```
@click.command()
@click.option("--outcomes", type=click.Path(exists=True), default="data/robot_outcomes.parquet")
@click.option("--prompt", type=click.Path(exists=True), default="prompts/comparison_prompt.md")
@click.option("--backend", type=click.Choice(["anthropic", "ollama"]), default="anthropic")
@click.option("--model", type=str, default="claude-opus-4-7")
@click.option("--tournament-size", type=int, default=8)
@click.option("--results-dir", type=click.Path(), default="results")
def cli(outcomes: str, prompt: str, backend: str, model: str, tournament_size: int, results_dir: str) -> None:
    ...
```

The script must read the API key from `os.environ["ANTHROPIC_API_KEY"]` and must never write the key to any committed file.

## File 7: prompts/comparison_prompt.md

Versioned LLM prompt template. Required sections:

1. Role: senior physical AI oncology trial reviewer.
2. Context: glioblastoma resection comparison across this project's robot, prior version snapshots, competitor robots, and hybrid teams.
3. Inputs: per-round pair of metric records.
4. Comparison weights (frozen at v3.9.0): Quality 0.40, Time 0.25, Cost 0.20, Safety 0.10, Patient experience 0.05.
5. Output schema: JSON object with `winner_entity_id`, `confidence`, `rationale_short`, `rationale_long`, `quality_delta`, `time_delta`, `cost_delta`, `safety_delta`, `patient_experience_delta`.
6. Tone and length constraints.
7. Versioning: header `# Comparison Prompt v3.9.0`. Future v3.10.0 prompt lives in a sibling file with explicit version bump; the v3.9.0 file is immutable after the snapshot.

## File 8: results/comparison.json

Structured machine-readable results produced by the comparison agent. Required keys:

```
{
  "release_version": "v3.9.0",
  "tournament_size": 8,
  "round_count": 32,
  "weights": {"quality": 0.40, "time": 0.25, "cost": 0.20, "safety": 0.10, "patient_experience": 0.05},
  "leaderboard": [
    {"entity_id": "this_project_v3_9_0", "skill_mu": 642.1, "skill_sigma": 87.3, "rank": 1},
    {"entity_id": "competitor_NeuroMate", "skill_mu": 614.5, "skill_sigma": 91.1, "rank": 2}
  ],
  "rounds": [...],
  "generated_at": "2026-05-09T18:00:00Z"
}
```

## File 9: results/comparison_report.md

Narrative findings with embedded tables. Required sections: executive summary, leaderboard, per-dimension breakdown, statistical confidence, limitations, methodology pointer to `docs/comparison_methodology.md`, citation block.

## File 10: results/comparison_report.pdf

Generated from `results/comparison_report.md` via:

```
pandoc results/comparison_report.md -o results/comparison_report.pdf \
  --pdf-engine=xelatex \
  --metadata title="GBM 1hr Trial v3.9.0 Comparison Report" \
  --metadata author="Kevin Kawchak"
```

The future session must verify the PDF builds cleanly. Approximate size: 5 MB.

## File 11: viz/metrics_dashboard.html

Interactive Plotly dashboard. Required panels:

1. Leaderboard bar chart with skill mu and sigma error bars.
2. Per-dimension violin plot (Quality, Time, Cost, Safety, Patient Experience).
3. Per-iteration composite score box plot per entity.
4. Force violation rate scatter (this project versus competitors).
5. Resection completeness versus total seconds scatter.

The HTML file is self-contained (Plotly bundled). Approximate size: 4 MB.

## File 12: viz/metrics_summary.png

Static matplotlib summary chart for inclusion in the PDF report. Single figure with 2 by 2 subplots: leaderboard bar chart, composite score box plot, force violation rate, resection completeness versus total seconds. 1920 by 1080 pixels.

## v3.9.0 Release Snapshot

After Files 1 through 12 are committed, the future session creates `competitions/glioblastoma-1hr-trial/releases/v3.9.0/` and writes:

- `manifest.json`: SHA-256 hashes of every file under `competitions/glioblastoma-1hr-trial/`.
- `metrics.json`: copy of the v3.9.0 leaderboard from `results/comparison.json`.
- `iterations_index.jsonl`: copy of `data/iterations/index.jsonl`.
- `sample_seeds.txt`: the 64 seeds, one per line.

The snapshot is immutable after Commit 5.

## Validation After Commit 5

- `python -m src.metrics.compute --aggregate-iterations` produces `data/robot_outcomes.parquet` with the expected row count (64 robot rows plus 30 baseline rows equals 94 rows).
- `python -m src.llm.compare_agent --tournament-size 8` produces all four results files plus the two viz files.
- `pandoc` builds `results/comparison_report.pdf` cleanly.
- The Plotly dashboard opens in a modern browser without errors.
- The release snapshot directory contains the four required files.
- `ruff format --check .` passes.
- `ruff check .` passes.

## Source Files Cited

- `competitions/instructions/competition_protocol.md`. Source for the five comparison dimensions, the composite score weights, the Gaussian skill rating model with mu_0 = 600 and sigma_0 = 200, the multi-round tournament structure, and the on-premise LLM constraint.
- `competitions/instructions/glioblastoma_context.md`. Source for the patient and procedure boundaries that the metric scoring respects.
- `competitions/instructions/robot_specification.md`. Source for the safety limits whose violation count feeds the safety score.
- `competitions/instructions/file_format_conventions.md`. Source for the PNG, HTML, and PDF size budgets.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff rules that File 5 and File 6 must satisfy.
- `competitions/inputs/site-1/chunk_1_site_text.md`. Source for the Gaussian skill rating model and the validation episode pattern.
- `competitions/inputs/paper-a/chunk_03_experiments.md`. Source for the multi-round tournament structure that the v3.9.0 default size of 8 inherits.
- `competitions/inputs/paper-a/chunk_04_results.md`. Source for the Elo rating pattern used by the leaderboard.
- `competitions/inputs/paper-b/chunk_05_results_round4_code.md`. Source for the round-by-round LLM judging pattern used by `src/llm/compare_agent.py`.
- `new-trial/national-24-7-trial/paper/full-paper/final-paper/main.tex`. Source for the LaTeX paper structure that informs the PDF output via pandoc.
- `new-trial/national-24-7-trial/paper/full-paper/final-paper/sections/methods.tex`. Source for the methods narrative pattern that `comparison_report.md` mirrors.
- `national-platform/usl_standard/`. Source for the Unification Standard Level scoring contribution to the quality score.
- `new-trial/psl_framework.md`. Source for the Physical AI Standard Level Omniscient and Omnipresent dimensions that contribute to the quality score.
- `patient-journey/master_journey.py`. Source for the per-stage outcome aggregation pattern that informs the per-iteration metric computation.
