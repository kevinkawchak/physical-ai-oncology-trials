# Competition Protocol

This file fixes how the future simulation's outputs become competition entries that can be compared to prior versions of this project's robot, to competitor robots, and to hybrid human-robot teams. The protocol is referenced by `commit_05_comparison_competition.md` and is informed by the existing competition input papers in `competitions/inputs/`.

## Goals

- a. The state of the art millisecond resolution robot files, robot movement files, and iterations for competitions must be planned so that each set of files comprehensively works well together or as a single set, and will be correctly generated at scale by Claude Code in the future.
- b. The final set of generated files can be iterated and compared over time to earlier file sets or judged vs. competitor robots or competitor human/robot teams, separate sets or as a whole in the future.
- c. Any future learning advantages will be due to AI's computational benefits learning from and improving from past results, log files, and iteration histories. The competition does not use formal machine learning processes with parameter updates.

## Three Competitor Categories

### Category 1: Prior Versions of This Project

Each release of this repository (v3.9.0, v3.10.0, and so on) snapshots the simulation outputs in a release-tagged subdirectory under `competitions/glioblastoma-1hr-trial/releases/v3.9.0/`. The future session writes the snapshot path into `results/comparison.json` so that every later release can compute deltas against it.

Snapshot layout per release:

```
competitions/glioblastoma-1hr-trial/releases/v3.9.0/
  manifest.json          # SHA-256 hashes of all snapshot files
  metrics.json           # quality, time, cost summary for the release
  iterations_index.jsonl # 64 iteration metadata records
  sample_seeds.txt       # the 64 seeds used to generate the iterations
```

The future session must include a `compare_release.py` script that reads two release snapshots and produces a delta report in `results/release_delta_<old>_to_<new>.md`.

### Category 2: Competitor Robots

The future Commit 5 LLM agent must be able to consume competitor robot outputs that follow the same `metrics.schema.json` contract. Competitor outputs live under `competitions/glioblastoma-1hr-trial/external/<competitor_name>/`. The future session creates the directory and a `.gitkeep` placeholder.

Listed competitors that the protocol explicitly supports (each listed only by published category; the future session may add more):

- Renishaw NeuroMate stereotactic neurosurgical robot
- Brainlab Cirq robotic arm with Curve navigation
- Synaptive Medical Modus V exoscope coupled to manual surgical instrumentation
- Mazor X Stealth Edition repurposed for cranial application
- Manual surgery (no robot) by board-certified neurosurgeon

For each competitor a `metrics.json` file is sufficient input. The future LLM agent does not require the competitor's raw sensor stream; the agent only needs the per-iteration metric record.

### Category 3: Hybrid Human-Robot Teams

A hybrid team is defined by a non-zero `human_intervention_seconds` value in the metric record. The future Commit 5 metrics schema reserves an integer column for that field. The future LLM agent must report, for each comparison, the breakdown of robot-only versus human-supervised time.

## Comparison Dimensions

The future Commit 5 metric record fixes five comparison dimensions. The future LLM agent must report each dimension separately and must report a composite score equal to a fixed weighted sum.

| Dimension | Weight | Source schema field | Direction |
|-----------|--------|---------------------|-----------|
| Quality | 0.40 | `quality_score` | higher is better |
| Time | 0.25 | `total_seconds` | lower is better |
| Cost | 0.20 | `cost_usd` | lower is better |
| Safety | 0.10 | `safety_score` | higher is better |
| Patient experience | 0.05 | `patient_experience_score` | higher is better |

The composite score formula is fixed at v3.9.0 and is not subject to iteration sweep. The weights live in `prompts/comparison_prompt.md` and in `src/metrics/compute.py`.

## Skill Rating Borrowed from Orbit Wars

The future session must implement a Gaussian skill rating in `src/metrics/compute.py` with the following parameters, mirroring the `competitions/inputs/site-1/` Orbit Wars Kaggle competition:

- Initialization: mu_0 = 600, sigma_0 = 200
- Update rule: TrueSkill-style Bayesian update on each pairwise comparison episode
- Episode definition: one iteration of this project versus one iteration of the competitor's run that targets the same seed

Each release publishes a `competitions/glioblastoma-1hr-trial/leaderboard.md` file that lists, for the release version, the top 10 competitors by mu and includes 95 percent confidence intervals.

## Multi-Round Tournament Borrowed from CodeClash

The future Commit 5 LLM agent runs multi-round tournaments using the Elo rating pattern from `competitions/inputs/paper-a/`. Each round consists of:

- One iteration from this project's robot.
- One iteration from a chosen competitor.
- One pairwise comparison computed by the LLM agent and stored as a record in `results/tournament_log.jsonl`.

The future session must support tournaments of size 8 (32 rounds), 16 (120 rounds), and 64 (2,016 rounds). The default tournament size for v3.9.0 is 8.

## On-Premise LLM Constraint

Per the project thesis, the LLM agent runs on-premises. The future Commit 5 agent must support two backends:

- Default: Anthropic API with `claude-opus-4-7` model. The API key lives in `.env` and is read by `src/llm/compare_agent.py` via `os.environ`.
- Alternate: local Ollama serving an open-weight model (default `llama3.1:70b`). Selectable via `--backend ollama`.

The agent must never write the API key to any committed file.

## Competition Reproducibility

- Every iteration has a documented seed.
- Every release snapshots the full set of seeds in `sample_seeds.txt`.
- Every release records the SHA-256 hash of every output file in `manifest.json`.
- Every release records the Python, Rust, and C++ compiler versions used to build the binaries in `manifest.json`.
- Every release records the Git SHA of the commit that produced the snapshot in `manifest.json`.

## Source Files Cited

- `competitions/inputs/site-1/chunk_1_site_text.md`. Source for the Gaussian N(mu, sigma squared) skill rating model and the mu_0 = 600 initialization. The Orbit Wars competition validates the model on a 1v1 and 4-player free-for-all multi-agent competition.
- `competitions/inputs/site-1/chunk_2_tables.md`. Source for the per-submission validation episode pattern that the future tournament loop mirrors.
- `competitions/inputs/paper-a/chunk_03_experiments.md`. Source for the C(8,2) by 6 arenas by 10 tournaments by 15 rounds tournament structure that informs the v3.9.0 default tournament size of 8.
- `competitions/inputs/paper-a/chunk_04_results.md`. Source for the Elo rating pattern that the future multi-round tournament implements.
- `competitions/inputs/paper-b/chunk_05_results_round4_code.md`. Source for the round-by-round code-clash judging pattern that the future LLM agent borrows for the comparison report.
