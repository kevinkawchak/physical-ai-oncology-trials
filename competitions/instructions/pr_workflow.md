# Seven-Commit Single-PR Workflow

This file fixes how the future Claude Code Opus 4.7 1M Max session structures its seven commits inside a single pull request. The pattern matches the workflow used to produce this v3.9.0 instruction set and matches the workflow used by prior releases v3.7.0 and v3.8.0.

## Branch Naming

The future session creates a branch named `claude/glioblastoma-1hr-trial-<short_id>` where `<short_id>` is a six-character random alphanumeric string. The session pushes the branch on every commit and opens the pull request after Commit 7.

## Commit Sequence

| Commit | Title prefix | Goal | Authored files |
|--------|--------------|------|----------------|
| 1 | `feat:` | Project overview | 7 files including README and architecture |
| 2 | `feat:` | Sensor specifications | 8 files including schemas and ingest script |
| 3 | `feat:` | Sensor to xyz mapping | 9 files including schemas, mapper, control loop, sample trace |
| 4 | `feat:` | Iteration design | 9 files including iterations config, orchestrator, runner |
| 5 | `feat:` | Comparison and competition | 12 files including metrics, agent, prompt, report, dashboard |
| 6 | `fix:` | Error fixes | varies; only files touched by error review |
| 7 | `chore:` | Repository updates | 3 files: main README, CHANGELOG, releases.md |

## Per-Commit Procedure

For each commit the future session follows the four-step procedure below.

1. Read the corresponding `commit_NN_*.md` instruction file once.
2. Author the listed files. Do not author files outside the list. Do not omit files in the list.
3. Run the pre-commit checklist in `ci_compliance_checklist.md`.
4. Commit and push immediately. Do not batch multiple commits.

## Push Cadence

Push after each commit. Do not wait until all seven commits are local. The push cadence enables the user (or a reviewer) to follow progress in real time and enables the GitHub CI to surface lint failures incrementally rather than as a single multi-commit failure.

## Pull Request Lifecycle

- Branch created at the start of Commit 1.
- Pushes after each of Commits 1 through 7.
- Pull request opened after Commit 7 if and only if the user has explicitly requested it. The default behavior is to push the branch and stop.
- Pull request title: `v3.9.0: Glioblastoma 1-hour millisecond resolution simulation`.
- Pull request body uses the format defined in `commit_07_repository_updates.md`.

## Autonomy

The future session must execute Commits 1 through 7 autonomously. The user cannot interact between commits. The session must not stall, must not ask questions, and must not enter plan mode. If the session encounters an error it cannot resolve, the session must record the error in `logs/iteration_run.txt`, continue with the next commit, and surface the error in the Commit 6 error fix pass.

## Order Independence and Forward References

The early commits intentionally contain forward references to files authored by later commits. The CI per-file-ignores entry covers `F821` for forward references inside Python files. Markdown forward references resolve as soon as the later commit lands; the GitHub repository view automatically follows them.

The future session must not attempt to backfill forward references in the early commits. Backfilling triggers spurious diff churn and increases the risk of merge conflicts inside the PR.

## Snapshot at Commit 5

After Commit 5 lands, the future session takes a release snapshot of the simulation outputs into `competitions/glioblastoma-1hr-trial/releases/v3.9.0/`. The snapshot is the artifact that future v3.10.0, v3.11.0, and so on will compare against. The snapshot includes:

- `manifest.json` with SHA-256 hashes of every output file.
- `metrics.json` with the v3.9.0 quality, time, cost summary.
- `iterations_index.jsonl` with the 64 iteration metadata records.
- `sample_seeds.txt` with the 64 seeds.

The snapshot is versioned by Git and is therefore immutable after Commit 5.

## Source Files Cited

- `releases.md`. Source for the prior release notes pattern that Commit 7 mirrors at v3.9.0.
- `CHANGELOG.md`. Source for the Keep a Changelog format that Commit 7 mirrors.
- `README.md`. Source for the architecture diagram block convention that Commit 7 augments at v3.9.0.
