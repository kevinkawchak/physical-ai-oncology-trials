# Commit 6: Error Fixes

This file specifies the error-review-and-patch pass that the future Claude Code Opus 4.7 1M Max session must run as its sixth commit. The session must touch only files that have a documented error. The session must not introduce new files in this commit; new files belong in Commits 1 through 5.

## Goal

Catch and patch the residual errors that survive Commits 1 through 5. Common error categories include schema drift across Commits 2, 3, and 5; cross-file path drift after the SVG-to-ASCII replacement; CI lint failures triggered by the Python files added in Commits 2, 3, 4, and 5; minor inconsistencies between hand-authored documents and script-generated outputs.

## Pre-Commit Error Scan

Before authoring any patch, the future session must run the seven checks below in order and record the output to `logs/error_scan.txt`. The checks are deliberately small so the LLM context can absorb each output independently.

| Check | Command | Expected outcome |
|-------|---------|-------------------|
| 1 ruff format | `ruff format --check competitions/glioblastoma-1hr-trial/` | exit 0 |
| 2 ruff lint | `ruff check competitions/glioblastoma-1hr-trial/` | exit 0 |
| 3 yamllint | `yamllint -d relaxed competitions/glioblastoma-1hr-trial/config/` | exit 0 |
| 4 schema cross-validate | `python -m src.sensors.ingest --validate data/sensor_sample.jsonl` | exit 0, no schema errors |
| 5 path cross-reference | `find competitions/glioblastoma-1hr-trial -name '*.svg'` | only allowed aggregate SVG present |
| 6 generator determinism | run canonical Parquet emission twice, compare SHA-256 | hashes match |
| 7 PR-wide markdown lint | `python -m markdown_it_py competitions/glioblastoma-1hr-trial/**/*.md` | parses without error |

If any check fails, the future session patches the offending file, re-runs the failing check, and proceeds only when the check passes.

## Eight Common Errors and Their Patches

### Error 1: Mismatched channel count

The sensor channel inventory in `docs/sensor_spec.md` and the property list in `schemas/sensor_record.schema.json` may drift. Patch by re-counting both and bringing them to exactly 50 channels plus `tick_ms`, `meta_seed`, and `meta_iteration_id`.

### Error 2: SVG file rejected at lint time

If any `.svg` file larger than 100 KB lands under `competitions/glioblastoma-1hr-trial/`, the patch deletes the file and adds an entry to `viz/.gitignore` blocking re-introduction. The aggregate `viz/xyz_path_aggregate.svg` is allowed at 1 KB; all other SVG paths must be replaced by the ASCII alternative under `viz/*.txt`.

### Error 3: ruff format E501 inside string literal

ruff's `ignore = ["E501"]` suppresses the warning. If `ruff format --check` still complains, the patch wraps the string with explicit `"\n".join([...])` to keep each fragment under 120 characters.

### Error 4: yamllint trailing whitespace

Patch by `sed -i 's/[[:space:]]*$//' competitions/glioblastoma-1hr-trial/config/*.yaml`.

### Error 5: yamllint missing document-start

Patch by adding `---` as the first line of every `.yaml` file under `competitions/glioblastoma-1hr-trial/config/`.

### Error 6: Cargo.toml version mismatch

The Rust runner's `Cargo.toml` package version must match `pyproject.toml` version (3.9.0). Patch by updating both to the same string.

### Error 7: Forward reference inside Python file

Python files authored in early commits may reference modules that did not yet exist. The repository-level `ruff.toml` per-file-ignores entry covers F821 (undefined name) for `competitions/glioblastoma-1hr-trial/**/*.py`. If `ruff check` still reports F821, the patch updates `ruff.toml` to add the per-file-ignores entry described in `competitions/instructions/ci_compliance_checklist.md`.

### Error 8: Cross-document path drift

If `docs/architecture.md` (Commit 1) references `viz/xyz_path.svg` but Commit 3 produced `viz/xyz_path.txt`, the patch updates the architecture document to match the actual file. The future session must run `grep -r 'xyz_path.svg' competitions/glioblastoma-1hr-trial/` and patch every match.

## Patch Authoring Rules

- One patch commit modifies only the files that contain a documented error.
- Each patched file gets a one-line comment near the change citing which error category was fixed.
- The commit message lists every patched file under the body's "Fixed:" section.
- The commit message does not introduce new content; new content waits for the next release cycle.

## Validation After Commit 6

- All seven checks above pass.
- `git diff origin/main...HEAD --stat` shows changes only to files that already existed in Commits 1 through 5.
- The CI workflow runs to completion green for Python 3.10, 3.11, and 3.12.

## Source Files Cited

- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint commands and their expected exit codes.
- `competitions/instructions/file_format_conventions.md`. Source for the SVG-vs-ASCII decision rule that informs Error 2 above.
- `competitions/instructions/chunking_strategy.md`. Source for the per-commit file count budget that limits the size of the patch commit.
- `competitions/instructions/pr_workflow.md`. Source for the rule that Commit 6 is a `fix:` commit and modifies only files already touched by Commits 1 through 5.
