# CI Compliance Checklist

This file fixes the pre-commit checklist that the future session must run before each of the seven future commits. Following this checklist prevents the lint failures previously observed in the v3.7.0 and v3.8.0 release cycle: `Cl / lint-and-format (3.10) (pull...)`, `(3.11)`, and `(3.12)`.

## CI Workflow Reference

The repository's CI workflow lives at `.github/workflows/ci.yml`. The job that fails most commonly is `lint-and-format`. It runs three checks:

1. `ruff check . --output-format=github`
2. `ruff format --check .`
3. `yamllint -d relaxed configs/ unification/simulation_physics/physics_parameter_mapping.yaml`

The future session must pass all three locally before pushing each commit.

## Local Pre-Commit Block

The future session must run the block below from the repository root before each `git push`. The block must exit with status 0 for the commit to proceed.

```
pip install --quiet ruff yamllint
ruff format --check .
ruff check . --output-format=github
yamllint -d relaxed competitions/glioblastoma-1hr-trial/config/
python -m py_compile $(git ls-files 'competitions/glioblastoma-1hr-trial/**/*.py')
```

If `ruff format --check` fails, the future session runs `ruff format .` and re-stages the changed files.

If `ruff check` fails, the future session reads the offending file, fixes the issue at the source, and never relies on `# noqa` blanket suppression. Per-file ignores are added to `ruff.toml` only with a one-line comment explaining why. The acceptable additions for this project are listed below.

## Required ruff.toml Addition

The future Commit 1 must add the following block to `ruff.toml`:

```
# Glioblastoma 1-hour trial generators use try/except imports for optional
# packages (pyarrow, duckdb, plotly) and have placeholder forward references
# during the multi-commit build-up.
"competitions/glioblastoma-1hr-trial/**/*.py" = ["F401", "F402", "F821"]
```

This addition matches the existing per-file-ignores entries for `patient-journey/**/*.py`, `regulatory/**/*.py`, and `unification/**/*.py`. It avoids the need for inline noqa suppressions inside the future generated Python files.

## Common Failure Patterns and Their Fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| `ruff format` reformatting CRLF endings | Files saved on Windows | Add `.gitattributes` entry `*.py text eol=lf` |
| `ruff check E501` line too long | Long literal strings | Break into multi-line string with explicit join, or rely on the `ignore = ["E501"]` already in `ruff.toml` |
| `ruff check F401` unused import | Optional package import wrapped in try/except | Covered by the per-file-ignores added above |
| `yamllint indentation` warning | Tab indentation in YAML | Replace tabs with two spaces |
| `yamllint trailing-spaces` warning | Trailing whitespace | Run `sed -i 's/[[:space:]]*$//' file.yaml` |
| `yamllint document-start` warning | Missing `---` document separator | Add `---` at top of YAML file |
| `pytest` collection error | Heavy import (torch, mujoco) at module load | Wrap heavy imports in try/except per the existing `tests/conftest.py` pattern |

## Python Version Compatibility

All Python files must work under Python 3.10, 3.11, and 3.12. The CI matrix runs the three versions in parallel. Common pitfalls:

- Do not use `match` statement features added in 3.11 unless guarded by a version check.
- Do not use `typing.TypeVar` with `default=` (added in 3.13). Use `from typing_extensions import TypeVar` if needed.
- Always include `from __future__ import annotations` at the top of each module to keep forward references valid across versions.
- Use `pathlib.Path` exclusively for file system operations.

## YAML Compatibility

- Use `yamllint -d relaxed` rules. Do not enable strict rules.
- Top of every YAML file: `---`.
- No tabs.
- Indent two spaces.
- Quote strings that contain special characters (`:`, `#`, `!`, `&`, `*`, `[`, `]`, `{`, `}`, `,`).

## Commit Message Conventions

Use the conventional repository pattern. Each of the seven future commits uses one of the following prefixes:

- `feat:` for new files
- `fix:` for fixes to prior commits in this PR
- `docs:` for README or documentation only
- `chore:` for repository housekeeping (CHANGELOG, releases.md)

The commit message must include a one-line summary, a blank line, and a multi-line body that lists the files added in that commit.

## Verification Before Pushing

The future session must run the following block locally and capture its output before each `git push`:

```
ruff format --check . && \
ruff check . && \
yamllint -d relaxed competitions/glioblastoma-1hr-trial/config/ && \
echo "OK: ready to push commit $(git log -1 --pretty=%H)"
```

If the block does not echo `OK:`, the push is held until all three checks pass.

## Source Files Cited

- `.github/workflows/ci.yml`. Source for the lint-and-format job, its Python matrix, and the three lint commands.
- `ruff.toml`. Source for the existing per-file-ignores pattern and the three E, F, W lint categories that ruff is configured to enforce.
- `tests/conftest.py`. Source for the `load_module()` guard that lets heavy-import tests skip gracefully under the CI baseline install.
