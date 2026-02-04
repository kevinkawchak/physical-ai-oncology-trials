# Contributing

Contributions are welcome from engineers working on physical AI systems (robotics, ML, integration, and validation) for oncology clinical trial settings.

## What We Accept

| Contribution Type | Examples |
|-------------------|----------|
| **Integration guides** | New framework integration, updated version support |
| **Pipeline code** | Training scripts, conversion tools, validation utilities |
| **Benchmark results** | Published or reproducible performance data with citations |
| **Regulatory tooling** | Compliance checkers, documentation generators |
| **Privacy tooling** | De-identification methods, access control extensions |
| **Bug fixes** | Corrections to existing code, configs, or documentation |

## Requirements for All Contributions

1. **Recency**: Referenced frameworks and tools must have been updated within the last 3 months.
2. **Oncology relevance**: Changes must have a clear application to oncology clinical trials.
3. **Reproducibility**: Code must include configurations and instructions sufficient for reproduction.
4. **Cross-platform awareness**: Where applicable, consider compatibility with the `unification/` framework (Isaac, MuJoCo, Gazebo, PyBullet).

## Development Workflow

### 1. Fork and Branch

```bash
git clone https://github.com/<your-fork>/physical-ai-oncology-trials.git
cd physical-ai-oncology-trials
git checkout -b your-branch-name
```

### 2. Install Development Tools

```bash
pip install ruff yamllint
```

### 3. Make Changes

- Follow existing code structure and naming conventions.
- Pin dependency versions in `requirements.txt`.
- Do not commit patient data, PHI, credentials, or API keys.

### 4. Lint and Format

```bash
ruff check .
ruff format .
yamllint -d relaxed configs/
```

### 5. Submit a Pull Request

- Fill out the PR template completely, including the safety/compliance checklist.
- Reference any related issues.

## Benchmark and Results Data

If you are adding or updating performance numbers in any `results.md` file:

- **Published results**: Include a citation (author, venue, year) in the table or a footnote.
- **Reproduced results**: Note the hardware, framework version, and random seed used.
- **Projected or illustrative data**: Clearly label as *Illustrative* in the table. Do not present projected figures as measured results.

## Code Style

- Python: formatted with `ruff format`, linted with `ruff check`.
- YAML: validated with `yamllint -d relaxed`.
- Markdown: standard GitHub-flavored markdown.

## Safety and Compliance

- Any code that automates clinical workflows (CRF auto-fill, adverse event reporting, medication handling) **must** document the required human oversight steps.
- Do not introduce changes that bypass safety gates or remove human-in-the-loop requirements.
- Regulatory and privacy tools are reference implementations. Contributors should not represent them as validated medical device software.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
