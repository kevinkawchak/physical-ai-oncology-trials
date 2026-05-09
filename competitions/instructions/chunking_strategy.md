# Chunking Strategy

This file defines how the future Claude Code Opus 4.7 1M Max session must split the millisecond resolution simulation across multiple commits and across multiple files within each commit so that no single LLM context exceeds practical working memory and no single repository file becomes unwieldy.

## Why Chunking is Mandatory

A single sensor channel sampled at 1 kHz across 1 hour produces 3,600,000 records. With 50 channels plus a millisecond timestamp the canonical 1-hour record set is 180,000,000 numeric values plus the timestamp column. Authoring those values inline as markdown text would require approximately 1.8 GB of token output and cannot fit in any single LLM context. The chunking strategy below is therefore not optional.

## Three-Layer Strategy

The future session uses three complementary chunking layers.

### Layer 1: Generators Not Data

The future session must author small deterministic generator scripts and small human-review samples. The future session must not paste 3.6 million records into any markdown or JSON file. This pattern matches `patient-journey/stage_05_surgery.py` which describes the surgical trajectory in code rather than as a data dump and matches `new-trial/national-24-7-trial/hour-00/hour_00_simulation.md` which records minute-resolution narrative rather than millisecond logs.

For each canonical 1-hour data product the future session authors:

- One generator script in `src/` of approximately 200 to 400 lines.
- One human-review sample of 1,000 records (the first second at 1 kHz) in CSV at approximately 100 KB.
- One streaming append-only sample of 1,000 records in JSONL at approximately 250 KB.
- One full Parquet file produced by running the generator script. The Parquet file is committed but is not authored character by character.

Repository size budget per generator: the generator script and its three samples should together stay under 500 KB of authored content.

### Layer 2: Per-Commit File Budget

The future session works across seven commits inside one pull request. Each commit budget below is a maximum, not a minimum. If the future session approaches the budget it must split overflow content into a follow-up commit rather than oversize a single commit.

| Commit | File count budget | Authored content budget | Generated content budget |
|--------|-------------------|-------------------------|--------------------------|
| 1 Project overview | 7 | 200 KB | 0 |
| 2 Sensor specifications | 8 | 250 KB | 60 MB Parquet (script-generated) |
| 3 Sensor to xyz mapping | 9 | 300 KB | 90 MB Parquet (script-generated) |
| 4 Iterations | 10 | 300 KB | 5.7 GB Parquet across 64 iterations (script-generated; LFS or single-file fallback) |
| 5 Comparison | 12 | 500 KB | 50 MB Parquet plus 5 MB PDF plus 4 MB HTML (script-generated) |
| 6 Error fixes | varies | 100 KB | 0 |
| 7 Repository updates | 3 | 60 KB | 0 |

The generated content budgets above are produced by running the generator scripts after the future session commits the scripts. The generated Parquet files do not enter the LLM context window because they are binary and are produced by a deterministic script call.

### Layer 3: Within-File Chunking

For data products that must be authored inline (the human-review samples), the future session uses the following chunking conventions.

- Sample CSV files cap at 1,000 rows or 100 KB, whichever is smaller. Sampling rule: take the first 1 second at full 1 kHz resolution. This shows the high-frequency dynamics directly.
- Sample JSONL files cap at 1,000 records or 250 KB, whichever is smaller. Sampling rule: stratified sample of 250 records from each of the 4 procedure phases beyond setup.
- Markdown narrative logs cap at 500 lines per file. The 1-hour narrative is authored as `docs/architecture.md` at minute resolution, not millisecond resolution.
- ASCII diagrams cap at 80 columns and 60 lines per `.txt` file, matching the existing convention in `new-trial/national-24-7-trial/hour-00/hour_00_diagram_facility.txt` which is 54 lines.

## Reading Pattern for the Future Session

When the future session begins each commit it reads the following files in order. This keeps the working context lean.

1. `competitions/instructions/README.md`. One read at session start.
2. `competitions/instructions/glioblastoma_context.md`. One read at session start.
3. `competitions/instructions/robot_specification.md`. One read at session start.
4. `competitions/instructions/file_format_conventions.md`. One read at session start.
5. The single `commit_NN_*.md` instruction file for the commit being executed. One read at the start of that commit.
6. Any sibling instruction files referenced by the commit instruction file. Read on demand.
7. The minimum number of source files from the parent repository that the commit instruction file cites. Read on demand.

## Reading Pattern Counter-Examples

The future session must avoid the following patterns.

- Do not read all of `patient-journey/` at once. Read only the file cited by the current commit instruction.
- Do not read all of `new-trial/national-24-7-trial/` at once. Read only the cited hour files (typically `hour-00/`).
- Do not paste any Parquet binary contents into the LLM context. Always reference Parquet files by path and let the generator script produce them.

## Source Files Cited

- `competitions/inputs/paper-a/README.md`. Source for the multi-chunk markdown specification pattern that this directory inherits. The CodeClash paper was deliberately split into 10 chunks of approximately 200 KB each so a future LLM session could process them without exhausting context.
- `competitions/inputs/paper-b/README.md`. Same source pattern for the FAERS paper chunked across 10 files.
- `new-trial/national-24-7-trial/README.md`. Source for the 7-files-per-hour pattern that this directory mirrors.
