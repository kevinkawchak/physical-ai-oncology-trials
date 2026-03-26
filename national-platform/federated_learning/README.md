# Federated Learning - Chunked LaTeX Source

Chunked version of the paper **"Federated Learning Physical AI Oncology Trials Unification"** for token-limited AI processing.

## Source

- **Repository**: [kevinkawchak/pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl)
- **Original file**: `paper/main.tex` (930 lines, 56,626 chars)
- **Original file is NOT modified** — these are read-only chunked copies

## Chunk Files

| File | Lines | Description |
|------|-------|-------------|
| `01_preamble_intro_methods.tex` | 216 | Preamble, document setup, Section 1 (Introduction), and Section 2 (Methods) |
| `02_results_architecture_to_cooperation.tex` | 241 | Section 3 Results: Platform Architecture, Privacy, Regulatory, Cross-Framework, Standards, and Cooperation |
| `03_results_workflows_to_cli.tex` | 320 | Section 3 Results: Workflow Demos, Peer Review, Code Trust, Clinical Analytics, Digital Twins, Tech Stack, Federated Coordinator, and CLI Tools |
| `04_discussion_limitations_conclusion.tex` | 153 | Section 4 (Discussion), Section 5 (Limitations), Section 6 (Conclusion), and back matter |

## Bibliography

- `references.bib` — 27 references from the original paper

## Reconstruction

To reconstruct the original file from chunks:

```bash
cat 01_preamble_intro_methods.tex \
    02_results_architecture_to_cooperation.tex \
    03_results_workflows_to_cli.tex \
    04_discussion_limitations_conclusion.tex \
    > main.tex
```

## Processing Instructions

When processing these chunked files with AI tools (e.g., Claude Code Opus 4.6):

1. **Read chunks in numerical order** (01, 02, 03, 04) to maintain document flow
2. **Chunk 01 contains the preamble** — all `\usepackage`, `\lstdefinestyle`, and formatting definitions needed to understand the rest of the document
3. **Cross-references**: `\label{}` and `\ref{}` commands may reference content in other chunks. The label definitions are:
   - Chunk 01: `\label{sec:introduction}`, `\label{sec:methods}`
   - Chunk 02: `\label{sec:results}`
   - Chunk 03: `\label{sec:examples}`
   - Chunk 04: `\label{sec:discussion}`, `\label{sec:limitations}`, `\label{sec:conclusion}`
4. **Bibliography**: `references.bib` is shared across all chunks. Citation keys (e.g., `\cite{kawchak2026fl}`) appear throughout
5. **Tables and listings** span within individual chunks (no table/listing crosses a chunk boundary)
6. **Concatenation produces the exact original** — no content is added, removed, or modified during chunking
