# National MCP Servers - Chunked LaTeX Source

Chunked version of the paper **"National MCP Servers for Physical AI Oncology Clinical Trial Systems"** for token-limited AI processing.

## Source

- **Repository**: [kevinkawchak/national-mcp-pai-oncology-trials](https://github.com/kevinkawchak/national-mcp-pai-oncology-trials)
- **Original file**: `paper/National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` (1,011 lines, 62,156 chars)
- **Original file is NOT modified** — these are read-only chunked copies

## Chunk Files

| File | Lines | Description |
|------|-------|-------------|
| `01_preamble_introduction.tex` | 183 | Preamble, document setup, and Section 1 (Introduction) |
| `02_methods_results_architecture.tex` | 261 | Section 2 (Methods) and Section 3.1 (Five-Server MCP Architecture) |
| `03_results_conformance_to_interop.tex` | 245 | Sections 3.2–3.10 (Conformance Levels through Interoperability Testbed) |
| `04_discussion_limitations_conclusion.tex` | 322 | Section 4 (Discussion), Section 5 (Limitations), Section 6 (Conclusion), and back matter |

## Bibliography

- `references.bib` — 19 references from the original paper

## Reconstruction

To reconstruct the original file from chunks:

```bash
cat 01_preamble_introduction.tex \
    02_methods_results_architecture.tex \
    03_results_conformance_to_interop.tex \
    04_discussion_limitations_conclusion.tex \
    > National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
```

## Processing Instructions

When processing these chunked files with AI tools (e.g., Claude Code Opus 4.6):

1. **Read chunks in numerical order** (01, 02, 03, 04) to maintain document flow
2. **Chunk 01 contains the preamble** — all `\usepackage`, `\newcommand`, and style definitions needed to understand the rest of the document
3. **Cross-references**: `\label{}` and `\ref{}` commands may reference content in other chunks. The label definitions are:
   - Chunk 01: `\label{sec:introduction}`
   - Chunk 02: `\label{sec:methods}`, `\label{sec:results}`
   - Chunk 04: `\label{sec:discussion}`, `\label{sec:limitations}`, `\label{sec:conclusion}`
4. **Bibliography**: `references.bib` is shared across all chunks. Citation keys (e.g., `\cite{kawchak2026national}`) appear throughout
5. **Tables and figures** span within individual chunks (no table/figure crosses a chunk boundary)
6. **Concatenation produces the exact original** — no content is added, removed, or modified during chunking
