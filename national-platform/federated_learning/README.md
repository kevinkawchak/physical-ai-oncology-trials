# Chunked LaTeX File: main.tex

## Overview

The original `main.tex` (930 lines) has been split into 4 sequential chunks to stay within the 20,000-token-per-file limit of Claude Code Opus 4.6. **No content has been modified.** The chunks are a direct line-for-line partition of the original file.

## File Manifest

| File | Lines | Original Range | Content |
|------|-------|----------------|---------|
| `main_chunk1_preamble_intro_methods.tex` | 215 | 1–215 | Document preamble, packages, style config, `\begin{document}`, abstract, Table of Contents, §1 Introduction, §2 Methods |
| `main_chunk2_results_architecture_pillars_examples.tex` | 318 | 216–533 | §3 Results (start): Architecture Overview, Pillars 1–5, Workflow Demonstrations (domain + agentic AI + physical AI examples) |
| `main_chunk3_results_peerreview_trust_analytics.tex` | 243 | 534–776 | §3 Results (cont.): Triple AI Peer Review, Code Trust & Safety, Clinical Analytics, Digital Twins, Technology Stack, CLI Tools |
| `main_chunk4_discussion_limitations_conclusion.tex` | 154 | 777–930 | §4 Discussion, §5 Limitations & Future Work, §6 Conclusion, References, Acknowledgments, Ethical Disclosures, Rights, Citation, `\end{document}` |

**Total: 930 lines (matches original)**

## How to Reassemble

To reconstruct the original `main.tex` exactly:

```bash
cat main_chunk1_preamble_intro_methods.tex \
    main_chunk2_results_architecture_pillars_examples.tex \
    main_chunk3_results_peerreview_trust_analytics.tex \
    main_chunk4_discussion_limitations_conclusion.tex \
    > main.tex
```

## Processing Instructions for AI Assistants

When working with these chunked files, follow these rules to maintain context across chunks:

1. **Read order matters.** Chunks must be read in numerical order (1 → 2 → 3 → 4). Each chunk continues directly from the previous one with no overlap or gap.

2. **Chunk 1 contains all preamble/setup.** Package imports, custom styles, `\hypersetup`, `\lstdefinestyle`, and `\setlist` configuration are all in chunk 1. Any edits to document-level settings must happen there.

3. **`\begin{document}` is in chunk 1 (line 70); `\end{document}` is in chunk 4 (line 930).** No chunk is independently compilable LaTeX — they are raw slices.

4. **Cross-references span chunks.** Labels defined in one chunk (e.g., `\label{tab:quantitative}` in chunk 1, `\label{tab:architecture}` in chunk 2) may be referenced from any other chunk via `\ref{}` or `\cite{}`. When editing, verify that referenced labels still exist.

5. **Section continuity.** §3 Results spans chunks 2 and 3. If editing Results content, both chunks must be consulted to understand the full section scope.

6. **Bibliography.** `\bibliography{references}` appears in chunk 4. The external `references.bib` file is required for compilation and is not included in this chunked set.

7. **To compile,** always reassemble first (see command above), then run `pdflatex` + `bibtex` as usual. Never attempt to compile individual chunks.

## Chunk Boundary Reference

- **Chunk 1 ends** after §2.3 (Prompt-Driven Development Methodology), right before the `\section{Results}` heading.
- **Chunk 2 starts** at `\section{Results}` and ends after §3.6.3 (Physical AI Examples) and the paragraph on synergistic inter-script operation.
- **Chunk 3 starts** at §3.7 (Triple AI Peer Review Pipeline) and ends after §3.12 (CLI Tools for Trial Operations).
- **Chunk 4 starts** at `\section{Discussion}` and runs through `\end{document}`.
