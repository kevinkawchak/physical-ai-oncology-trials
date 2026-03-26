# Federated Learning - Chunked LaTeX Source

Chunked version of `main.tex` from [kevinkawchak/pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl/blob/main/paper/).

## Source

- **Original file:** `paper/main.tex` (930 lines, ~57 KB)
- **Original repository:** [pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl)
- **DOI:** [10.5281/zenodo.18795507](https://doi.org/10.5281/zenodo.18795507)

## Chunk Files

| File | Lines | Content |
|------|-------|--------|
| `fl_chunk_01_preamble_intro_methods.tex` | 1-216 | Preamble, abstract, Introduction (§1), Methods (§2) |
| `fl_chunk_02_results_part1.tex` | 217-534 | Results (§3) part 1: Architecture, Privacy, Regulatory, Unification, Standards, Cooperation, Workflow demos |
| `fl_chunk_03_results_part2_discussion_conclusion.tex` | 535-930 | Results (§3) part 2: Peer review, Code trust, Analytics, Digital twins, Tech stack, Discussion (§4), Limitations (§5), Conclusion (§6), References |

## Bibliography

- `references.bib` — Complete bibliography file (copy from original repository)

## Reconstruction

To reconstruct the original file from chunks:

```bash
cat fl_chunk_01_preamble_intro_methods.tex \
    fl_chunk_02_results_part1.tex \
    fl_chunk_03_results_part2_discussion_conclusion.tex \
    > main.tex
```

## Context Preservation Notes

- **Chunk 1** contains the complete LaTeX preamble (`\documentclass`, all `\usepackage` declarations, hyperref setup, listing styles, list formatting), `\begin{document}`, `\maketitle`, abstract, keywords, table of contents, and all of §1 (Introduction) and §2 (Methods). All document setup is in this chunk.
- **Chunk 2** contains §3 (Results) subsections 3.1–3.8: Platform Architecture Overview, Pillar 1 (Privacy), Pillar 2 (Regulatory), Pillar 3 (Cross-Framework Unification), Pillar 4 (Standards & Benchmarking), Pillar 5 (Multi-Organization Cooperation), Workflow Demonstrations, and Agentic AI Examples.
- **Chunk 3** contains §3 (Results) subsections 3.9–3.14 (Physical AI Examples, Triple AI Peer Review, Code Trust & Safety, Clinical Analytics, Digital Twins, Technology Stack, Federated Coordinator, CLI Tools), plus §4 (Discussion), §5 (Limitations and Future Work), §6 (Conclusion), `\bibliography{references}`, Acknowledgments, Ethical Disclosures, Rights and Permissions, Cite This Article, and `\end{document}`.
- **Cross-references:** `\ref` and `\cite` commands span chunks. Labels defined in one chunk may be referenced in another. The `references.bib` file is required for bibliography resolution across all chunks.
- **Original file is NOT modified.** These chunks are read-only copies split at logical section boundaries.
