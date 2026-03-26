# National MCP Servers - Chunked LaTeX Source

Chunked version of `National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` from [kevinkawchak/national-mcp-pai-oncology-trials](https://github.com/kevinkawchak/national-mcp-pai-oncology-trials/blob/main/paper/).

## Source

- **Original file:** `paper/National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` (1,011 lines, ~62 KB)
- **Original repository:** [national-mcp-pai-oncology-trials](https://github.com/kevinkawchak/national-mcp-pai-oncology-trials)
- **DOI:** [10.5281/zenodo.18916731](https://doi.org/10.5281/zenodo.18916731)

## Chunk Files

| File | Lines | Content |
|------|-------|--------|
| `national_mcp_chunk_01_preamble_intro_methods.tex` | 1-282 | Preamble, abstract, Introduction (§1), Methods (§2) |
| `national_mcp_chunk_02_results.tex` | 283-689 | Results (§3): Five-server architecture, conformance, safety, tests, integration |
| `national_mcp_chunk_03_discussion_conclusion.tex` | 690-1011 | Discussion (§4), Limitations (§5), Conclusion (§6), References, Acknowledgments |

## Bibliography

- `references.bib` — Complete bibliography file (copy from original repository)

## Reconstruction

To reconstruct the original file from chunks:

```bash
cat national_mcp_chunk_01_preamble_intro_methods.tex \
    national_mcp_chunk_02_results.tex \
    national_mcp_chunk_03_discussion_conclusion.tex \
    > National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
```

## Context Preservation Notes

- **Chunk 1** contains the complete LaTeX preamble (`\documentclass`, all `\usepackage` declarations, `\newcommand` definitions), `\begin{document}`, `\maketitle`, abstract, keywords, and all of §1 (Introduction) and §2 (Methods). All document setup is in this chunk.
- **Chunk 2** contains the entire §3 (Results) section with all subsections: five-server architecture tables, conformance levels, safety modules, test coverage, integration adapters, repository scale, deployment infrastructure, SDK tooling, JSON schemas, emergency stop code listing, and interoperability testbed.
- **Chunk 3** contains §4 (Discussion) with architecture analysis and comparisons, §5 (Limitations and Future Work), §6 (Conclusion), `\bibliography{references}`, Acknowledgments, Ethical Disclosures, Rights and Permissions, Cite This Article, and `\end{document}`.
- **Cross-references:** `\ref` and `\cite` commands span chunks. Labels defined in one chunk may be referenced in another. The `references.bib` file is required for bibliography resolution across all chunks.
- **Original file is NOT modified.** These chunks are read-only copies split at logical section boundaries.
