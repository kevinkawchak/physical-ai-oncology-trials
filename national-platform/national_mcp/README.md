# National MCP Servers Paper — Chunked Files

Chunked version of `National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` from the [national-mcp-pai-oncology-trials](https://github.com/kevinkawchak/national-mcp-pai-oncology-trials) repository.

## Source

- **Original file**: `paper/National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` (1,012 lines)
- **Original repo**: [kevinkawchak/national-mcp-pai-oncology-trials](https://github.com/kevinkawchak/national-mcp-pai-oncology-trials)
- **Bibliography**: `references.bib` (included in this directory)

## Chunk Files

| # | File | Lines | Description |
|---|------|-------|-------------|
| 1 | `01_preamble_introduction.tex` | 1–183 | Document preamble, title, abstract, and Section 1 (Introduction) |
| 2 | `02_methods_results.tex` | 184–689 | Section 2 (Methods), Section 3 (Results): five-server MCP architecture, conformance, safety, testing, integration, deployment, SDK, schema, E-stop, testbed |
| 3 | `03_discussion_conclusion.tex` | 690–1012 | Section 4 (Discussion), Section 5 (Limitations and Future Work), Section 6 (Conclusion), Acknowledgments, Ethical Disclosures, Rights and Permissions, Citation |

## Context Preservation

- **Chunk 01** contains the full LaTeX preamble (`\documentclass` through `\begin{document}`, `\maketitle`, abstract, and Introduction)
- **Chunk 02** contains the complete Methods and Results sections with all code listings, tables, and figures
- **Chunk 03** contains Discussion through `\end{document}`, including the `\bibliographystyle` and `\bibliography` commands
- **Bibliography**: The `references.bib` file is included in this directory for cross-referencing

## Reconstruction

To reconstruct the original file from chunks:

```bash
cat 01_preamble_introduction.tex 02_methods_results.tex 03_discussion_conclusion.tex > National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
```

## Notes

- Original file is preserved unmodified in the source repository
- Chunking is necessary to stay within the 20,000 token-per-file limit for Claude Code Opus 4.6 processing
- Files are split at logical section boundaries to maintain context
