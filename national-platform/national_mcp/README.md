# Chunked LaTeX File — Reassembly Guide

## Original File

`National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex` (1,011 lines)

This file was split into 4 sequential chunks to stay within per-file token limits (e.g., Claude Code Opus 4.6's 20,000-token cap). **No content was added, removed, or modified.** The chunks are a pure line-range split of the original.

## Chunk Inventory

| File | Lines | Content |
|---|---|---|
| `chunk1_preamble_and_introduction.tex` | 1–181 | Document class, packages, macros, `\begin{document}`, title, abstract, keywords, full Introduction section (§1) including subsections on fragmentation, MCP opportunity, contributions, prior architecture, and regulatory context. |
| `chunk2_methods_and_results_part1.tex` | 182–465 | Methods section (§2: development process, prompt engineering, peer review) and first half of Results (§3 through conformance levels table — five-server architecture, all tool inventories, conformance levels). |
| `chunk3_results_part2.tex` | 466–689 | Second half of Results (§3 continued: safety modules, test coverage, integration adapters, repository scale, deployment infrastructure, SDK/tooling, schemas, emergency stop code listing, interoperability testbed). |
| `chunk4_discussion_and_conclusion.tex` | 690–1011 | Discussion (§4: national standard analysis, interoperability, patient safety, federated learning, robot procedure data flow, national deployment topology, governance, comparison table, CI/CD), Limitations & Future Work (§5), Conclusion (§6), bibliography, acknowledgments, ethical disclosures, rights/permissions, citation. |

## How to Reassemble

Concatenate the four chunks in order to reconstruct the original file exactly:

```bash
cat chunk1_preamble_and_introduction.tex \
    chunk2_methods_and_results_part1.tex \
    chunk3_results_part2.tex \
    chunk4_discussion_and_conclusion.tex \
    > National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
```

## Preserving Context Across Chunks

When processing individual chunks with an LLM, keep the following in mind:

1. **Chunk 1 contains all LaTeX setup.** The `\documentclass`, `\usepackage` declarations, custom commands (`\orcidicon`, column types), `lstdefinestyle`, and `\begin{document}` all live here. Any chunk processed alone will not be valid standalone LaTeX — this is by design.

2. **Cross-references span chunks.** Labels such as `\label{sec:methods}`, `\label{tab:five-servers}`, `\label{lst:chain}`, etc., are defined in one chunk and may be referenced via `\ref{}` or `\cite{}` in another. When editing a chunk, do not remove or rename labels without checking all other chunks for corresponding `\ref{}` calls.

3. **The bibliography is in chunk 4.** All `\cite{}` keys used throughout chunks 1–3 resolve against `\bibliographystyle{unsrt}` and `\bibliography{references}` at the end of chunk 4. The external file `references.bib` is required for compilation.

4. **Section numbering is implicit.** LaTeX auto-numbers sections. The ordering Introduction (§1) → Methods (§2) → Results (§3) → Discussion (§4) → Limitations (§5) → Conclusion (§6) depends on chunk order being preserved.

5. **`\end{document}` is only in chunk 4.** Do not append it to earlier chunks.

6. **Figures and listings span chunks.** Code listings (Listings 1–2) and verbatim figures appear in chunks 2, 3, and 4. Their `\label` and `\caption` pairs must stay intact.

## Compilation

After reassembly, compile with the standard LaTeX toolchain:

```bash
pdflatex National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
bibtex National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems
pdflatex National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
pdflatex National_MCP_Servers_for_Physical_AI_Oncology_Clinical_Trial_Systems.tex
```

Required external dependencies: `arxiv.sty`, `references.bib`, `orcid_icon.png`.
