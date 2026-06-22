## output-draft-protocol

Stage 2 produced the **draft (scaffold)** of the Phase 2, multicenter, randomized,
controlled protocol in `trial-phase-2/draft-protocol/`. The scaffold is the earlier
skeleton from which the full and final stages were rendered: every NIH section and
subsection heading is present and ordered, but the body is composed of bracketed
`[DRAFTING INSTRUCTION]` blocks (the `\draftinstr` macro) rather than full prose.

`main.tex` carries the `\documentclass[11pt]{article}` and `\usepackage{protostyle}`
preamble, the Phase 2 title, author Kevin Kawchak, the June 23, 2026 date, a DRAFT
cover labeled "Draft scaffold, v1.1.0", the keyword block, a clickable table of
contents, and one `\input` per NIH section for `sections/sec-00` through
`sections/sec-12`. The shared `protostyle.sty` (Burgundy `#800020` Phase 2 palette,
`mermaidfig`, `tabularx` column primitives, the PNG-free TikZ ORCID mark, and the
`\draftinstr` marker) and `references.bib` were copied in from the publication build.

Each of the 13 `sections/sec-NN.tex` files opens with the correct `\section{...}`
title and the publication's `\subsection{...}` headings, and under each heading a
concise `\draftinstr{...}` block states (a) the Phase 2 publication section and the
Phase 1 model file to follow, (b) the exact repository source files to process
(`trial-protocol/nih-protocol/0X_*.md`, `trial-protocol/inputs/2030-pdac-1min-final-paper`,
`trial-protocol/inputs/21cfr312_adapt`, `trial-protocol/inputs/auto-bill-02`, the
Phase 1 `trial-protocol/draft-protocol/sections/*`, the Phase 2
`final-protocol/publication/sections/*`, and the `trial-phase-2/mermaid` figures),
and (c) which figure number and table to render there.

The figure and table placement follows the established numbering exactly: Figure 1
in sec-00; Figure 2 and `tab:soa` in sec-01; Figures 3-6 with `tab:concerns` and
`tab:coinvest` in sec-02; Figure 7 and `tab:objend` in sec-03; Figures 8-9 in
sec-04; Figure 10 in sec-05; Figures 11-14 with `tab:arms` and `tab:sensors` in
sec-06; no figure in sec-07; Figures 15-17 in sec-08; Figure 18 with `tab:power` and
`tab:secendpts` in sec-09; Figures 19-21 in sec-10; Figure 22 with `tab:jurisdictions`,
`tab:amend`, and `tab:abbrev` in sec-11; and the `ieeetr` bibliography and back
matter in sec-12. All twenty-two figures map to `trial-phase-2/mermaid/fig-NN-*.md`
pointers, and every `\cite` key resolves against `references.bib`.

The stage also includes this `output-draft-protocol.md`, the verbatim
`prompt-draft-protocol.md`, and a comprehensive `README.md`. The project compiles in
Overleaf with pdfLaTeX (`pdflatex -> bibtex -> pdflatex -> pdflatex`). Hard rules
were observed: single hyphens only in prose, `\S` for the section symbol, no PNG, and
only valid `references.bib` cite keys.
