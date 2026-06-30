## output-draft-ind

Stage 2 narrative for the *Phase 1 PDAC IND: AI Generation* build (IND v1.0,
repository v4.3.0). Process B executed
[`../sub-prompts/prompt-2-draft-ind.md`](../sub-prompts/prompt-2-draft-ind.md) in
`trial-ind/draft-ind/`.

### What was produced

- `main.tex` with the cover page (the COVER PAGE block: title, Draft 1.0, DOI and
  ORCID hyperlinks, San Diego, July 1 2026, IND v1.0 and repository v4.3.0), the
  ReGARDD ordering (the Cover Letter and the FDA Forms 1571 and 3674 precede the
  numbered Table of Contents, so the Introduction is numbered 3), and `\clearpage`
  per major section.
- `indstyle.sty`, the shared grayscale style adapting the 21 CFR Part 312 template
  and the final-paper paperstyle primitives, with black body text and the
  eight-tone grayscale `mm*` figure styles, committed once.
- `references.bib`, copied from `../inputs/references.bib`, with the `ieeetr`
  bibliography emitting clickable URLs and DOI text plus clickable DOI URLs.
- Twelve `sections/sec-*.tex` scaffolds, one per ReGARDD TOC section, each filled
  with `[DRAFTING INSTRUCTION]` markers that name the exact repository files the
  full stage will process and the grayscale figures (`fig-01 .. fig-22`) to render.

### Verification (Stage 2 error-fix pass)

A static pass confirmed: balanced braces in `main.tex`, `indstyle.sty`, and all
twelve section files; every `\input{sections/...}` target present; the document
preamble, `\begin{document}`, and `\end{document}` intact; and no en-dashes,
em-dashes, or smart quotes anywhere (single dashes only). The numbered Table of
Contents is produced with `\@starttoc{toc}` so the ReGARDD numbering is exact
(1 FDA Forms, 2 Table of Contents, 3 Introduction, through 11 Relevant
Information), with the References and Back Matter as unnumbered back matter.

### Compile

`pdflatex main` then `bibtex main` then `pdflatex main` twice, in Overleaf. The
packaged `draft-ind-LaTeX.zip` (16 files) compiles as a self-contained project.
The full stage replaces each `\draftinstr` marker with finished prose, full-width
tables, and the grayscale TikZ figures.
