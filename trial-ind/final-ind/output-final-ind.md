## output-final-ind

Stage 4 narrative for the *Phase 1 PDAC IND: AI Generation* build (IND v1.0,
repository v4.3.0). Process B executed
[`../sub-prompts/prompt-4-final-ind.md`](../sub-prompts/prompt-4-final-ind.md) in
`trial-ind/final-ind/`. There is no `publication` subdirectory under `final-ind`.

### What was produced

- The twelve `sections/sec-*.tex` reached maximum quality, carried over from the
  full stage and improved: about 596,600 characters of finished IND body across
  the twelve files (the whole package, with `references.bib`, `main.tex`, and
  `indstyle.sty`, is approximately ten times the source paper
  `trial-documents/final-paper/publication`).
- All 22 grayscale TikZ figures are rendered, numbered as a single document
  sequence Figure 1 to Figure 22 in document order, with no duplicates and none
  missing. The tables are numbered in section sequences (Table 1.1, Table 3.1, and
  so on); the document carries about ninety full-width tables at the body measure.
- One commit per section, pushed to the working branch in real time.

### Final corrections implemented (improved from the full IND)

- Figure numbering unified to Figure 1 to 22; a duplicate governance figure in
  `sec-05` was removed and replaced with a cross-reference to the canonical
  Figure 20 in §10.4, so every figure is unique.
- Table numbering unified to section sequences; column widths tuned to the body
  measure with ragged-right cells so there are no large interword gaps and nothing
  runs off the right margin.
- Prose deepened with additional regulatory detail, worked numeric examples, and
  explicit 21 CFR cross-references toward the ten-times target.
- Senior-author white-space polish: `\needspace` before every float, `\clearpage`
  per self-standing section (in `main.tex`), no stranded or one-to-two-word lines,
  single dashes only, and the section symbol for codified references.

### Verification (Stage 4 double-check)

A full static pass across all twelve sections confirmed: every LaTeX environment
balanced; no nested `tikzpicture` inside `mermaidfig`; every `\caption` inside a
float; no leftover `\draftinstr`; no en-dashes, em-dashes, or smart quotes; every
`\cite` key present in `references.bib`; the figure sequence complete and unique
(1 to 22); and a geometric scan of all 22 figures found zero horizontal
box-overlap pairs (the tight pairs are vertical stacks with adequate gaps), with
curved arrows carrying the specified looseness.

### Compile

`pdflatex main` then `bibtex main` then `pdflatex main` twice, in Overleaf. The
packaged `final-ind-LaTeX.zip` is a self-contained project.
