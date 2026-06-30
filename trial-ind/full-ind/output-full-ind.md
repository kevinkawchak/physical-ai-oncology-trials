## output-full-ind

Stage 3 narrative for the *Phase 1 PDAC IND: AI Generation* build (IND v1.0,
repository v4.3.0). Process B executed
[`../sub-prompts/prompt-3-full-ind.md`](../sub-prompts/prompt-3-full-ind.md) in
`trial-ind/full-ind/`.

### What was produced

- The twelve `sections/sec-*.tex` were rendered from the draft scaffold to finished
  IND prose, replacing every `[DRAFTING INSTRUCTION]` with regulator-grade text,
  full-width ragged-right tables, and the grayscale TikZ figures. Total section
  body is about 315,000 characters across the twelve files, plus `main.tex`,
  `indstyle.sty`, and `references.bib`.
- 20 grayscale TikZ `mermaidfig` figures and 31 full-width tables
  (`tabularx`/`xltabular` at `\textwidth`) carry the quantitative data: the DL1
  160 / DL2 220 / DL3 300 mg dose levels and the 3+3 / 28-day-DLT rule, the
  eight-arm device specification (56 DOF, 640 channels, 3 N per-arm and 18 N
  cumulative force caps, 3 ms cross-arm E-stop), the five-vessel no-fly gate, the
  perioperative advisory sweep (29 / 3 / 0 of 32), the Dutch 2025 benchmark
  comparators, the operating characteristics with exact Clopper-Pearson intervals,
  the safety-reporting clocks, and the analysis populations.
- One commit per section, pushed to the working branch in real time as each
  section was written; the structural files (`indstyle.sty`, `main.tex`,
  `references.bib`, `README.md`) were committed first.

### Verification (Stage 3 error-fix pass)

A static pass across all twelve sections confirmed: no nested `tikzpicture` inside
`mermaidfig` (the environment already opens the picture); all LaTeX environments
balanced after repairing one missing `\end{table}` in
`sec-04-investigator-brochure.tex`; every `\caption` inside a float; every
`\cite` key (19 distinct) present in `references.bib`; balanced braces; and no
en-dashes, em-dashes, or smart quotes anywhere.

### Carried into the final stage

The final stage unifies the figure and table numbering into a single sequence in
document order, renders the remaining figures from the Stage 1 catalog so all
grayscale figures appear, deepens the prose toward the ten-times character target,
and applies the senior-author polish (`\clearpage` per self-standing section,
tuned table column widths, `\vspace`/`\hspace`/`\needspace`, and the no-stranded-
line settings).

### Compile

`pdflatex main` then `bibtex main` then `pdflatex main` twice, in Overleaf. The
packaged `full-ind-LaTeX.zip` is a self-contained project.
