## output-full-paper

Stage 3 of the mermaid -> draft -> full -> final build is complete. Process B
executed [`prompt-full-paper.md`](prompt-full-paper.md) and produced the full version
of *Phase 1 Pancreatic Cancer Trial Efficient LLM Document Generations* (paper v1.0,
repository v4.2.0), one commit per file pushed in real time.

Every draft `[DRAFTING INSTRUCTION]` was resolved into full prose. All 24 Mermaid
figures were reproduced as TikZ `mermaidfig` figures of the same complexity, numbered
sequentially as Figures 1 to 24 across the Introduction (1-3), Methods (4-15),
Results (16-22), and Discussion (23-24). Six full-width tables carry the gate
taxonomy, the stage outputs, the six acceleration targets, the five verification
checks, the prior single-prompt repositories, and the 2025 evidence chronology.

The figures were verified twice for the three figure properties: every `\draw`
references a defined node (no text-box and arrow overlaps from undefined anchors),
curved edges use bounded `to[out=,in=]` looseness, and box spacing is set by an
even coordinate grid. To guarantee no figure runs off the right margin, the
`mermaidfig` environment now wraps its content in an `adjustbox` max-width box that
scales down only the widest figures while leaving the rest unchanged. The tables use
the `>{\raggedright\arraybackslash}` column types at `\textwidth`, with column widths
tuned to the text per column following the
`trial-protocol/final-protocol/publication` methods.

All 24 citation keys resolve against `references.bib`, single dashes are used
throughout, and the references render with clickable URLs and clickable DOI URLs. The
next stage, [`../final-paper`](../final-paper), applies the senior-author polish
(`\clearpage` per section, `\vspace`/`\hspace` adjustments, and a re-verification of
every figure and table) for maximum quality.
