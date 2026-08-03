## prompt-8-final-apply

**Stage.** PART II, Stage 8 of 8. **Output.** `funding/pdac-funding-applications/final-apply/`.

### Objective

The senior author's proof-reading pass. No new argument is introduced; every
existing one is made to sit correctly on the page. There is **no
`publication/` subdirectory** at this stage.

### Corrections carried forward from `full-apply`

1. **Float carriage.** Wrap every figure-plus-caption in a float so a figure too
   tall for the space left on a page no longer strands that space. Barrier the
   float queue at every section, and `\clearpage` between sections, so no figure
   leaves the section that discusses it.
2. **Rigid caption distance.** Confirm by inspection that the frame-to-caption
   distance is identical for all twenty figures whether the figure floats or is
   set inline: the environment closes with a rigid skip, `\vspace{-0.7cm}`
   follows, and `\figcaption` opens with `\nointerlineskip`.
3. **Caption balance.** Rewrite any caption whose three lines differ noticeably
   in length, so the centred block reads as a balanced triangle.
4. **Widows, orphans, and short last lines.** Maximal penalties plus a
   stretchable `\parfillskip`; then read every page and rewrite any paragraph
   that still ends in one or two words.
5. **No stranded heading.** Reserve the heading plus three lines of its own text
   so a heading can never be the last thing on a page.
6. **Table breaks.** Every long table is breakable and repeats its header, so no
   table runs off the foot of a page.
7. **`\clearpage` discipline.** A `\clearpage` before each major section that
   opens with a figure or a wide table; none where it would leave a page more
   than about a third empty.
8. **Symbols.** `\S` for every codified section reference; single dashes only.
9. **Links.** Every DOI printed as text and hyperlinked; `\UrlBreaks` on all
   characters so no link overflows the measure.

### Diagram polish

Each of the twenty figures is improved once more from `full-apply`: tighter
alignment grids, consistent stroke weights, labels moved off edges, legends
where a figure carries more than three fills, and consistent glyph scale.

### Commits

Twelve section commits plus main, style, bib, README, error-fix, and zip, then
the release commit. Push each immediately.
