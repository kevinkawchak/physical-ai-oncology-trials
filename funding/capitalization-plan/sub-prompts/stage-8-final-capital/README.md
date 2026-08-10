## Stage 8 sub-prompt - final-capital

[![Stage](https://img.shields.io/badge/Stage-8%20of%208-00417A.svg)](.)
[![Output](https://img.shields.io/badge/Output-..%2Ffinal--capital-3C7DB2.svg)](../../final-capital)
[![Pass](https://img.shields.io/badge/Pass-senior%20author-6C757D.svg)](../../final-capital)
[![Publication dir](https://img.shields.io/badge/publication%2F-not%20generated-9AA1A8.svg)](../../final-capital)
[![Commits](https://img.shields.io/badge/Commits-16-9AA1A8.svg)](.)

### Instruction

Take the senior author's proof-reading pass over `full-capital` and write the
result to `funding/capitalization-plan/final-capital/`. No new argument is
introduced. Every existing one is made to sit correctly on the page.

**No `publication/` subdirectory is generated at this stage, by instruction.**

### Deliverables and commit order

The same sixteen-commit order as stages 6 and 7.

### The author's own correction list, applied

`funding/pdac-funding-applications/final-apply/publication/useredits.md` records
what the author actually did by hand to the parent work after the machine
finished. Each item is applied here before the author has to ask.

| The author's item | Applied here as |
|:--|:--|
| 01, 02 Table and figure IDs and captions fixed | Every `Figure N` and `Table N` cross-reference in the body is checked against the float that carries it |
| 03 Page overflow fixed | Zero overfull boxes is a release condition, not a target |
| 04 DOIs and URLs shown and clickable | Every `@article` and `@misc` with a DOI carries both `doi` and a `url` pointing at `https://doi.org/`; `\dlink` prints the DOI as text and links it |
| 05 URLs edited until all references work | Every URL in `references.bib` is a resolvable form; none is a search page |
| 06 Table and figure references checked in the body | Every float is referred to at least once in prose within its own section |
| 08 Single-word lines fixed | `\parfillskip=0pt plus 0.75\textwidth` plus a manual sweep for any paragraph ending in one or two words |
| 09 Table columns fixed | The stage 7 column-width method re-run against the compiled widths, not the estimated ones |
| 11 Figure X shifts | `\DiagramXShift` with `adjustwidth` where a figure sits optically left of the measure |
| 13 Arrow overlaps fixed | The third and final overlap sweep, per figure, recorded in the stage README |
| 14, 15 Headings, boxes, and "OR" placement fixed | Every in-figure heading raised clear of the first row; every gate label placed below its glyph |

### Formatting methods to learn and implement

- **`\clearpage` discipline.** A barrier goes between two sections only where
  the next section opens with a float or a full-width table. A barrier before a
  section that opens with prose strands a third of a page and is not placed.
- **`\vspace` and `\hspace`.** The figure-to-caption distance is fixed at
  `\vspace{-0.65cm}` and is never varied to fit a page. Page fitting is done
  with prose, with `\clearpage`, and with the float placement specifier, in that
  order.
- **`\needspace`.** Section, subsection and subsubsection each reserve their
  heading plus their first lines, so no heading is the last thing on a page.
- **`\raggedright` interword spacing.** `\RaggedRightRightskip` is
  `0pt plus 2em`, `\tolerance` 2000, `\emergencystretch` 3 em. The combination
  removes the large interword gaps a fully justified measure produces while
  keeping the right margin from fraying.
- **Self-standing pages.** Each page is read on its own. Where one is
  overcrowded, a sentence is cut. Where one is more than a third empty, a
  sentence is added or a float is moved. Some white space is correct; a page
  that is half empty in the middle of an argument is not.

### Release conditions

| Condition | How it is checked |
|:--|:--|
| 0 LaTeX errors | `pdflatex` twice plus `bibtex`, exit 0 |
| 0 overfull boxes | `grep -c Overfull main.log` returns 0 |
| 0 undefined citations | `grep -c "undefined" main.log` returns 0 |
| 0 undefined references | `grep -c "There were undefined references" main.log` returns 0 |
| Every figure spaced identically | `grep -c 'vspace{-0.65cm}'` equals the figure and table count |
| No em dash or double dash | `grep -c -- '--' sections/*.tex` returns 0 outside `p{}` ranges |
| No raster | `find . -name '*.png' -o -name '*.jpg'` returns nothing |
