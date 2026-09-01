# final-move-in - stage 3 of 3, the polished package (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20of%203-00417A.svg)](../sub-prompts/final-move-in)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](sections)
[![Tables](https://img.shields.io/badge/Tables-56%20at%20body%20measure-3C7DB2.svg)](sections)
[![Compile](https://img.shields.io/badge/Compile-0%20errors%2C%2067%20pages-brightgreen.svg)](main.tex)
[![Overfull](https://img.shields.io/badge/Overfull-0-brightgreen.svg)](main.tex)
[![Short pages](https://img.shields.io/badge/Short%20pages-0-brightgreen.svg)](main.tex)
[![Stranded headings](https://img.shields.io/badge/Stranded%20headings-0-brightgreen.svg)](main.tex)
[![Citations](https://img.shields.io/badge/Bibliography-76%20of%2076%20cited-brightgreen.svg)](references.bib)
[![publication](https://img.shields.io/badge/publication%2F-not%20generated-9AA1A8.svg)](../sub-prompts/final-move-in)
[![Bundle](https://img.shields.io/badge/Overleaf-final--move--in--LaTeX.zip-6C757D.svg)](final-move-in-LaTeX.zip)
[![PDF](https://img.shields.io/badge/PDF-main.pdf%2C%2067%20pages-6C757D.svg)](main.pdf)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI%20v1.0-10.5281%2Fzenodo.22216519-blue.svg)](https://doi.org/10.5281/zenodo.22216519)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../README.md)

The senior author's proof-reading pass over `full-move-in`. **No new argument is
introduced.** Every existing one is made to sit correctly on the page, every
number is verified twice, and the dialect, punctuation and spacing checks are
run to zero.

There is no `publication/` subdirectory at this stage, by instruction.

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover, the contents at a tightened lead, and one `\input` per section |
| [`movestyle.sty`](movestyle.sty) | Three stage 3 changes and no others, listed below |
| [`references.bib`](references.bib) | 76 entries, all 76 cited |
| [`sections/`](sections) | 17 files: front matter, fifteen documents, back matter |
| [`final-move-in-LaTeX.zip`](final-move-in-LaTeX.zip) | The Overleaf bundle, 21 files. Unpacked into an empty directory and compiled with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`, it returns 0 errors and 67 pages, so the author fixes nothing |
| [`main.pdf`](main.pdf) | The compiled package, 67 pages, built from these sources in the same pass that produced the bundle, so neither can be newer than the other |

## Measured result across the three stages

| Metric | Draft | Full | Final |
|:--|:--|:--|:--|
| Errors | 0 | 0 | 0 |
| Overfull boxes | 0 | 0 | 0 |
| Underfull boxes | 0 | 0 | 0 |
| Undefined citations | 0 | 0 | 0 |
| Undefined references | 0 | 0 | 0 |
| Bibliography entries printed | 2 | 76 | 76 |
| Pages | 27 | 71 | 67 |
| Contents pages | 4 | 4 | 3 |
| Pages under twelve body lines | 5 | 4 | 0 |
| Pages ending on a heading | not measured | 5 | 0 |
| Tables | 17 shells | 56 | 56 |
| Fixed-width columns, all ragged-prefixed | 51 of 51 | 130 of 130 | 130 of 130 |
| Source characters | 60,992 | 167,972 | 175,256 |

The predecessor's `all_documents.tex` is 150,972 characters; this stage is at
175,256, a ratio of 1.16. The difference is structural rather than verbose: this
package carries 56 full-width tables where the predecessor carries none, and a
table row costs more source characters per printed line than a paragraph does.

## What stage 3 changed, item by item

The stage was driven by measurement, not by reading. The stage 2 PDF was
converted to text, every page was counted, and every page carrying fewer than
twelve body lines or ending on a heading was investigated at its cause.

| # | Defect measured at stage 2 | Cause | Fix, and which instrument |
|:--|:--|:--|:--|
| 1 | Contents ran to four pages, the fourth carrying two lines | `\l@part` lead at 0.62 em and `\l@section` at 0.12 em over 17 part and 96 section entries | Leads reduced to 0.42 em and 0.06 em, contents line spacing to 0.92. Contents now three pages |
| 2 | Page 36 carried a table caption alone, three lines | The commissioning table filled the page section 5 left | `\clearpage` before document 08 §6, the only in-document barrier in the package |
| 3 | Page 56 carried two lines, the close of document 13 | The cost table, its caption and the closing paragraph could not fit together | `\needspace{18\baselineskip}` before the cost table, so the three move as a block |
| 4 | Page 47 carried three lines, the close of document 12 | The last subdivision of §6 broke away from the three above it | `\needspace{16\baselineskip}` before §6, so the four subdivisions move together |
| 5 | Five pages ended on a section heading | `\section` reserved 3.4 baselines, which is satisfied by space a full-width table cannot use | The reservation raised to 9 baselines |
| 6 | A caption could be separated from its table by a page break | Neither table wrapper forbade the break after `\end{tabularx}` | Both wrappers now close with `\nopagebreak` |
| 7 | Fourteen tables of ten rows or more could not break across pages | `tabularx` is not a breaking environment | Fourteen tables moved to `xltabular` with a repeating header, through a new `\mvltable` wrapper |
| 8 | The abbreviation table carried 30 entries, 13 of which never appeared in the body, and two blank cells | The list was written by hand rather than derived | Rebuilt mechanically from the body: 24 entries, 12 full rows, no blank cell |
| 9 | Two bold header cells were wider than their columns | Header width was not counted when the column was sized | Two columns widened |
| 10 | Thirty-eight fixed columns wrapped to more lines than their neighbors | Widths set by eye at stage 2 | All 38 retuned against the longest token in each column |

Six paragraphs were tightened by a clause each, in documents 00, 01, 12, 13 and
15. In every case the instrument is the first one in the fix hierarchy, a
sentence, and never a skip: no `\vspace` was added to the body anywhere in this
stage.

## The three `movestyle.sty` changes

1. Contents leads reduced, as item 1 above.
2. `\mvltable` added, the wrapper for a table that must break across pages.
3. Both table wrappers close with `\nopagebreak`, and `\section` reserves 9
   baselines instead of 3.4.

Everything else in the style is identical to stage 1.

## Verification passes

**Pass 1, against the source.** Every number, date, name and quoted phrase was
checked against the repository file that supplied it: the budget frame against
`sec-08-budget-and-leverage.tex` of the ten-application work; the simulation
results and their stated limitations against `sec-05-trial-evidence.tex`; the
cost benchmarks and the projected \$36,330 against the capitalization plan; the
author record against the accomplishments document; the twenty deposited works
against the seminar deck README; and every codified citation against the source
named in `references.bib`.

**Pass 2, internal consistency.**

| Check | Result |
|:--|:--|
| Eleven full-time equivalent fractions sum to 3.95 | correct |
| Eleven charged salaries sum to \$521,000 | correct |
| Six budget lines sum to \$700,000, and five years to \$3,500,000 | correct |
| Six stall classes sum to 46 | correct |
| Eight robot types sum to 14 instances | correct |
| Three escalation levels at up to six give the eighteen ceiling | correct |
| Five cohorts of five sites at \$17,500,000 give \$87,500,000 | correct |
| Fifteen `\docpart` headings, fifteen cover strip cells, fifteen index rows, and "15 Documents" on the cover | correct |
| Every table referred to by number exists and carries that number | correct, 56 of 56 |
| August 23, 2026 on the cover, in the citation line, and nowhere a different date for the same event | correct |
| v4.7.0 in the cover deposit line and every README badge | correct |
| The identifier is the placeholder `10.5281/zenodo.xxxxxxxx` in all three places and is nowhere fabricated | correct |

## Audits run to zero

| Audit | Result |
|:--|:--|
| Dialect word list, 37 entries | 0 |
| Em dash, en dash, double hyphen, triple hyphen | 0 |
| Literal `SS` where the section symbol belongs | 0 |
| Fixed-width columns without the ragged prefix | 0 of 130 |
| "estimated" beside \$36,330 | 0 |
| Surviving drafting instructions | 0 |
| Bare `\hspace` in a section file | 0 |
| Bare `\vspace` in a section file | 3, all in the back matter around one rule, the parent build's own idiom |
| Raster images anywhere in the subtree | 0 |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../full-move-in/`](../full-move-in) | The input to this stage; every section begins as its stage 2 counterpart |
| [`../../pdac-funding-applications/final-apply/main.tex`](../../pdac-funding-applications/final-apply/main.tex) | The `\clearpage` discipline: a barrier only where the measured page requires it, never scattered |
| [`../../pdac-funding-applications/final-apply/applystyle.sty`](../../pdac-funding-applications/final-apply/applystyle.sty) | The `\needspace` idiom for holding a heading to its content, extended here to hold a block to itself |
| [`../../capitalization-plan/final-capital/`](../../capitalization-plan/final-capital) | The practice of recording each defect with its measured size rather than absorbing it silently |
| [`../inputs/`](../inputs) | Verification pass 1: every claim checked back to the artifact it came from |
| [`../sub-prompts/final-move-in/`](../sub-prompts/final-move-in) | The five sub-prompts this stage executes |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
