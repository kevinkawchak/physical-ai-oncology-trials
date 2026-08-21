# full-move-in - stage 2 of 3, the complete package (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20of%203-00417A.svg)](../sub-prompts/full-move-in)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](sections)
[![Tables](https://img.shields.io/badge/Tables-56%20at%20body%20measure-3C7DB2.svg)](sections)
[![Instructions resolved](https://img.shields.io/badge/Drafting%20instructions-96%20resolved%2C%200%20left-brightgreen.svg)](sections)
[![Compile](https://img.shields.io/badge/Compile-0%20errors%2C%2071%20pages-brightgreen.svg)](main.tex)
[![Overfull](https://img.shields.io/badge/Overfull-0-brightgreen.svg)](main.tex)
[![Citations](https://img.shields.io/badge/Bibliography-76%20of%2076%20cited-brightgreen.svg)](references.bib)
[![Bundle](https://img.shields.io/badge/Overleaf-full--move--in--LaTeX.zip-6C757D.svg)](full-move-in-LaTeX.zip)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI%20v1.0-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../README.md)

Stage 2 turns the stage 1 skeleton into the complete package. All 96 bracketed
drafting instructions are answered from the repository file each named, and then
deleted. Every table is populated and set to the exact body measure. The audit
is a recursive grep for `draftnote` over `sections/`, which returns zero.

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | Unchanged from stage 1 except for the header comment, which now records the column-width method |
| [`movestyle.sty`](movestyle.sty) | Carried forward, with one addition made during the stage error pass: `\mvltable`, a wrapper for a table that must break across pages |
| [`references.bib`](references.bib) | 76 entries, all 76 cited from the body |
| [`sections/`](sections) | 17 files: front matter, fifteen documents, back matter |
| [`full-move-in-LaTeX.zip`](full-move-in-LaTeX.zip) | The Overleaf bundle, 21 files, rebuilt in the same pass as the compile |

## Measured result

| Metric | Stage 1 | Stage 2 |
|:--|:--|:--|
| Errors | 0 | 0 |
| Overfull boxes | 0 | 0 |
| Underfull boxes | 0 | 0 |
| Undefined citations | 0 | 0 |
| Bibliography entries printed | 2 | 76 of 76 |
| Pages | 27 | 71 |
| Tables | 17 shells | 56 populated |
| Source characters, `main.tex` plus `sections/` | 60,992 | 167,972 |
| Drafting instructions | 96 | 0 |

The predecessor package's `all_documents.tex` is 150,972 characters. This stage
is at 167,972, a ratio of 1.11. The difference is structural: this package
carries 56 full-width tables against the predecessor's none, and a table row
costs more source characters per printed line than a paragraph does.

## The column-width method, applied

The method is the author's, taken from `funding/pdac-funding-applications`, and
is stated in full at
[`../sub-prompts/full-move-in/prompt-2-column-widths.md`](../sub-prompts/full-move-in/prompt-2-column-widths.md).
Seven rules govern every one of the 56 tables:

| # | Rule | Audit |
|:--|:--|:--|
| 1 | Every table is a `tabularx` or `xltabular` set to `{\textwidth}` | 56 of 56 |
| 2 | One `X` column per table, and it is the prose column | 56 of 56 |
| 3 | Every fixed column is `>{\raggedright\arraybackslash}p{...}` | 130 of 130 |
| 4 | Width is set from the longest unbreakable token, not the average | Three failures found and fixed in the error pass |
| 5 | The bold header cell counts as a candidate for the widest cell | Two failures found and fixed |
| 6 | `\arraystretch` 1.16 and `\tabcolsep` 4.6 pt, unchanged from the parent | Style-enforced |
| 7 | Every table is wrapped, so the trailing interword space cannot report overfull | 56 of 56 |

## Defects found and fixed in the stage error pass

Fourteen overfull boxes and two uncited entries, each with its measured size.

| Defect | Size | Cause | Fix |
|:--|:--|:--|:--|
| Document index, `Part` header | 0.65 pt | The bold header was wider than the 0.7 cm column, and wider than any body cell in it | Column widened to 1.0 cm, taken from the neighbor |
| Procedure index, `Conventional` cell, ten occurrences | 21.42 pt each | A 1.3 cm column holding a twelve-character word | Column widened to 2.2 cm |
| Evidence table, `progression-` | 4.55 pt | A 1.7 cm column holding a hyphenated compound | Column widened to 2.3 cm |
| Activation checklist page | Overfull vertical box, 196.14 pt | A twenty-row `tabularx` cannot break across pages | Moved to `xltabular` with a repeating header |
| Procedure index page | Overfull vertical box, 223.34 pt | Same cause | Same fix |
| Author record table | Would have been the third | Twenty-four rows | Moved to `xltabular` pre-emptively |
| `cfr13part121` uncited | - | 13 CFR 121.702 was in the bibliography but never referred to | Cited in §1 of document 15, on the ownership firewall |
| `fundapp1` uncited | - | The first funding application was in the bibliography but never referred to | Added as a row in the author record table of document 15 §6 |

## Where each document's content came from (Rule 5)

| Document | Source resolved from the stage 1 instruction |
|:--|:--|
| 01, 02, 03 | [`../../../regulatory/`](../../../regulatory) for the three adapted frameworks; [`../inputs/`](../inputs) for the California bills carried into `references.bib` |
| 04 | [`../../science-golden-age/`](../../science-golden-age) for the policy basis; the predecessor bill H. R. 9510 from the deck README in [`../inputs/READMES/`](../inputs/READMES) |
| 05, 08, 09, 10, 11 | Codified sources cited in `references.bib`, sized against the site parameter table fixed in `sec-00` §3 |
| 06, 12 | [`../../capitalization-plan/final-capital/sections/`](../../capitalization-plan/final-capital/sections) for the gate discipline that states a stop condition before the work begins |
| 07 | [`../../../trial-ind/`](../../../trial-ind) and [`../../../trial-protocol/`](../../../trial-protocol) for the filing and the protocol; [`../../../regulatory/`](../../../regulatory) for the three adaptations |
| 13 | [`../../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex`](../../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex) for the four-source evidence table with each author's stated limitation; [`../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`](../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex) for the three cost benchmark rows |
| 14 | [`../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`](../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex) for the budget frame, reused rather than re-derived; [`../../capitalization-plan/final-capital/`](../../capitalization-plan/final-capital) for the 21 CFR part 54 firewall wording |
| 15 | The same budget frame; [`../inputs/`](../inputs) for the author record; [`../../potential-partners/UC-San-Diego/`](../../potential-partners/UC-San-Diego) for the three positioning corrections |

## What stage 3 will do

Stage 3 introduces no new argument. It applies `\clearpage` discipline, the
author's spacing vocabulary, the dialect and punctuation audits, and two context
verification passes. The list of what it changed is written into
[`../final-move-in/README.md`](../final-move-in/README.md) after the fact.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
