# full-capital - Stage 7, every instruction resolved

[![Stage](https://img.shields.io/badge/Stage-7%20of%208-00417A.svg)](../sub-prompts/stage-7-full-capital)
[![Figures](https://img.shields.io/badge/Figures-20%20drawn-3C7DB2.svg)](sections)
[![Tables](https://img.shields.io/badge/Tables-21-6C757D.svg)](sections)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors-6C757D.svg)](.)
[![Overfull](https://img.shields.io/badge/Overfull%20boxes-0-6C757D.svg)](.)
[![References](https://img.shields.io/badge/References-41%20all%20cited-9AA1A8.svg)](references.bib)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Every `\draftinstr` left by [`../draft-capital`](../draft-capital) is resolved
by reading the file it named, and all twenty figures are drawn in TikZ. Nothing
bracketed survives this stage.

## Files

| File | Contents |
|:--|:--|
| [`main.tex`](main.tex) | Cover, badges, clickable contents, twelve `\input` lines |
| [`capstyle.sty`](capstyle.sty) | The shared style, with the stage-7 retunings below |
| [`references.bib`](references.bib) | 41 entries, every one cited from the body |
| [`sections/`](sections) | One `.tex` per section, `sec-00` to `sec-11` |
| `full-capital-LaTeX.zip` | Overleaf bundle of everything above |

## Compile record

| Check | Result |
|:--|:--|
| `pdflatex` exit code | 0 |
| Overfull boxes | 0 |
| Underfull boxes above `\hbadness=4000` | 0 |
| Undefined citations | 0 |
| Undefined references | 0 |
| Bibliography entries, all cited | 41 |
| Figures drawn | 20 |
| Tables | 21 |
| Pages | 43 |
| Raster images | none |

## The spacing invariant now covers tables

Stage 6 applied `\vspace{-0.65cm}` to figures only. The master prompt requires
it of every new diagram **and table**, so `capstyle.sty` was retuned here:
`apptable` closes with the same rigid `\vskip 24.5pt` that `appfig` closes with,
and `\tabcap` opens with `\nointerlineskip`. Every table in the paper is now
written exactly as every figure is:

```latex
\end{apptable}
\vspace{-0.65cm}
\tabcap{...}
```

The audit is arithmetic. `grep -c 'vspace{-0.65cm}'` over `sections/` returns
41, which is 20 figures plus 21 tables, and equals the count of `\figcaption`
plus the count of `\tabcap`.

## Column-width method, applied

Every table is `tabularx` at `\textwidth` with exactly one `X` column carrying
the longest prose, and every fixed column prefixed
`>{\raggedright\arraybackslash}`. Fixed widths were set from the widest atomic
cell rather than by dividing the measure, and three were corrected after the
first compile measured them:

| Table | Column | Was | Now | Why |
|:--|:--|:--|:--|:--|
| 6, owned register | Identifier | 4.0 cm | 4.4 cm | `\dlink{10.5281/zenodo.20780121}` is unbreakable and overflowed by 2.98 pt in eleven rows |
| 16, six quantities | Comparator | 1.7 cm | 2.1 cm | The bold header `Comparator` overflowed by 10.02 pt |
| 19, risk register | Likelihood | 1.6 cm | 1.9 cm | The bold header `Likelihood` overflowed by 4.88 pt |
| 20, build stages | Commits | 1.4 cm | 1.7 cm | The bold header `Commits` overflowed by 2.74 pt |

A bold header in Times at 10.95 pt is wider than the widest body cell in three
of the four cases, which is the failure mode this method exists to catch.

## Figure verification, run twice

Both passes are recorded here, as the stage sub-prompt requires.

### Pass a: text, box and arrow overlap

| Figure | The pair that could have collided | Clearance held |
|:--|:--|:--|
| 1 | Adjacent decision diamonds on the spine | 3.15 cm pitch against a 2.66 cm diamond, 4.9 mm clear |
| 2 | Header row against body rows | 7.8 mm row pitch against a 7.2 mm cell, 0.6 mm rule gap |
| 3 | Four record nodes side by side | 3.75 cm pitch against a 3.4 cm record, 3.5 mm clear |
| 4 | Tile label against the dashed cluster border | Every `fit` names both `(n)` and `(nl)` |
| 5 | The evidence edge against the absent record | `bend right=16` carries it 8 mm below |
| 6 | Sponsor boundary against site boundary | 9 mm corridor carrying no node |
| 7 | The two gate exits | 20 and 14, not a shared bend |
| 8 | Three money columns | Identical 21 mm width and 8.4 mm height |
| 9 | WP4 to WP5 and WP8 to WP10 | The only two crossings, both in open canvas |
| 10 | Four firewall crossings | Outbound and return pairs 4 mm apart at two x values |
| 11 | Firewall rules against container edges | 8 mm of empty canvas above and below each |
| 12 | Twelve message labels | Uniform 0.45 cm row pitch, 2.1 mm clear at `\tiny` |
| 13 | Twelve gantt bars | 0.52 cm pitch against a 0.36 cm bar, 1.6 mm clear |
| 14 | Five halt-to-gate edges | Fanned across 1.2 cm, arrowheads 3 mm apart |
| 15 | Three concurrent branches | 5.35 cm horizontal pitch against a 2.7 cm box |
| 16 | Grid against interval panel | 10 mm corridor spanned by nothing |
| 17 | Two convergent pairs | One straight and one 18-degree bend, 4 mm apart |
| 18 | Two trust boundaries against clusters | 11 mm and 12 mm to the nearest cluster edge |
| 19 | Five edges converging on M1 | Symmetric 22/12/0/-12/-22 fan |
| 20 | The long public-to-third-party edge | `bend left=16`, 7 mm above the sponsor cluster |

### Pass b: curved arrows and their looseness

Every bend in the paper is stated numerically and falls in one of two bands:
12 to 22 for a short hop between adjacent nodes, and 26 to 36 for a return edge
that must clear an intervening node. Nothing is below 10, which would be
indistinguishable from a straight line, and nothing is above 40, which would
re-enter the node band above.

| Band | Figures | Values used |
|:--|:--|:--|
| Short hop, 12 to 22 | 4, 5, 7, 8, 9, 10, 12, 17, 19, 20 | 10, 12, 14, 16, 18, 20, 22 |
| Return or long clear, 26 to 36 | 3, 7, 15 | 26, 32, 36 |

### Pass c: spacing between boxes

Horizontal pitch is at least the node text width plus 6 mm and vertical pitch at
least the node height plus 5 mm in every figure. Cluster `inner sep` is 6 or
7 pt throughout and no node touches its cluster border.

## Defects found and fixed in this stage

Six, all fixed here rather than carried into stage 8.

1. **Record value cells 1.39 pt too narrow.** `$1,396,000` at `\tiny` Times does
   not fit a 9 mm text width. Figure 3's value column is now 13 mm.
2. **An empty `\foreach` body.** A loop in Figure 3 iterated a list and emitted
   nothing; removed.
3. **`\dlink` overflow in Table 6.** Eleven rows overflowed by 2.98 pt; the
   column is 4.4 cm.
4. **pgfmath dimension parse failure.** `minimum width={\w*0.415} cm` is read as
   a single expression and raises `Unknown operator 'c' or 'cm'`. The unit
   belongs inside the braces: `{\w*0.415cm}`.
5. **A 143 pt overfull vbox on page 9.** Figure 3, Table 5 and the closing
   paragraph of \S1 were competing for one page. Figure 3's vertical bar panel
   was replaced by three horizontal bars, which removed 8.5 pt of canvas and a
   column of empty air, and a `\clearpage` was placed before the last
   subsection of \S1. The float budget was also retuned from the parent's
   topnumber 3 to one float per page top.
6. **The table count was wrong.** The draft index said nineteen; the paper
   carries twenty-one. The index is rebuilt and all sixteen table
   cross-references in the body are renumbered against it.

## Rule 5 source map

| This stage used | From | Where it appears |
|:--|:--|:--|
| `chunk-01`, `03`, `04`, `05`, `08` | `../../science-golden-age` | Every quotation in \S1 and the 3 to 1 target in \S4 |
| `final-apply/sections/sec-08-budget-and-leverage.tex` | `../../pdac-funding-applications` | The four-layer frame in \S3, reused verbatim |
| `final-apply/sections/sec-05-trial-evidence.tex` | `../../pdac-funding-applications` | Table 16 and Table 17 in \S6 |
| `final-apply/sections/sec-06-physical-ai-governance.tex` | `../../pdac-funding-applications` | The 3 ms and 500 ms stops, the trust-boundary argument |
| `final-apply/sections/sec-07-moores-partnership.tex` | `../../pdac-funding-applications` | The IIT intake path in \S8 |
| `final-apply/sections/sec-09-build-method.tex` | `../../pdac-funding-applications` | The single-operator method behind \S7 |
| `applications/app-05-nih-sbir-seed/` | `../../pdac-funding-applications` | The two award amounts throughout |
| `applications/emailed-source/` | `../../pdac-funding-applications` | The 4 to 8 August 2026 record in \S8 |
| `Physical AI Oncology Trial Founding Documents.md` | `../../supplementary` | Table 6, the thirteen owned rows |
| `Physical-AI-Oncology-Trial-Competition-Proposal.zip` | `../../supplementary/source-files` | The 13 January 2026 baseline in \S2 |
| `UC-San-Diego/` | `../../potential-partners` | \S8 and the first-in-human correction in \S6 |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | repository root | \S2, \S5 and \S6 |
| `../mermaid/`, `../plantuml/`, `../d2/`, `../diagrams-python/`, `../graphviz/` | this directory tree | The twenty figures, each drawn to its own specification |
