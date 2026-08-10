# draft-capital/sections - the twelve section sources

[![Stage](https://img.shields.io/badge/Stage-6%20of%208-00417A.svg)](../../sub-prompts/stage-6-draft-capital)
[![Sections](https://img.shields.io/badge/Sections-12-3C7DB2.svg)](.)
[![Rule](https://img.shields.io/badge/Rule%206-one%20.tex%20per%20section-6C757D.svg)](../../prompts/prompt-capital.md)
[![Stage kind](https://img.shields.io/badge/Contents-Skeleton-6C757D.svg)](..)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One `.tex` per section, each corresponding to an `\input` line in
[`../main.tex`](../main.tex). This is Rule 6 of the master prompt: the paper is
never a single source file, and every section is committed on its own.

At this stage the twelve files carry sized figure slots and bracketed drafting instructions.

Every `\draftinstr` in these twelve files names an exact repository path with `\appfile`, so stage 7 has nothing to search for. Twenty `\figslot` calls hold the figure numbers, so nothing renumbers after this stage.

## The twelve files

| File | § | Title | Carries |
|:--|:--|:--|:--|
| [`sec-00-front.tex`](sec-00-front.tex) | 0 | Abstract, Executive Summary, Reader's Guide | Tables 1, 2, 3 |
| [`sec-01-novel-performer-case.tex`](sec-01-novel-performer-case.tex) | 1 | The Novel-Performer Case | Figures 1, 2, 3; Tables 4, 5 |
| [`sec-02-entity-and-asset.tex`](sec-02-entity-and-asset.tex) | 2 | The Entity and the Asset | Figures 4, 5, 6; Tables 6, 7 |
| [`sec-03-gate-and-programme.tex`](sec-03-gate-and-programme.tex) | 3 | The \$1.6M Gate and the \$3.5M Programme | Figures 7, 8, 9; Tables 8, 9, 10, 11 |
| [`sec-04-capital-bridge.tex`](sec-04-capital-bridge.tex) | 4 | Non-Dilutive to Dilutive Bridge | Figures 10, 11, 12; Tables 12, 13 |
| [`sec-05-twelve-milestones.tex`](sec-05-twelve-milestones.tex) | 5 | Twelve Milestones a Program Officer Can Audit | Figures 13, 14, 15; Tables 14, 15 |
| [`sec-06-clinical-evidence.tex`](sec-06-clinical-evidence.tex) | 6 | The Clinical Evidence a Funder Is Buying | Figures 16, 17; Tables 16, 17 |
| [`sec-07-operating-plan.tex`](sec-07-operating-plan.tex) | 7 | Small-Business Operating Plan | Figure 18; Table 18 |
| [`sec-08-san-diego-traction.tex`](sec-08-san-diego-traction.tex) | 8 | San Diego and the August 2026 Record | Figure 19 |
| [`sec-09-risks-and-limits.tex`](sec-09-risks-and-limits.tex) | 9 | Risks, Stop Conditions, and What This Is Not | Table 19 |
| [`sec-10-build-method.tex`](sec-10-build-method.tex) | 10 | Build Method and Reproducibility | Figure 20; Table 20 |
| [`sec-11-references-backmatter.tex`](sec-11-references-backmatter.tex) | 11 | Back Matter | Table 21 |

## Conventions every file in this directory obeys

| Convention | Form |
|:--|:--|
| Figure spacing | `\end{appfig}` then `\vspace{-0.65cm}` then `\figcaption{...}` |
| Table spacing | `\end{apptable}` then `\vspace{-0.65cm}` then `\tabcap{...}` |
| Table measure | `tabularx` at `\textwidth`, exactly one `X` column |
| Table columns | every fixed column prefixed `>{\raggedright\arraybackslash}` |
| Captions | exactly three manually balanced lines |
| Dashes | single hyphens only; no em dash, en dash, double dash or triple dash |
| Codified references | the section symbol, never the letters `SS` |
| Images | none; every figure is pure TikZ |

## Rule 5 source map

Each file's own header comment names the repository sources it draws on. The
directory-level map is in [`../README.md`](../README.md), and the complete
permitted source set is in
[`../../sub-prompts/stage-6-draft-capital/README.md`](../../sub-prompts/stage-6-draft-capital/README.md).
