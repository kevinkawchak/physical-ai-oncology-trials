# final-apply - Stage 8 of the PART II schedule (senior-author pass)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-8%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-8-final-apply.md)
[![Figures](https://img.shields.io/badge/Figures-20%20polished-3C7DB2.svg)](.)
[![Tables](https://img.shields.io/badge/Tables-18-6C757D.svg)](.)
[![Compiles](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull%2C%200%20underfull-6C757D.svg)](.)
[![Publication dir](https://img.shields.io/badge/publication%2F-none%2C%20by%20instruction-9AA1A8.svg)](.)
[![Paper DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21787424-blue.svg)](https://doi.org/10.5281/zenodo.21787424)

The senior author's proof-reading pass over
[`../full-apply`](../full-apply). No new argument is introduced; every existing
one is made to sit correctly on the page. **There is no `publication/`
subdirectory at this stage, by instruction.**

## What this stage changed

| Change | Why it was needed |
|:--|:--|
| **Float barriers at every section** | A figure could otherwise drift past the heading of the section that discusses it. `\FloatBarrier` is attached to `\section` in `applystyle.sty` |
| **`\clearpage` discipline** | Placed only where the next section opens with a figure or a full-width table. Sections 7 and 10 open with prose and take no barrier, because one there would leave the preceding page more than a third empty |
| **Caption rebalance, all twenty** | The captions were three source lines, but LaTeX rewrapped them in the 0.94-linewidth centred box, so the rendered lines were unbalanced. Every caption now carries explicit breaks chosen to equalise three lines; widest spread 13 characters, median 5 |
| **Prose tightened in two passes** | 48,007 characters down to 41,970, without removing a figure, a table, or a claim |
| **Spacing invariant re-verified** | After the caption rewrite, every `\end{appfig}` is still followed by exactly `\vspace{-0.7cm}` then `\figcaption` |

## Length against the one-quarter target

| Measure | Parent work | This stage | Ratio |
|:--|:--|:--|:--|
| Prose characters, excluding TikZ bodies, tables and comments | 131,774 | 41,970 | 1/3.14 |
| Source characters, `sections/` plus `main.tex` | 301,310 | 105,884 | 1/2.85 |

The target is approximately one quarter and this stage reaches roughly one
third. The exact figures are recorded here rather than rounded, because the
remaining gap is not closed by further cutting: what is left is the argument,
and removing more would remove claims rather than words. Two levers remain
available to the author if the ratio matters more than the content: dropping the
per-application paragraphs in §3 and §4 in favour of the tables that already
carry the same rows, which would save roughly 4,500 characters, and merging §9
into §2, which would save roughly 3,500.

## Figure verification, run twice at this stage

| Check | Result |
|:--|:--|
| No text box or arrow overlap; every node on a stated pitch recorded in a source comment | 20 of 20 |
| Curved-edge looseness stated explicitly; none above 1.1 | 20 of 20 |
| Box-to-box spacing at least 6mm minor axis, 10mm major axis | 20 of 20 |
| Caption exactly three lines, spread at most 13 characters | 20 of 20 |
| Frame-to-caption distance identical, floating or inline | 20 of 20 |
| No black fill; palette limited to the eight `patient-robot-advocacy` tokens | 20 of 20 |

## Formatting methods carried from the parent work

| Method | Where it is applied |
|:--|:--|
| `\RaggedRight` with `\RaggedRightRightskip=0pt plus 2em` | Body text, so no line shows a large interword gap |
| Widow, club, display-widow and broken penalties at 10000 | Every page; no single line is stranded |
| `\parfillskip` at `0pt plus 0.75\textwidth` | No paragraph ends in a one-word or two-word line |
| `\needspace` on `\section`, `\subsection`, `\subsubsection` | No heading is the last thing on a page |
| `\UrlBreaks` on every character, re-asserted after `url` and `hyperref` | No link runs off the right margin |
| `\url` rather than `\href{X}{X}` for long addresses | An `\href` display string cannot break under `\RaggedRight`; `\url` is clickable and breakable |
| `\appfile` character scanner | A repository path has no spaces and would otherwise overflow by up to 188pt |
| Every fixed column `>{\raggedright\arraybackslash}p{...}`, every table at `\textwidth` | All eighteen tables |
| `\S` for every codified section reference; single dashes only | Throughout |

## Files

| File | What it is |
|:--|:--|
| `main.tex` | Cover, contents, twelve `\input` lines with the `\clearpage` discipline |
| `applystyle.sty` | Paper style with float barriers added |
| `references.bib` | Shared bibliography |
| `sections/sec-00-front.tex` .. `sec-11-references-backmatter.tex` | Twelve sections |
| `final-apply-LaTeX.zip` | Overleaf bundle |

## Verification

```
pdflatex main -> bibtex main -> pdflatex main -> pdflatex main
0 errors   0 overfull hboxes   0 underfull hboxes   0 undefined citations   34 pages
```

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../full-apply/`](../full-apply) | The entire content of this stage, then edited |
| [`../../supplementary/source-files/patient-robot-advocacy.zip`](../../supplementary/source-files) | The float-carriage, spacing, penalty and table-column methods listed above, and the 131,774-character prose baseline |
| [`../mermaid/`](../mermaid), [`../plantuml/`](../plantuml), [`../d2/`](../d2), [`../diagrams-python/`](../diagrams-python), [`../graphviz/`](../graphviz) | The construction notes each figure was re-checked against |
| [`../applications/`](../applications) | §2, §3, §4 and §8 values, re-verified against the ten READMEs |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
