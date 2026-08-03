# draft-apply - Stage 6 of the PART II schedule (paper skeleton)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-6%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-6-draft-apply.md)
[![Sections](https://img.shields.io/badge/Sections-12-3C7DB2.svg)](sections)
[![Figures](https://img.shields.io/badge/Figures-20%20placed-6C757D.svg)](.)
[![Compiles](https://img.shields.io/badge/pdfLaTeX-0%20errors%2C%200%20overfull-6C757D.svg)](.)
[![Paper DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)

The first complete skeleton of *10 Funding Applications: A Phase 1,
First-in-Human, PDAC Clinical Trial Protocol of a LLM-Directed Robotic Whipple
with Daraxonrasib (RMC-6236)*, Draft 1.0.

Every section carries **bracketed `[DRAFTING INSTRUCTION]` markers** naming the
exact repository files and directories `full-apply` must process, and all twenty
figures are placed as sized slots carrying their final number and type tag, so
figure numbering never moves again.

## Files

| File | What it is |
|:--|:--|
| `main.tex` | Cover, clickable table of contents, twelve `\input` lines, back matter |
| `applystyle.sty` | The paper style: `appstyle.sty` with paper furniture added and the ten application cover variants removed |
| `references.bib` | The shared bibliography, carried from [`../applications/references.bib`](../applications/references.bib) |
| `sections/sec-00-front.tex` | Abstract, how to read, figure and table inventory |
| `sections/sec-01-golden-age-mandate.tex` | The $200 billion realignment. Figures 1, 2 |
| `sections/sec-02-ten-applications.tex` | The ten recipients. Figures 3, 6 |
| `sections/sec-03-surgical-set.tex` | Set A, applications 01 to 05. Figures 5, 8 |
| `sections/sec-04-medical-oncology-set.tex` | Set B, applications 06 to 10. Figure 19 |
| `sections/sec-05-trial-evidence.tex` | Chronology, simulations, RASolute 302. Figures 4, 7, 16 |
| `sections/sec-06-physical-ai-governance.tex` | Boundaries, loop, records. Figures 9, 10, 12, 13, 14, 15 |
| `sections/sec-07-moores-partnership.tex` | UC San Diego. **No figure, deliberately** |
| `sections/sec-08-budget-and-leverage.tex` | Budget and cost share. Figures 11, 17 |
| `sections/sec-09-build-method.tex` | The eight-stage build. Figure 18 |
| `sections/sec-10-risks-and-limits.tex` | What is not claimed. Figure 20 |
| `sections/sec-11-references-backmatter.tex` | Abbreviations, availability, citation, references |
| `draft-apply-LaTeX.zip` | Overleaf bundle of the four items above |

## Cover, and how it varies from RFA-RM-27-001-v2

The master prompt requires the cover to vary in appearance from the
[`RFA-RM-27-001-v2`](../../RFA-RM-27-001-v2) theme. That theme is a centred
block of fillable form fields. This cover is instead a **left-accent masthead**
(`\paymast`) with an eyebrow line, a badge row, a **ten-cell application strip**
(`\paystrip`) in two rows of five, and a rule-separated identity block. No
element of the form-field theme is reused.

## What `applystyle.sty` changes from `appstyle.sty`

| Change | Why |
|:--|:--|
| Ten application cover variants removed | The paper has one cover, not ten |
| `\paymast`, `\paycell`, `\paystrip` added | The masthead and application strip |
| `\draftinstr` added | Bracketed drafting instructions, draft stage only |
| `\figslot` added | Sized figure placeholders carrying the final number and tag |
| `\l@section` lead reduced to 0.20em | Twelve sections would otherwise spill the contents onto a second, mostly empty page |
| `\bmhead` adds a TOC entry | Back-matter headings appear in the contents |
| `\appfile` rewritten as a character scanner | A repository path inside a drafting instruction has no spaces and overflowed the measure by up to 188pt; `\nolinkurl` and `\path` both failed inside a macro argument, so the scanner inserts a `\penalty300` break opportunity after every character |
| Environment names unchanged | `appfig`, `appfloat`, `apptable`, `figcaption`, `apprefs` keep their names, so a figure written for an application attachment compiles here without edit |

## Verification at this stage

```
pdflatex main -> bibtex main -> pdflatex main -> pdflatex main
0 errors   0 overfull hboxes   0 undefined citations   20 pages
```

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../applications/appstyle.sty`](../applications/appstyle.sty) | The base of `applystyle.sty` |
| [`../applications/references.bib`](../applications/references.bib) | Copied verbatim |
| [`../applications/`](../applications), all ten | Named in the drafting instructions of §2, §3, §4, §6, §8 |
| [`../mermaid/`](../mermaid), [`../plantuml/`](../plantuml), [`../d2/`](../d2), [`../diagrams-python/`](../diagrams-python), [`../graphviz/`](../graphviz) | Every `\figslot` caption and type tag is taken from these five stage directories |
| [`../../science-golden-age/`](../../science-golden-age) | Named in the drafting instructions of §1 and §8 |
| [`../../RFA-RM-27-001-v2/`](../../RFA-RM-27-001-v2) | §5 and §8 drafting instructions; the cover theme this one varies from |
| [`../../supplementary/`](../../supplementary) | §1, §5, §9, §10 drafting instructions |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | §5 drafting instructions |
| [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | §6 drafting instructions |
| [`../../potential-partners/`](../../potential-partners) | §7 and §10 drafting instructions |
| [`../prompts/prompt-apply.md`](../prompts/prompt-apply.md) | §9 drafting instructions |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
