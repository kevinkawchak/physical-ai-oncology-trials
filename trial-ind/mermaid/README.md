# mermaid - Stage 1 grayscale figure catalog (IND v1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Figures](https://img.shields.io/badge/Figures-22-000000.svg)](.)
[![Palette](https://img.shields.io/badge/Palette-grayscale%208--tone-6C757D.svg)](.)
[![Renders](https://img.shields.io/badge/Renders-GitHub%20Mermaid%20%2B%20LaTeX%20TikZ-3F3F3F.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-1%20of%204-6C757D.svg)](../sub-prompts/prompt-1-mermaid.md)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-blue.svg)](../../README.md)

Stage 1 of the `trial-ind/` build: 22 new, comprehensive grayscale Mermaid figures
for *Phase 1 PDAC IND: AI Generation*. Each renders natively on GitHub from a
```` ```mermaid ```` block and is reproduced one-to-one as a TikZ `mermaidfig` in
the draft, full, and final LaTeX stages. Every figure is new to this IND, has no
internal component overlap, and names the exact repository source files it draws
from (Rule 5). The body text of the IND keeps the template color (black); only the
figures are recolored, to a strictly grayscale eight-tone ramp.

## Color scheme (grayscale, maps 1:1 to `indstyle.sty` `mm*` styles)

| Role | `classDef` / `mm*` | Tone |
|:--|:--|:--|
| End goal, decision, outcome | `goal` / `mmgoal` | `#000000` |
| LLM / process / system | `proc` / `mmproc` | `#3F3F3F` |
| Acceleration / emphasis | `accent` / `mmaccent` | `#6C757D` |
| Secondary process | `step` / `mmstep` | `#9AA0A6` |
| Input / source file | `input` / `mmin` | `#ECECEC` |
| Context / support | `ctx` / `mmctx` | `#F5F5F5` |
| Decision / gate (diamond) | `dec` / `mmdec` | `#D9D9D9` |
| Rules / raw data / audit | `dark` / `mmdark` | `#222222` |

## Figure inventory (Rule 5: source files named in each file)

| # | File | Title | Primary source files |
|:--|:--|:--|:--|
| 1 | `fig-01-ind-build-pipeline.md` | Single-prompt IND build pipeline | prompts, sub-prompts; final-paper sec-03 |
| 2 | `fig-02-ind-toc-architecture.md` | ReGARDD IND TOC architecture | ReGARDD template; draft-ind main.tex |
| 3 | `fig-03-fda-1571-3674-flow.md` | Cover Letter + FDA 1571 / 3674 flow | FDA-1571 instructions; ReGARDD template |
| 4 | `fig-04-six-acceleration-targets.md` | Six greatest-acceleration IND targets | final-paper sec-04 (Fig 16), sec-05 |
| 5 | `fig-05-ind-irb-package-composition.md` | Initial IND + IRB package composition | final-paper sec-04 (Fig 17); protocol sec-01 |
| 6 | `fig-06-document-landscape-before-during-after.md` | Before / during / after document landscape | final-paper sec-03 (Fig 6); protocol sec-10 |
| 7 | `fig-07-pre-submission-authoring.md` | Pre-submission IND authoring | final-paper sec-03 (Fig 9); llm-adoption; protocol sec-09 |
| 8 | `fig-08-three-timeline-buckets.md` | Three timeline buckets, time saved per document | final-paper sec-04 (Fig 19), sec-05 |
| 9 | `fig-09-five-grounding-verifications.md` | Five name-matching / grounding verifications | final-paper sec-04 (Fig 20) |
| 10 | `fig-10-patient-time-saved-cascade.md` | Patient time-saved cascade | final-paper sec-05 (Fig 24); protocol sec-02 |
| 11 | `fig-11-figure-grounding-mermaid-to-tikz.md` | Figure grounding Mermaid to TikZ | final-paper sec-04 (Fig 15); indstyle.sty |
| 12 | `fig-12-daraxonrasib-document-thread.md` | Daraxonrasib PDAC document thread | final-paper sec-05 (Fig 23); protocol sec-04 |
| 13 | `fig-13-post-submission-maintenance.md` | Post-submission IND maintenance authoring | final-paper sec-03 (Fig 11); protocol sec-10 |
| 14 | `fig-14-daraxonrasib-mechanism.md` | Daraxonrasib mechanism / class | protocol sec-06; references.bib |
| 15 | `fig-15-three-plus-three-escalation.md` | 3+3 dose-escalation automaton | protocol sec-04, sec-09 |
| 16 | `fig-16-cmc-architecture.md` | CMC information architecture | ReGARDD template; protocol sec-06 |
| 17 | `fig-17-pharm-tox-structure.md` | Pharmacology / toxicology summary structure | protocol sec-06; references.bib |
| 18 | `fig-18-perioperative-advisory.md` | Perioperative pause-and-restart advisory | protocol sec-06 |
| 19 | `fig-19-safety-reporting-triggers.md` | Safety clocks + six Physical AI triggers | protocol sec-08, sec-10 |
| 20 | `fig-20-governance-oversight.md` | Governance + three-tier oversight | protocol sec-10, sec-09 |
| 21 | `fig-21-objectives-endpoints.md` | Objectives mapped to endpoints | protocol sec-03, sec-08 |
| 22 | `fig-22-previous-human-experience.md` | Previous human experience evidence base | protocol sec-02; references.bib |

## How these figures translate to LaTeX

Each Mermaid `classDef` maps to a TikZ `mm*` node style in
[`../draft-ind/indstyle.sty`](../draft-ind/indstyle.sty), so the LaTeX figure
carries the same nodes, edges, grayscale tones, and quantitative data as the
Mermaid source. The full and final stages verify each figure twice for text-box
and arrow overlaps, for the specified arrow looseness, and for proper box spacing.

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| [`../../trial-documents/final-paper/publication/sections`](../../trial-documents/final-paper/publication/sections) | the acceleration figures (6, 9, 11, 15, 16, 17, 19, 20, 23, 24) adapted in context to the IND |
| [`../../trial-protocol/final-protocol/publication/sections`](../../trial-protocol/final-protocol/publication/sections) | the clinical content (drug, design, endpoints, safety, oversight) |
| [`../inputs/ReGARDD_IND_Template.docx`](../inputs/ReGARDD_IND_Template.docx) | the IND TOC and section structure (figures 2, 3, 16) |
| [`../inputs/FDA-1571_Instructions_R14_03-21-2023.md`](../inputs/FDA-1571_Instructions_R14_03-21-2023.md) | the 1571 / 3674 cover-sheet logic (figure 3) |
| [`../inputs/references.bib`](../inputs/references.bib) | citations grounding the clinical figures |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
