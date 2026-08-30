# Sub-prompt schedule - Pancreatic Cancer LLM Clinical Trial System (v4.6.0)

[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Stages](https://img.shields.io/badge/Stages-8-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts)
[![Figures](https://img.shields.io/badge/Figures-25-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Model](https://img.shields.io/badge/Model-Claude%20Code%20Opus%205-800020.svg)](https://claude.ai/code)

## What this directory is

One master prompt was supplied by the author. Claude Code Opus 5 decomposed it
into the eight sub-prompts held here, then executed them in order without
further author intervention. Every sub-prompt writes to its own output
directory, and every file it writes is committed and pushed the moment it is
finished, so the author can follow branch progress in real time rather than
receiving one large drop at the end.

The schedule is adapted from the eight-stage build the author used for
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan),
whose final source set is at
[funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication).
That work produced 20 figures and 21 tables across 12 sections; this work
produces 25 figures and 25 tables across 11 sections, and is targeted at 1.25
times its character count.

## The eight stages

| Stage | Sub-prompt directory | Output directory | Files | Commits |
|:--|:--|:--|:--|:--|
| 1 | [stage-1-mermaid](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-1-mermaid) | [mermaid](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/mermaid) | 6 figure specifications | 7 |
| 2 | [stage-2-plantuml](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-2-plantuml) | [plantuml](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml) | 4 figure specifications | 5 |
| 3 | [stage-3-d2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-3-d2) | [d2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/d2) | 6 figure specifications | 7 |
| 4 | [stage-4-diagrams-python](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-4-diagrams-python) | [diagrams-python](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/diagrams-python) | 4 figure specifications | 5 |
| 5 | [stage-5-graphviz](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-5-graphviz) | [graphviz](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/graphviz) | 5 figure specifications | 6 |
| 6 | [stage-6-draft-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-6-draft-new-trial) | [draft-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/draft-new-trial) | style, main, bib, README, 11 sections, zip | 16 |
| 7 | [stage-7-full-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-7-full-new-trial) | [full-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial) | style, main, bib, README, 11 sections, zip | 16 |
| 8 | [stage-8-final-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-8-final-new-trial) | [final-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial) | style, main, bib, README, 11 sections, zip | 16 |

Stages 1 through 5 produce specifications only. No LaTeX is written until stage
6, and no figure is drawn in TikZ until stage 7, so figure numbering is fixed
before any prose depends on it and never moves afterward.

## Why five diagram platforms

The figure count per platform is set by purpose, not by quota. Each platform
owns the claims its native constructs express best.

| Platform | Owns | Native constructs used | Figures |
|:--|:--|:--|:--|
| Mermaid | Order in time, and decisions taken at a point in time | flowchart, sequenceDiagram, stateDiagram-v2, gantt | 1, 4, 7, 11, 17, 21 |
| PlantUML | Formal notation with actors, guards, and concurrency | use case, state with guards, activity with fork and join | 3, 10, 14, 23 |
| D2 | Nesting and tabulation | containers, grids, sql tables, layers | 2, 8, 12, 16, 18, 22 |
| Diagrams (Python) | Clustered infrastructure carrying glyphs | clustered nodes with icon tiles | 6, 13, 20, 25 |
| Graphviz | Records, clusters, and fault or decision trees | dot records, subgraph clusters, fault tree | 5, 9, 15, 19, 24 |

## Palette, fixed for all 25 figures

| Token | Hex | Role |
|:--|:--|:--|
| Burgundy | `#800020` | Primary emphasis fill, heavy edges, headings |
| Lighter burgundy 1 | `#A32A3C` | Secondary emphasis fill with white text |
| Lighter burgundy 2 | `#E2D6D9` | Pale soft fill with black text |
| Charcoal | `#2E2E2E` | Strokes, rules, and text only, never a fill |
| Slate Gray | `#6B6B6B` | Process, oversight, and neutral edges |
| Mist Gray | `#C9C9C9` | Neutral fill, and its lighter tints |
| White | `#FFFFFF` | Ground |

Black filled boxes are forbidden. Charcoal is used as a stroke and a text color
only, so no figure carries a near-black fill. The paper body remains black text
throughout; color appears in figures and in link marking only.

## Sources consumed by the schedule

Every stage reads from files already in this repository. Nothing is invented.

| Source | Used by |
|:--|:--|
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Figures 6 to 9, Tables 8 to 10, section 3 |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Figures 10 to 13, Tables 11 to 13, section 4 |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Figures 11 and 12, Tables 12 and 13, section 4 |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figures 14 to 16 and 21 to 24, Tables 14 to 16 and 21 to 24, sections 5 and 7 |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | Figures 17 to 20, Tables 17 to 20, section 6, and the whole build method |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Figure 17, Table 17, section 6 |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Figures 19 and 23, Tables 19 and 22, sections 6 and 7 |
| [new-trial-system/abstracts](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/abstracts) | Tables 6 and 24, sections 2 and 7 |
| [new-trial-system/references](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/references) | Every citation in the paper |
| [new-trial-system/template-new-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/template-new-system) | Cover page, contents, and back matter shape |

## Commit rule

One file, one commit, pushed immediately. Within stages 6, 7 and 8 the order is
fixed: style file, then `main.tex`, then `references.bib`, then the stage
`README.md`, then one commit per section `.tex`, then a defect-correction commit
that fixes every error found across the stage, then the repository-update
commit. No stage is allowed to hold work back for a batch push.
