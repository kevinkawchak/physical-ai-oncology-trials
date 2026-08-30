# mermaid-type figure specifications

[![Platform](https://img.shields.io/badge/Platform-Mermaid-A32A3C.svg)](https://mermaid.js.org)
[![Figures](https://img.shields.io/badge/Figures-6%20of%2025-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/mermaid)
[![Stage](https://img.shields.io/badge/Produced%20by-stage--1--mermaid-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-1-mermaid)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

Six of the paper's twenty-five figures are mermaid-type, because six of its
claims are about order in time or about a decision taken at a point in time.
Each file below is the complete specification for one figure: its perspective,
its two-line caption exactly as printed, valid Mermaid source, a TikZ
construction table with absolute coordinates, an edge-routing paragraph, and the
repository files the figure's numbers come from.

Nothing in this directory is rendered as a raster. The Mermaid source states the
claim in a machine-readable form; the paper draws the same claim in TikZ, using
the construction table, so the published figure is vector throughout.

## The six figures

| File | Fig | § | Construct | Perspective |
|:--|:--|:--|:--|:--|
| [fig-01-policy-chain-to-capability-gap.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-01-policy-chain-to-capability-gap.md) | 1 | 1 | flowchart LR | Eleven Federal actions, three capabilities supplied, one left unfilled |
| [fig-04-one-generation-turn.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-04-one-generation-turn.md) | 4 | 2 | sequenceDiagram | One turn from master prompt to pushed commit, with both return paths |
| [fig-07-ind-assembly-clock.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-07-ind-assembly-clock.md) | 7 | 3 | gantt | Twelve IND modules in hours against the prior system's months |
| [fig-11-escalation-ladder.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-11-escalation-ladder.md) | 11 | 4 | flowchart LR | Phase 0 gate through 3+3 escalation to Phase 2 randomization |
| [fig-17-funding-artifact-calendar.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-17-funding-artifact-calendar.md) | 17 | 6 | gantt | Fourteen funding artifacts on one 74-day calendar |
| [fig-21-two-review-clocks.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-21-two-review-clocks.md) | 21 | 7 | sequenceDiagram | One manuscript through human review and AI review on one clock |

All four native Mermaid constructs are used, and none is used twice for the same
kind of claim: two flowcharts for decision structure, two sequence diagrams for
message order, two gantt charts for calendar span.

## Files from other directories used here

| Source directory or archive | Used by | For what |
|:--|:--|:--|
| [new-trial-system/references/trump-ai-cancer-2025-2026.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/references/trump-ai-cancer-2025-2026.bib) | Figure 1 | Eleven citation keys and the date each action carries |
| [new-trial-system/prompts/prompt-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/prompt-new-trial.md) | Figure 4 | The master prompt row 1 begins with |
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Figure 7 | The twelve IND module files and the deposit date |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Figure 11 | Phase 0 gate quantities, the 3+3 design, the n = 18 cap |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Figure 11 | n = 220, eight centers, 140 events, hazard ratio 0.60 |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Figure 17 | The ten applications and their August deposit |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Figures 17 and 21 | The two RFA versions and the tripartisan review schedule |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | All six | The figure frame, caption and spacing invariants adapted throughout |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figure 21 | The AI peer review study's review-cycle numbers |
| [new-trial-system/abstracts/README.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/abstracts/README.md) | Figures 1, 7, 11, 17, 21 | Every deposit date used on a calendar or a clock |

## Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white. Charcoal
appears only as a stroke, a rule, or text. No figure in this directory carries a
black or near-black fill.
