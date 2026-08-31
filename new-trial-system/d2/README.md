# d2-type figure specifications

[![Platform](https://img.shields.io/badge/Platform-D2-A32A3C.svg)](https://d2lang.com)
[![Figures](https://img.shields.io/badge/Figures-6%20of%2025-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/d2)
[![Stage](https://img.shields.io/badge/Produced%20by-stage--3--d2-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-3-d2)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

Six of the paper's twenty-five figures are d2-type, because six of its claims
are grids or containers. A grid states a comparison as a cell; a container
states membership as position. Both lose meaning if redrawn as a chain of
arrows, which is the test applied when the platform was chosen.

Three of the six are true grids and carry no edges at all. That is deliberate:
in this paper an edge means causation or sequence, so a comparison that has
neither must not draw one.

## The six figures

| File | Fig | § | Construct | Perspective |
|:--|:--|:--|:--|:--|
| [fig-02-old-versus-new-system-grid.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-02-old-versus-new-system-grid.md) | 2 | 1 | grid | Ten operating axes, prior system against new system |
| [fig-08-ind-1571-crosswalk.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-08-ind-1571-crosswalk.md) | 8 | 3 | sql tables | Codified content item to section file to repository path |
| [fig-12-protocol-inheritance-containers.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-12-protocol-inheritance-containers.md) | 12 | 4 | containers | What Phase 2 carried, replaced, and added |
| [fig-16-statute-to-sop-layers.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-16-statute-to-sop-layers.md) | 16 | 5 | layers | One requirement from statute to operating-room instruction |
| [fig-18-money-grid.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-18-money-grid.md) | 18 | 6 | grid | Three asks, four cost layers, two overhead regimes |
| [fig-22-peer-review-economics-grid.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-22-peer-review-economics-grid.md) | 22 | 7 | grid | Six economic axes of one review round, with ratios |

All four native D2 constructs named in the stage instruction are used:
containers, grids, sql tables, and layers. The three grids differ in shape and
in what a cell means: Figure 2 carries values, Figure 18 carries scope and
money, and Figure 22 carries a computed ratio.

## Files from other directories used here

| Source directory or archive | Used by | For what |
|:--|:--|:--|
| [new-trial-system/abstracts/README.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/abstracts/README.md) | Figures 2, 12, 22 | Single authorship, deposit dates, and the 2026 artifact count |
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Figures 2 and 8 | Twelve section files, the 22-figure catalog, and the deposit DOI |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Figures 12 and 16 | The twelve Phase 1 sections and the Phase 0 gate quantities |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Figure 12 | The Phase 2 sections, version line, and predicate declaration |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figures 2, 16, 22 | The two bills, the clinician trust framework, and the AI peer review study |
| [national-platform](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/national-platform) | Figures 8 and 16 | Adapted 21 CFR 312, 21 CFR 50 and ICH E6(R3) text, and the site standard package |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | Figures 2 and 18 | Every money value, the two overhead regimes, and the `d2*` TikZ vocabulary |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Figure 18 | The ten applications, of which 05 is the mechanism the grid prices |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Figures 18 and 22 | Budget and milestones, and the scheduled review milestones |

## Palette

Burgundy `#800020` for header cells with white text, lighter burgundy 1
`#A32A3C` for emphasis cells, lighter burgundy 2 `#E2D6D9` for new-system value
cells, Mist Gray `#C9C9C9` for axis and counterfactual cells, Slate Gray
`#6B6B6B` for ghost strokes, Charcoal `#2E2E2E` for rules and text, white for
prior-system value cells. No cell in this directory carries a black or
near-black fill.
