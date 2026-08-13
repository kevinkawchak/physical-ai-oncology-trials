# graphviz-type figure specifications

[![Platform](https://img.shields.io/badge/Platform-Graphviz-A32A3C.svg)](https://graphviz.org)
[![Figures](https://img.shields.io/badge/Figures-5%20of%2025-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/graphviz)
[![Stage](https://img.shields.io/badge/Produced%20by-stage--5--graphviz-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-5-graphviz)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

Five of the paper's twenty-five figures are graphviz-type, because five of its
claims are structural rather than chronological: a record with fields, a fault
tree, a lineage, a port record, and a decision tree. The dot idiom's thin
charcoal strokes on white are kept deliberately, so these five read as
structure rather than as process.

## The five figures

| File | Fig | § | Construct | Perspective |
|:--|:--|:--|:--|:--|
| [fig-05-figure-storage-record.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-05-figure-storage-record.md) | 5 | 2 | dot record | What a specification stores against what a raster stores |
| [fig-09-clinical-hold-fault-tree.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-09-clinical-hold-fault-tree.md) | 9 | 3 | fault tree | The combinations of basal failure that produce a clinical hold |
| [fig-15-bill-lineage-clusters.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-15-bill-lineage-clusters.md) | 15 | 5 | cluster with records | Five bill versions, their deltas, and the four companions that supplied them |
| [fig-19-award-work-package-ports.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-19-award-work-package-ports.md) | 19 | 6 | record with ports | Where each dollar lands, and the two packages the award does not reach |
| [fig-24-disagreement-resolution-tree.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-24-disagreement-resolution-tree.md) | 24 | 7 | decision tree | Five terminal dispositions for a reviewer disagreement |

## Files from other directories used here

| Source directory or archive | Used by | For what |
|:--|:--|:--|
| The five specification directories under [new-trial-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system) | Figure 5 | The sizes the storage record reports |
| [funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan) | Figures 5, 19, 24 | The prior twenty specifications re-read as input, every money value, and the `gv*` vocabulary |
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Figure 9 | CMC, brochure, consent, and sponsor reporting commitments |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Figures 9 and 19 | Pause and stopping rules, force caps, and the Phase 0 gate |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figures 9, 15, 24 | The two bills, the clinician trust framework, and the AI peer review study |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Figure 19 | Application 05, the SBIR mechanism the award is drawn under |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Figures 19 and 24 | Budget and milestones, and the disagreement-documentation requirement |
| [new-trial-system/abstracts/README.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/abstracts/README.md) | Figure 15 | The seven June 2026 deposit dates |

## Palette

Burgundy `#800020` for key nodes and record headers with white text, lighter
burgundy 1 `#A32A3C` for mid nodes, lighter burgundy 2 `#E2D6D9` for soft nodes
and record fields, Mist Gray `#C9C9C9` and its tints for neutral nodes and gate
glyphs, Slate Gray `#6B6B6B` for dashed cluster strokes, Charcoal `#2E2E2E` for
node strokes at 0.55 pt, edges, and text, white for ground. No node in this
directory carries a black or near-black fill.
