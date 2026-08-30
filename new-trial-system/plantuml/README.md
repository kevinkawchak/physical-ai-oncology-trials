# plantuml-type figure specifications

[![Platform](https://img.shields.io/badge/Platform-PlantUML-A32A3C.svg)](https://plantuml.com)
[![Figures](https://img.shields.io/badge/Figures-4%20of%2025-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml)
[![Stage](https://img.shields.io/badge/Produced%20by-stage--2--plantuml-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-2-plantuml)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

Four of the paper's twenty-five figures are plantuml-type, because four of its
claims need formal notation to be stated honestly: a fork that must join, a
transition that fires only when a named quantity evaluates true, or a duty that
belongs to one actor and not another. Drawing any of these as a flowchart would
lose the property that makes the claim checkable.

## The four figures

| File | Fig | § | Construct | Perspective |
|:--|:--|:--|:--|:--|
| [fig-03-master-prompt-fork-join.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-03-master-prompt-fork-join.md) | 3 | 2 | activity, fork and join | Which part of the build is concurrent and which part is strictly serial |
| [fig-10-participant-state-guards.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-10-participant-state-guards.md) | 10 | 4 | state with guards | One participant's states and the quantity behind every transition |
| [fig-14-statutory-actor-duties.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-14-statutory-actor-duties.md) | 14 | 5 | use case | Six actors, eleven duties, and the two no prior-system actor can discharge |
| [fig-23-tripartisan-review-concurrency.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-23-tripartisan-review-concurrency.md) | 23 | 7 | activity, fork and join | Three manufacturers concurrent over one frozen artifact, one human gate |

Figures 3 and 23 are both activity diagrams with a fork and a join, and they are
deliberately different claims: Figure 3 forks five specification stages that
share a fixed figure plan, while Figure 23 forks three reviewers that share a
frozen artifact hash. The first join produces a specification set, the second
produces a disagreement set.

## Files from other directories used here

| Source directory or archive | Used by | For what |
|:--|:--|:--|
| [new-trial-system/prompts/prompt-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/prompt-new-trial.md) | Figure 3 | The master prompt and its eight-stage instruction |
| [new-trial-system/sub-prompts/README.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/sub-prompts/README.md) | Figure 3 | The stage table the activity renders |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Figures 10 and 14 | Phase 0 gate, tip-force caps, restart windows, stopping rules, consent opt-out |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Figure 10 | The monitoring board's de-escalation authority |
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Figure 10 | Day-30 safety window and day-90 pathology assessment as filed |
| [new-trial-system/inputs/HR-9510-Bill-v5.zip](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/inputs/HR-9510-Bill-v5.zip) | Figure 14 | Findings, amendment text, and the cost-ledger duty |
| [new-trial-system/inputs/VVUQ-Physical-AI-Oncology-Trial-Bill.zip](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/inputs/VVUQ-Physical-AI-Oncology-Trial-Bill.zip) | Figure 14 | Statutory text, definitions, attestations, the verification service |
| [new-trial-system/inputs/Earning-the-Clinician's-Trust.zip](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figure 14 | Autonomy disclosure and opt-out duties |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figure 23 | The triple-review study and its consensus finding |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Figure 23 | The production and two-reviewer role assignment, and the human-authority clause |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | All four | The `uml*` TikZ vocabulary adapted here |

## Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white. Charcoal
is a stroke and a text color only, so no fork bar, state, or use case in this
directory carries a black or near-black fill.
