## Stage 6 sub-prompt - the draft paper

[![Stage](https://img.shields.io/badge/Stage-6%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-6-draft-new-trial)
[![Output](https://img.shields.io/badge/Output-draft--new--trial-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/draft-new-trial)
[![Figures](https://img.shields.io/badge/Figures-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/draft-new-trial)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/draft-new-trial)
[![Commits](https://img.shields.io/badge/Commits-16-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/draft-new-trial)

### Instruction

Produce the draft paper in
[new-trial-system/draft-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/draft-new-trial),
one commit per file, pushed the moment each file is written. The order is fixed:
`trialstyle.sty`, then `main.tex`, then `references.bib`, then `README.md`, then
one commit per section `.tex`, then a defect-correction commit, then the
repository-update commit.

The draft fixes four things no later stage may move: the eleven-file section
set, the numbering of all twenty-five figures through `\figslot`, the numbering
of all twenty-five tables, and a bracketed drafting instruction in every section
naming the exact repository file or directory stage 7 must read.

Adapt a table of contents, the back matter, and the supporting information from
[funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication),
and the cover page from
[new-trial-system/template-new-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/template-new-system).

### Requirements

| Deliverable | Requirement |
|:--|:--|
| Abstract | Final at this stage, under 1350 characters including spaces, no citation numbers, no links |
| Figure slots | 25, each carrying its final number and platform tag |
| Tables | 25, each with final number, column specification and two-line caption |
| Drafting instructions | One or more per section, each naming an exact repository path |
| Bundle | `draft-new-trial-LaTeX.zip`, a self-contained Overleaf project |
| Commits | 16: style, main, bib, README, 11 sections, defect pass, repository update |

### Sources read at this stage

| Source | Used for |
|:--|:--|
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Section 3 |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Section 4 |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Section 4 |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Sections 5 and 7 |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | Section 6, and the style and build method throughout |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Section 6 |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Sections 6 and 7 |
| [new-trial-system/references](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/references) | Every citation |
| The five specification directories under [new-trial-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system) | Every figure |

### Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white. Charcoal is
a stroke and a text color only. **No black fill.** The paper body is black text
throughout; color appears in figures and in link marking only.
