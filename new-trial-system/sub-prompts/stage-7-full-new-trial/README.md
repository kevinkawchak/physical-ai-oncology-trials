## Stage 7 sub-prompt - the full paper

[![Stage](https://img.shields.io/badge/Stage-7%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-7-full-new-trial)
[![Output](https://img.shields.io/badge/Output-full--new--trial-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)
[![Figures](https://img.shields.io/badge/Figures-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)
[![Commits](https://img.shields.io/badge/Commits-16-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)

### Instruction

Produce the full paper in
[new-trial-system/full-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial),
in the same commit order as stage 6.

Every figure slot placed in stage 6 is replaced by a drawn TikZ figure built
from its own specification file: the absolute-coordinate construction table and
the edge-routing paragraph, not from memory of another figure. Every table is
populated from the repository source stage 6 named. Every bracketed drafting
instruction is discharged and deleted; no `\draftinstr` survives.

Optimize column widths for aesthetics against the amount of text each column
actually carries, following the author's method in
[funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication).

### Requirements

| Verification | Requirement |
|:--|:--|
| Overlap | Every figure checked twice for text-box and arrow overlap against its own edge-routing paragraph |
| Curved edges | Every bend carries an explicit angle or looseness value; no default curve |
| Box spacing | Stated in the construction table in centimeters, never left to the layout |
| Complexity | Comparable across all twenty-five figures; no figure is a two-box sketch beside a twenty-node one |
| Quotation | Direct quotation from the four author-final `publication/` directories and the four input archives |
| Main sections | IND, Trial Protocol, Legislation, Funding Proposals and AI Peer Review within a narrow band of one another |
| Bundle | `full-new-trial-LaTeX.zip`, a self-contained Overleaf project |

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
