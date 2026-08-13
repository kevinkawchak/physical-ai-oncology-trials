## Stage 8 sub-prompt - the final paper

[![Stage](https://img.shields.io/badge/Stage-8%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-8-final-new-trial)
[![Output](https://img.shields.io/badge/Output-final--new--trial-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)
[![Figures](https://img.shields.io/badge/Figures-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)
[![Commits](https://img.shields.io/badge/Commits-16-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)

### Instruction

Produce the final paper in
[new-trial-system/final-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial),
in the same commit order as stages 6 and 7. **Do not create a
`final-new-trial/publication` directory**: that space belongs to the author.

Context and formatting quality reach maximum here. Every diagram, every table
and every page is verified against stage 7 and improved, and the corrections
identified during stage 7 are implemented rather than carried forward.

### Requirements

| Senior-author technique | Where it is applied |
|:--|:--|
| `\clearpage` discipline | A barrier only where the next section opens with a float, so no page is left more than a third empty |
| Table column widths | Re-cut against the compiled widths rather than the estimated ones |
| `\vspace` and `\hspace` | Used to remove large empty space without overcrowding; the -0.6cm caption invariant is never varied |
| `\needspace` | Before every inline float, so no figure strands the space above it |
| Caption balance | Two lines within a four-character spread, checked figure by figure |
| Stranded lines | No heading last on a page, no paragraph ending in a one-word or two-word line |
| US clinical terms | Every term on the master prompt's conversion list checked and corrected |
| Dashes | Single hyphens only; no em dash, en dash, double or triple hyphen in prose |
| Section symbol | `\S` for every codified reference |
| Bundle | `final-new-trial-LaTeX.zip`, a self-contained Overleaf project |

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
