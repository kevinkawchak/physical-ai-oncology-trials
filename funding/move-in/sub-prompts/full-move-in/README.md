# sub-prompts/full-move-in - stage 2 of 3 (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20of%203-00417A.svg)](.)
[![Sub--prompts](https://img.shields.io/badge/Sub--prompts-5-3C7DB2.svg)](.)
[![Output](https://img.shields.io/badge/Output-full--move--in%2F-6C757D.svg)](../../full-move-in)
[![Tables](https://img.shields.io/badge/Tables-at%20body%20measure-6C757D.svg)](prompt-2-column-widths.md)
[![Commit floor](https://img.shields.io/badge/Commits-10%2B-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../../README.md)

Stage 2 turns the stage 1 skeleton into the full package. Every bracketed
drafting instruction is answered from the repository file it names, every table
is populated and set to the body measure, and the quantitative case for the
Phase 1 robotic trial is assembled from author sources rather than restated.

## The five sub-prompts

| # | File | Governs |
|:--|:--|:--|
| 1 | [`prompt-1-resolve-drafting-instructions.md`](prompt-1-resolve-drafting-instructions.md) | Answering and deleting every `\draftnote` |
| 2 | [`prompt-2-column-widths.md`](prompt-2-column-widths.md) | The author's column-width method, stated as seven rules and a width budget |
| 3 | [`prompt-3-quantitative-evidence.md`](prompt-3-quantitative-evidence.md) | Which numbers carry the funding case, and the three honesty constraints that travel with them |
| 4 | [`prompt-4-fifteen-documents.md`](prompt-4-fifteen-documents.md) | The required contents of each of the fifteen documents and the eleven-person roster |
| 5 | [`prompt-5-compile-zip-and-readme.md`](prompt-5-compile-zip-and-readme.md) | The compile, the ten mechanical checks, the READMEs, and the bundle |

## Commit ledger for the stage

| Commits | What |
|:--|:--|
| 1 | `movestyle.sty` carried forward |
| 1 | `main.tex` |
| 1 | `references.bib` |
| 17 | one per section file |
| 1 | the error pass (Rule 7, second to last) |
| 1 | READMEs and the Overleaf bundle |
| **22** | **total, against a floor of 10** |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../draft-move-in/`](../draft-move-in) | Stage 1's drafting instructions are this stage's work list |
| [`../../../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex`](../../../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex) | The four-source evidence table and its stated limitations, sub-prompt 3 |
| [`../../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`](../../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex) | The $700,000 per year budget frame, reused rather than re-derived |
| [`../../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`](../../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex) | The three cost benchmark rows, with $36,330 described as projected |
| [`../../../potential-partners/UC-San-Diego/`](../../../potential-partners/UC-San-Diego) | The feasibility sequence and the three positioning corrections |
| [`../../inputs/READMES/`](../../inputs/READMES) | The twenty-paper chronology and the San Francisco roster |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
