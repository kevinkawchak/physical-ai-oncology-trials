# sub-prompts/final-move-in - stage 3 of 3 (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20of%203-00417A.svg)](.)
[![Sub--prompts](https://img.shields.io/badge/Sub--prompts-5-3C7DB2.svg)](.)
[![Output](https://img.shields.io/badge/Output-final--move--in%2F-6C757D.svg)](../../final-move-in)
[![Publication dir](https://img.shields.io/badge/publication%2F-not%20generated-9AA1A8.svg)](prompt-5-compile-zip-and-readme.md)
[![Proof pass](https://img.shields.io/badge/Proof-dialect%20%2B%20context%20%2B%20measure-6C757D.svg)](prompt-3-dialect-and-proofreading.md)
[![Commit floor](https://img.shields.io/badge/Commits-10%2B-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../../README.md)

Stage 3 is the senior author's pass. No new argument is introduced. Every
existing one is made to sit correctly on the page, every number is verified
twice, and the dialect, punctuation and spacing checks are run to zero.

## The five sub-prompts

| # | File | Governs |
|:--|:--|:--|
| 1 | [`prompt-1-clearpage-discipline.md`](prompt-1-clearpage-discipline.md) | `\clearpage`, `\FloatBarrier`, `\needspace`, and the fix hierarchy that prefers a sentence over a skip |
| 2 | [`prompt-2-vspace-hspace-and-tables.md`](prompt-2-vspace-hspace-and-tables.md) | The author's spacing vocabulary and the arithmetic table measure audit |
| 3 | [`prompt-3-dialect-and-proofreading.md`](prompt-3-dialect-and-proofreading.md) | The dialect word list, the punctuation checks, the register, and the five-step proof pass |
| 4 | [`prompt-4-context-verification.md`](prompt-4-context-verification.md) | Two verification passes: against the source, and against the paper's own internal consistency |
| 5 | [`prompt-5-compile-zip-and-readme.md`](prompt-5-compile-zip-and-readme.md) | The compile, the bundle, the two READMEs, and the repository close |

## What stage 3 changed, relative to stage 2

The list is written after the fact into
[`../../final-move-in/README.md`](../../final-move-in/README.md), item by item,
so a reader can see what the proof pass was worth rather than being told it
happened.

## Commit ledger for the stage

| Commits | What |
|:--|:--|
| 1 | `movestyle.sty` carried forward |
| 1 | `main.tex` with the tightened `\clearpage` discipline |
| 1 | `references.bib` |
| 17 | one per section file |
| 1 | the error pass (Rule 7, second to last) |
| 1 | READMEs and the Overleaf bundle |
| **22** | **total, against a floor of 10** |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../full-move-in/`](../full-move-in) | Stage 2 is the input; every correction identified there is applied here |
| [`../../../pdac-funding-applications/final-apply/main.tex`](../../../pdac-funding-applications/final-apply/main.tex) | The `\clearpage` discipline commentary, adapted in sub-prompt 1 |
| [`../../../pdac-funding-applications/final-apply/applystyle.sty`](../../../pdac-funding-applications/final-apply/applystyle.sty) | The spacing vocabulary itemized in sub-prompt 2 |
| [`../../../capitalization-plan/final-capital/`](../../../capitalization-plan/final-capital) | The habit of recording defects with measured sizes rather than absorbing them |
| [`../../inputs/`](../../inputs) | Every claim is verified back to the artifact it came from in sub-prompt 4 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
