# sub-prompts/draft-move-in - stage 1 of 3 (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-1%20of%203-00417A.svg)](.)
[![Sub--prompts](https://img.shields.io/badge/Sub--prompts-5-3C7DB2.svg)](.)
[![Output](https://img.shields.io/badge/Output-draft--move--in%2F-6C757D.svg)](../../draft-move-in)
[![Section files](https://img.shields.io/badge/Section%20files-17-6C757D.svg)](../../draft-move-in/sections)
[![Commit floor](https://img.shields.io/badge/Commits-10%2B-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../../README.md)

Stage 1 produces a compiling skeleton whose job is to remove every open question
before stage 2 begins. Its distinguishing feature is the bracketed drafting
instruction: each one names an exact repository file or directory that stage 2
must read, so the second stage resolves instructions rather than inventing
content.

## The five sub-prompts

| # | File | Produces |
|:--|:--|:--|
| 1 | [`prompt-1-scaffold-and-style.md`](prompt-1-scaffold-and-style.md) | `movestyle.sty`, shared by all three stages |
| 2 | [`prompt-2-cover-and-contents.md`](prompt-2-cover-and-contents.md) | `main.tex`: cover page, table of contents, 17 section inputs |
| 3 | [`prompt-3-fifteen-document-skeletons.md`](prompt-3-fifteen-document-skeletons.md) | `sections/sec-00` through `sec-15`, one commit each |
| 4 | [`prompt-4-bibliography-and-backmatter.md`](prompt-4-bibliography-and-backmatter.md) | `references.bib` and `sections/sec-16-backmatter.tex` |
| 5 | [`prompt-5-compile-zip-and-readme.md`](prompt-5-compile-zip-and-readme.md) | The compile, the error pass, the two READMEs, and `draft-move-in-LaTeX.zip` |

## Commit ledger for the stage

| Commits | What |
|:--|:--|
| 1 | `movestyle.sty` |
| 1 | `main.tex` |
| 1 | `references.bib` |
| 17 | one per section file |
| 1 | the error pass (Rule 7, second to last) |
| 1 | READMEs and the Overleaf bundle |
| **22** | **total, against a floor of 10** |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../../../pdac-funding-applications/sub-prompts/part-ii/prompt-6-draft-apply.md`](../../../pdac-funding-applications/sub-prompts/part-ii/prompt-6-draft-apply.md) | The bracketed drafting instruction convention and the acceptance test form |
| [`../../../pdac-funding-applications/final-apply/applystyle.sty`](../../../pdac-funding-applications/final-apply/applystyle.sty) | Everything `movestyle.sty` inherits, listed in sub-prompt 1 |
| [`../../inputs/`](../../inputs) | The three artifacts sub-prompts 2, 3 and 4 read |
| [`../../prompts/prompt-move-in.md`](../../prompts/prompt-move-in.md) | Clauses A through R, mapped onto the seventeen section files in sub-prompt 3 |
| [`../full-move-in/`](../full-move-in) | The consumer of every drafting instruction this stage writes |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
