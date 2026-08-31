# prompts - the master prompt and the build output of record (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Files](https://img.shields.io/badge/Files-2%20of%20record-00417A.svg)](.)
[![Prompt](https://img.shields.io/badge/Prompt-filed%20verbatim-3C7DB2.svg)](prompt-move-in.md)
[![Output](https://img.shields.io/badge/Output-Claude%20Code%20markdown%20only-6C757D.svg)](output-move-in.md)
[![Model](https://img.shields.io/badge/Model-Claude%20Code%20Opus%205-00417A.svg)](https://claude.com/claude-code)
[![Stage schedule](https://img.shields.io/badge/Schedule-draft%20%E2%86%92%20full%20%E2%86%92%20final-6C757D.svg)](../sub-prompts)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI%20v1.0-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-6C757D.svg)](https://orcid.org/0009-0007-5457-8667)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../README.md)

This directory holds the two files that make the v4.7.0 build reproducible: the
master prompt exactly as it was given, and the Claude Code markdown output
exactly as it was returned. Neither file is edited for readability. If a reader
wants to know why a decision in the paper was made, the answer is either an
instruction in `prompt-move-in.md` or a recorded judgment in
`output-move-in.md`.

## Files

| File | What it is | Rule it satisfies |
|:--|:--|:--|
| [`prompt-move-in.md`](prompt-move-in.md) | A `## prompt-move-in` heading followed by the entire master prompt, word for word, and nothing else | The prompt's closing instruction |
| [`output-move-in.md`](output-move-in.md) | A `## output-move-in` heading followed by the entire Claude Code markdown output, and nothing else. Source files are not reproduced there; they are in the three stage directories | The prompt's closing instruction |

## What the master prompt asks for

| Clause | Instruction | Where it is answered |
|:--|:--|:--|
| A | A La Jolla move-in documentation package for a chief executive and ten coworkers | [`final-move-in/sections/sec-14-staffing-and-roles.tex`](../final-move-in/sections/sec-14-staffing-and-roles.tex) |
| B | Conventional pancreatic cancer clinical trial requirements | [`sec-13-conventional-trial-requirements.tex`](../final-move-in/sections/sec-13-conventional-trial-requirements.tex) |
| C | Lobbying, federal funding, legislation proposals, IND, and protocols for FDA acceptance of LLM and robotic workflows | Documents 04, 07 and 15 |
| D | Author qualifications | `sec-00-front.tex` §0.3 and `sec-15` |
| E | Favorable federal funder responses | `sec-00-front.tex` §0.4, `sec-15` §15.1 |
| F | Three presidential chief executive recognition letters | `sec-00-front.tex` §0.4, `sec-15` §15.2 |
| G | Responsive industry communications | `sec-15` §15.3 |
| H | $700,000 per year for five years from at least one federal agency | `sec-14` §14.5 and `sec-15` §15.4 |
| I | Identify the correct number and type of documents | Fifteen documents, in four parts. The reasoning is in `sec-00-front.tex` §0.2 |
| J | Same table of contents format; each internal document starts on its own page | `main.tex`: `\part` per document, `\clearpage` before each |
| K | La Jolla language, tone and dialect | A dialect audit runs before every commit. No word list entry survives |
| L | The key driver is the author's own work and the White House and presidential support for independent scientists | `sec-00-front.tex` §0.1 and `sec-15` |
| M | `pdac-funding-applications` as the structural template; one sub-prompt directory per stage | [`../sub-prompts/`](../sub-prompts) |
| N | Deposit under `funding/move-in` | This subtree |
| O | A comprehensive README with badges in every directory | Fourteen READMEs |
| P | No spelling or grammatical error; professional white space | The stage 3 proof pass |
| Q | The PDF compiler works every time | Every stage compiles with pdfLaTeX and BibTeX before its commit |
| R | Keep the v4.6.0 pull request; this project is v4.7.0 | Pull request #77, base `main` |

## Files used from other directories (Rule 5)

| Source | Used where in this directory |
|:--|:--|
| [`../../pdac-funding-applications/prompts/`](../../pdac-funding-applications/prompts) | The two-file convention of this directory, one prompt of record and one output of record, and the heading form |
| [`../../capitalization-plan/prompts/`](../../capitalization-plan/prompts) | The habit of recording defects with measured sizes in the output file rather than absorbing them silently |
| [`../inputs/`](../inputs) | The three input artifacts the master prompt names |
| [`../sub-prompts/`](../sub-prompts) | The stage schedule the master prompt sets out, expanded one directory per stage |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
