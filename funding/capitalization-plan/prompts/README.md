# Prompts - Capitalization Plan (v4.5.0)

[![Prompt](https://img.shields.io/badge/Master%20prompt-verbatim-00417A.svg)](prompt-capital.md)
[![Output](https://img.shields.io/badge/Build%20output-complete-3C7DB2.svg)](output-capital.md)
[![Stages](https://img.shields.io/badge/Stages%20driven-8-6C757D.svg)](../sub-prompts)
[![Figures](https://img.shields.io/badge/Figures%20produced-20-6C757D.svg)](..)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

The two files that make this build reproducible: the instruction that produced
it and the record of what happened when it ran.

## Contents

| File | Contents | Rule |
|:--|:--|:--|
| [`prompt-capital.md`](prompt-capital.md) | A `## prompt-capital` heading followed by the master prompt, word for word and nothing else | Master prompt, prompts section |
| [`output-capital.md`](output-capital.md) | An `## output-capital` heading followed by the build output, and nothing else | Same |

Neither file carries commentary. `prompt-capital.md` is the instruction exactly
as received, including its typography, so a reader can check any claim in this
directory tree against the sentence that produced it. `output-capital.md` is the
narrative record of the eight stages: what was built, what was found and fixed,
and the two places where an instruction could not be followed literally.

## What the prompt asked for, and where each part landed

| The instruction | Where it was satisfied |
|:--|:--|
| A company-conversion paper on the five-part outline A to E | [`../final-capital/sections/`](../final-capital/sections), §1 to §5 |
| Twenty new diagrams across five machine-readable platforms | [`../mermaid/`](../mermaid), [`../plantuml/`](../plantuml), [`../d2/`](../d2), [`../diagrams-python/`](../diagrams-python), [`../graphviz/`](../graphviz) |
| A split by purpose, not by quota | 5 / 3 / 5 / 3 / 4, reasoned in [`../sub-prompts/README.md`](../sub-prompts/README.md) |
| A sub-prompt directory per diagram type, then draft, full, final | [`../sub-prompts/`](../sub-prompts), eight stage directories |
| `\vspace{-0.65cm}` between every diagram and table and its caption | `capstyle.sty`, and 41 occurrences across the final sections |
| Captions of three balanced lines | All twenty at three lines, spread at most three characters |
| No `final-capital/publication/` directory | Not generated |
| A cover varying from the RFA-RM-27-001-v2 theme | `\capmast` and `\capledger` in `capstyle.sty` |
| Clickable URLs, DOI text with clickable DOI targets, no overflow | `\dlink`, `\UrlBreaks` re-asserted after `url` and `hyperref`, 0 overfull boxes |
| Real-time commits, no batching | One commit per file at the moment it was written |
| A comprehensive README in every directory | Eleven directories, each with badges and a Rule 5 source map |
| No PNG or JPG | None generated; the diagrams stage emits a specification, not a `.py` file |

## Rule 5 source map

| This directory uses | From | For |
|:--|:--|:--|
| `prompts/prompt-apply.md`, `prompts/output-apply.md` | `../../pdac-funding-applications` | The two-file form and the heading convention |
| `prompts/README.md` | `../../pdac-funding-applications` | The structure of this README |
| The build itself | this directory tree | Everything recorded in `output-capital.md` |
