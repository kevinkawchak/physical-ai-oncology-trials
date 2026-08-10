# Prompts - Capitalization Plan (v4.5.0)

[![Prompt](https://img.shields.io/badge/Master%20prompt-verbatim-00417A.svg)](prompt-capital.md)
[![Update](https://img.shields.io/badge/Update%20prompt-verbatim-00417A.svg)](update-capital.md)
[![Output](https://img.shields.io/badge/Build%20output-complete-3C7DB2.svg)](output-capital.md)
[![Stages](https://img.shields.io/badge/Stages%20driven-8-6C757D.svg)](../sub-prompts)
[![Figures](https://img.shields.io/badge/Figures%20produced-20-6C757D.svg)](..)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

The files that make this build reproducible: the instructions that produced it
and the record of what happened when the first of them ran.

## Contents

| File | Contents | Rule |
|:--|:--|:--|
| [`prompt-capital.md`](prompt-capital.md) | A `## prompt-capital` heading followed by the master prompt, word for word and nothing else | Master prompt, prompts section |
| [`update-capital.md`](update-capital.md) | An `## update-capital` heading followed by the update prompt, word for word and nothing else | Same form, applied to the update pass |
| [`output-capital.md`](output-capital.md) | An `## output-capital` heading followed by the build output, and nothing else | Same |

None of the three carries commentary. `prompt-capital.md` is the instruction
exactly as received, including its typography, so a reader can check any claim
in this directory tree against the sentence that produced it.
`update-capital.md` is the instruction for the update pass that followed, in the
same form. `output-capital.md` is the narrative record of the eight stages: what
was built, what was found and fixed, and the two places where an instruction
could not be followed literally. It records the original build and is not
rewritten by later passes; what the update pass changed is recorded in
[`../README.md`](../README.md) and
[`../final-capital/README.md`](../final-capital/README.md).

## What the update prompt asked for, and where each part landed

| The instruction | Where it was satisfied |
|:--|:--|
| `main.pdf` and the Overleaf zip correct and updated together | Both rebuilt in one pass from one source set, [`../final-capital`](../final-capital) |
| Every figure caption opening `Figure N.` | 20 captions in [`../final-capital/sections/`](../final-capital/sections) |
| Every table caption opening `Table N.` | 21 captions in the same place |
| Captions centred, balanced, at most three lines | All 41 at three lines, centred within 0.53 pt of the page centre |
| Reference tables and paper mentions matching | Tables 2 and 3 in §0, the stage table in §10, and the four renumbered figures |
| Every figure and table referred to once in the body | 20 of 20 and 21 of 21, verified by parsing the sources |
| Figures, tables and captions centred in x | Root cause fixed in `capstyle.sty`; all 20 frames at 306.00 pt |
| Clickable DOIs and URLs in the references | `unsrturl` plus `\UrlFont`; 20 DOIs and 17 URLs now print and link |
| "projected" rather than "estimated" for $36,330 | §2 prose and all three rows of Table 17 |

## What the prompt asked for, and where each part landed

| The instruction | Where it was satisfied |
|:--|:--|
| A company-conversion paper on the five-part outline A to E | [`../final-capital/sections/`](../final-capital/sections), §1 to §5 |
| Twenty new diagrams across five machine-readable platforms | [`../mermaid/`](../mermaid), [`../plantuml/`](../plantuml), [`../d2/`](../d2), [`../diagrams-python/`](../diagrams-python), [`../graphviz/`](../graphviz) |
| A split by purpose, not by quota | 5 / 3 / 5 / 3 / 4, reasoned in [`../sub-prompts/README.md`](../sub-prompts/README.md) |
| A sub-prompt directory per diagram type, then draft, full, final | [`../sub-prompts/`](../sub-prompts), eight stage directories |
| `\vspace{-0.65cm}` between every diagram and table and its caption | `capstyle.sty`, and 41 occurrences across the final sections |
| Captions of three balanced lines | All 41 at three lines, each balanced to the narrowest spread its word boundaries allow |
| No `final-capital/publication/` directory | Not generated |
| A cover varying from the RFA-RM-27-001-v2 theme | `\capmast` and `\capledger` in `capstyle.sty` |
| Clickable URLs, DOI text with clickable DOI targets, no overflow | `unsrturl` and `\dlink`, `\UrlBreaks` re-asserted after `url` and `hyperref`, 0 overfull boxes |
| Real-time commits, no batching | One commit per file at the moment it was written |
| A comprehensive README in every directory | Eleven directories, each with badges and a Rule 5 source map |
| No PNG or JPG | None generated; the diagrams stage emits a specification, not a `.py` file |

## Rule 5 source map

| This directory uses | From | For |
|:--|:--|:--|
| `prompts/prompt-apply.md`, `prompts/output-apply.md` | `../../pdac-funding-applications` | The two-file form and the heading convention |
| `prompt-capital.md` | this directory | The form `update-capital.md` adapts: one heading, then the prompt verbatim |
| `prompts/README.md` | `../../pdac-funding-applications` | The structure of this README |
| The build itself | this directory tree | Everything recorded in `output-capital.md` |
