# sub-prompts - Process A output (Phase 1 PDAC IND: AI Generation, IND v1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Sub-prompts](https://img.shields.io/badge/Sub--prompts-4-000000.svg)](.)
[![Stages](https://img.shields.io/badge/Stages-mermaid%20%E2%86%92%20draft%20%E2%86%92%20full%20%E2%86%92%20final-6C757D.svg)](.)
[![IND](https://img.shields.io/badge/IND-Phase%201%20PDAC%20Daraxonrasib-3F3F3F.svg)](..)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-blue.svg)](../../README.md)

This directory holds the four **stage sub-prompts** that Process A generated from
the single master prompt in [`../prompts/prompt-ind.md`](../prompts/prompt-ind.md).
Process B then executes them in order, each writing one stage directory of the
`trial-ind/` build. The pattern is adapted from the
[`trial-protocol/sub-prompts`](../../trial-protocol/sub-prompts) workflow.

## The four sub-prompts

| # | Sub-prompt | Runs in stage | Adapted from (trial-protocol) |
|:--|:--|:--|:--|
| 1 | [`prompt-1-mermaid.md`](prompt-1-mermaid.md) | [`../mermaid/`](../mermaid) | `sub-prompts/prompt-1-mermaid.md` |
| 2 | [`prompt-2-draft-ind.md`](prompt-2-draft-ind.md) | [`../draft-ind/`](../draft-ind) | `sub-prompts/prompt-2-draft-protocol.md` |
| 3 | [`prompt-3-full-ind.md`](prompt-3-full-ind.md) | [`../full-ind/`](../full-ind) | `sub-prompts/prompt-3-full-protocol.md` |
| 4 | [`prompt-4-final-ind.md`](prompt-4-final-ind.md) | [`../final-ind/`](../final-ind) | `sub-prompts/prompt-4-final-protocol.md` |

## Convention (Rule 6, 7, 8)

Every sub-prompt opens with a `## prompt-<name>` heading followed by the prompt
text. One commit per distinguishable file; the second-to-last commit of every
stage fixes all errors; the last commit performs the remaining repository
updates. All commits and pull requests are pushed to GitHub in real time so the
branch can be monitored without intervention.

## What changed from the trial-protocol sub-prompts

- The four stages target an **IND** (the ReGARDD IND Table of Contents) rather
  than an NIH protocol, so the section list follows the eleven IND TOC items plus
  a Cover Letter and a References and Back Matter section.
- Figures are **grayscale** (eight-tone ramp) rather than colored, mapped 1:1 to
  the `indstyle.sty` `mm*` styles; the body text keeps the template color (black).
- The draft stage emits bracketed `[DRAFTING INSTRUCTION]` markers that name the
  exact repository files each later stage processes.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
