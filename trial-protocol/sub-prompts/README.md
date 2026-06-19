# sub-prompts - Process A output (Physical AI oncology trial protocol, v4.0.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Sub-prompts](https://img.shields.io/badge/Sub--prompts-4-00417A.svg)](.)
[![Stage](https://img.shields.io/badge/Stages-mermaid%20%E2%86%92%20draft%20%E2%86%92%20full%20%E2%86%92%20final-6C757D.svg)](.)
[![Protocol](https://img.shields.io/badge/Protocol-Phase%201%20IND%2FIDE-00417A.svg)](.)

This folder is the output of **Process A**: from the single master prompt filed
verbatim in [`../prompts/prompt-protocol.md`](../prompts/prompt-protocol.md),
Claude Code generated the four sub-prompts that **Process B** then runs in
sequence to grow the protocol from Mermaid figures to draft, full, and final
LaTeX. Each sub-prompt is adapted from the corresponding stage prompt of the
`inputs/auto-bill-02` workflow and re-targeted to the *Phase 1, First-in-Human,
Combined IND/IDE Clinical Trial Protocol of On-Premises LLM-Directed Robotic
Pancreaticoduodenectomy (Whipple) with Perioperative Daraxonrasib (RMC-6236)*.

## The four sub-prompts

| # | Sub-prompt | Runs in stage | Adapted from (auto-bill-02) |
|:--|:--|:--|:--|
| 1 | [`prompt-1-mermaid.md`](prompt-1-mermaid.md) | `../mermaid/` | `sub-prompts/prompt-3-mermaid-selection.md` + `prompt-4-figure-selection.md` |
| 2 | [`prompt-2-draft-protocol.md`](prompt-2-draft-protocol.md) | `../draft-protocol/` | `sub-prompts/prompt-5-draft-bill.md` |
| 3 | [`prompt-3-full-protocol.md`](prompt-3-full-protocol.md) | `../full-protocol/` | `sub-prompts/prompt-6-full-bill.md` |
| 4 | [`prompt-4-final-protocol.md`](prompt-4-final-protocol.md) | `../final-protocol/` | `sub-prompts/prompt-7-final-bill.md` |

## Convention (Rule 6, Rule 14 of the auto-bill-02 lineage)

Every sub-prompt opens with a `## prompt-<name>` heading followed by the prompt
text. Each stage that runs a sub-prompt files its own `prompt-*.md` verbatim and
writes a paired `output-*.md` narrative, exactly as the auto-bill-02 bill stages
did. One commit per distinguishable file; the second-to-last commit of every
stage fixes all errors; the last commit performs the remaining repository
updates.

## What changed from the auto-bill-02 sub-prompts

1. **Target.** Each prompt targets `trial-protocol/` and a Phase 1 IND/IDE
   clinical trial protocol (not a congressional bill).
2. **Template.** The base is the `trial-protocol/template` paper template
   (recolored `#00417A`), adapted to the `trial-protocol/nih-protocol` NIH-FDA
   IND/IDE template, which supersedes the paper format where they differ.
3. **Media.** The figure rule keeps full-width tables, centered ASCII, and TikZ
   `mermaidfig`, now colored with Corporate Blue `#00417A`, Professional Gray
   `#6C757D`, and Classic White `#FFFFFF` plus grayscale, instead of the bill's
   pure gray-scale palette.
4. **Sources.** The clinical, regulatory, and financial sources are
   `inputs/2030-pdac-1min-final-paper`, `inputs/21cfr312_adapt`,
   `inputs/auto-bill-02`, the `research/` markdowns, and `inputs/author_works.bib`.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
