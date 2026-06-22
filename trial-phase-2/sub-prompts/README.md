# sub-prompts - Process A output (Physical AI oncology Phase 2 trial protocol, v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Sub-prompts](https://img.shields.io/badge/Sub--prompts-4-800020.svg)](.)
[![Stage](https://img.shields.io/badge/Stages-mermaid%20%E2%86%92%20draft%20%E2%86%92%20full%20%E2%86%92%20final-6B6B6B.svg)](.)
[![Protocol](https://img.shields.io/badge/Protocol-Phase%202%20Randomized%20IND%2FIDE-800020.svg)](.)
[![Predicate](https://img.shields.io/badge/Predicate-Phase%201%20v1.0.0-6B6B6B.svg)](../../trial-protocol)

This folder is the output of **Process A**: from the single master prompt filed
verbatim in [`../prompts/prompt-protocol.md`](../prompts/prompt-protocol.md),
Claude Code generated the four sub-prompts that **Process B** then runs in
sequence to grow the Phase II protocol from Mermaid figures to draft, full, and
final LaTeX, and then to the author-edited `final-protocol/publication` paper.
Each sub-prompt is adapted from the corresponding Phase I `trial-protocol` stage
and re-targeted to the *Phase 2, Multicenter, Randomized, Controlled Clinical
Trial Protocol of On-Premises LLM-Directed Robotic Pancreaticoduodenectomy
(Whipple) with Perioperative Daraxonrasib (RMC-6236)*.

## The four sub-prompts

| # | Sub-prompt | Runs in stage | Adapted from (trial-protocol) |
|:--|:--|:--|:--|
| 1 | [`prompt-1-mermaid.md`](prompt-1-mermaid.md) | `../mermaid/` | `sub-prompts/prompt-1-mermaid.md` |
| 2 | [`prompt-2-draft-protocol.md`](prompt-2-draft-protocol.md) | `../draft-protocol/` | `sub-prompts/prompt-2-draft-protocol.md` |
| 3 | [`prompt-3-full-protocol.md`](prompt-3-full-protocol.md) | `../full-protocol/` | `sub-prompts/prompt-3-full-protocol.md` |
| 4 | [`prompt-4-final-protocol.md`](prompt-4-final-protocol.md) | `../final-protocol/` (+ `publication/`) | `sub-prompts/prompt-4-final-protocol.md` |

## Convention

Every sub-prompt opens with a `## prompt-<name>` heading followed by the prompt
text. Each stage that runs a sub-prompt files its own `prompt-*.md` verbatim and
writes a paired `output-*.md` narrative. One commit per distinguishable file; the
second-to-last commit of every stage fixes all errors; the last commit performs
the remaining repository updates.

## What changed from the Phase I sub-prompts

1. **Phase and design.** Each prompt targets `trial-phase-2/` and a Phase 2,
   multicenter, randomized 1:1, controlled study (n = 220, eight sites, primary
   progression-free survival, hazard ratio 0.60), rather than a Phase 1 single-arm
   dose-finding and early-feasibility study.
2. **Palette.** The figure rule recolors to the five-step Phase II palette
   (Burgundy `#800020`, Charcoal `#2E2E2E`, Slate `#6B6B6B`, Mist `#C9C9C9`, Cloud
   `#F5F5F5`) with Burgundy as the document color, replacing the Phase I Corporate
   Blue scheme.
3. **Co-investment.** A new figure and table family and a dedicated oversight
   subsection carry the Patient-Aligned Co-Investment Facility and the capital
   firewall.
4. **Publication.** The final stage additionally produces
   `final-protocol/publication/` as the author-edited paper URL directory.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
