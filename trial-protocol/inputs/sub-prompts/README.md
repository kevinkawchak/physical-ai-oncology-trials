# sub-prompts - Process A output (v5.0 build)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Prompts](https://img.shields.io/badge/Sub--prompts-7-blue.svg)](.)
[![Focus](https://img.shields.io/badge/Focus-Financial%20data-darkgreen.svg)](.)

This folder is the output of **Process A**: from the single
[`../master-prompt.md`](../master-prompt.md), Claude Code generated all seven
sub-prompts that **Process B** then runs in sequence. Each sub-prompt is adapted
from the corresponding prior prompt (the `auto-bill-01` sub-prompt and, behind
it, the VVUQ-05 prompt Markdown) and re-targeted for the single-prompt v5.0
build: the `kevinkawchak/single-prompt-bill` repository, `auto-bill-02/`, a
comprehensive financial-data focus, and the three permitted media (tables,
ASCII, gray-scale Mermaid) used together, built from
`cancer-automated/.../papers/VVUQ-05/final-bill` without its `/deliverables`.

## The seven sub-prompts

| # | Sub-prompt | Runs in stage | Adapted from (prior sources) |
|:--|:--|:--|:--|
| 1 | [`prompt-1-research.md`](prompt-1-research.md) | `01-research/` | `auto-bill-01/sub-prompts/prompt-1-research.md`; `VVUQ-05/update-bill/next-steps/prompt-1-next-steps.md` |
| 2 | [`prompt-2-template-selection.md`](prompt-2-template-selection.md) | `02-template-selection/` | `auto-bill-01/sub-prompts/prompt-2-template-selection.md`; the two candidate bill file sets |
| 3 | [`prompt-3-mermaid-selection.md`](prompt-3-mermaid-selection.md) | `03-mermaid-selection/` | `auto-bill-01/sub-prompts/prompt-3-mermaid-selection.md`; `Clinical-AI-Demos/.../ai-outputs/output-01`; `VVUQ-05/update-bill/figures-bill` |
| 4 | [`prompt-4-figure-selection.md`](prompt-4-figure-selection.md) | `04-figure-selection/` | `auto-bill-01/sub-prompts/prompt-4-figure-selection.md`; `VVUQ-05/update-bill/figures-bill/prompt-figures-bill.md` |
| 5 | [`prompt-5-draft-bill.md`](prompt-5-draft-bill.md) | `draft-bill/` | `auto-bill-01/sub-prompts/prompt-5-draft-bill.md`; `VVUQ-05/draft-bill/prompt-draft-bill.md` |
| 6 | [`prompt-6-full-bill.md`](prompt-6-full-bill.md) | `full-bill/` | `auto-bill-01/sub-prompts/prompt-6-full-bill.md`; `VVUQ-05/full-bill/prompt-full-bill.md` |
| 7 | [`prompt-7-final-bill.md`](prompt-7-final-bill.md) | `final-bill/` | `auto-bill-01/sub-prompts/prompt-7-final-bill.md`; `VVUQ-05/final-bill`; the prior draft-to-full-to-final corrections |

## Convention (Rule 14)

Every sub-prompt opens with a `## prompt-<name>` heading followed by the prompt
text, mirroring the prior repository's convention. Each stage that runs a
sub-prompt writes a paired `output-*.md` narrative file, and the three bill
stages additionally re-file their own `prompt-*.md` verbatim inside their
directory, exactly as the prior bills did.

## What changed from the v4.0 sub-prompts (the update, Rule 14)

1. **Target.** Every prompt now targets `auto-bill-02/` and H. R. 9510 **v5.0**,
   repository release **v2.0.0**.
2. **Subject.** The build objective moves from "twelve deliverables converted to
   LaTeX" to a **comprehensive financial-data focus**: financial findings, an
   operative financial core (SEC. 5), and financial appendices, built from the
   Bill v3.0 measure without its deliverables.
3. **Media.** The diagram rule widens from "gray-scale Mermaid only" to **three
   media used together** - tables, ASCII, and gray-scale Mermaid - so Stage 3
   selects both the Mermaid set and the ASCII set, and Stage 4 assigns the
   correct medium to every slot.
4. **Auto-commit / PR schedule.** One commit per file pushed in real time, with
   the nine prior PRs mapped to nine labeled stage milestones inside one
   continuously updated pull request on the single permitted branch
   (`claude/funny-meitner-7ywlgc`).

## Self-learning note

The sub-prompts encode the lessons carried forward from both prior builds: the
white-space and reference-formatting corrections made from full-bill to
final-bill, the `>{\raggedright\arraybackslash}p{<width>}` table column
convention, the `\clearpage`/`\vspace`/`\hspace` page balancing, the
single-hyphen and section-symbol rules, the centered white-background `asciifig`
primitive (v3.0), and the gray-scale `mermaidfig` TikZ primitive (v4.0), now
carried side by side.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
