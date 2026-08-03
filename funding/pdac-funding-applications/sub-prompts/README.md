# sub-prompts - Process A output (10 PDAC Funding Applications, v4.4.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Sub-prompts](https://img.shields.io/badge/Sub--prompts-5%20%2B%208-00417A.svg)](.)
[![Part I](https://img.shields.io/badge/Part%20I-10%20application%20file%20sets-3C7DB2.svg)](part-i)
[![Part II](https://img.shields.io/badge/Part%20II-summary%20paper-6C757D.svg)](part-ii)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-blue.svg)](../../../README.md)

This directory holds the **thirteen stage sub-prompts** that Process A generated
from the single master prompt in
[`../prompts/prompt-apply.md`](../prompts/prompt-apply.md). Process B executes
them in order. The master prompt requires a **separate sub-prompt schedule for
PART I and PART II**, so the two schedules live in two subdirectories and never
share a stage.

The pattern is adapted from
[`trial-ind/sub-prompts`](../../../trial-ind/sub-prompts): one heading of the
form `## prompt-<name>`, then the prompt text; one stage directory per
sub-prompt; one commit per distinguishable file; the second-to-last commit of
each stage fixes errors and the last performs repository updates.

## PART I schedule - the ten application file sets

| # | Sub-prompt | Writes | Adapted from |
|:--|:--|:--|:--|
| 1 | [`prompt-1-recipients-and-templates.md`](part-i/prompt-1-recipients-and-templates.md) | [`../applications/README.md`](../applications), style and bib contract | `trial-ind/sub-prompts/prompt-1-mermaid.md` (contract-first pattern) |
| 2 | [`prompt-2-set-a-surgical.md`](part-i/prompt-2-set-a-surgical.md) | `app-01` .. `app-05` | `trial-ind/sub-prompts/prompt-2-draft-ind.md` |
| 3 | [`prompt-3-set-b-medical-oncology.md`](part-i/prompt-3-set-b-medical-oncology.md) | `app-06` .. `app-10` | `trial-ind/sub-prompts/prompt-3-full-ind.md` |
| 4 | [`prompt-4-email-txt-and-attachments.md`](part-i/prompt-4-email-txt-and-attachments.md) | ten `.txt` emails, ten Overleaf zips | new, no `trial-ind` analogue |
| 5 | [`prompt-5-readmes-and-audit.md`](part-i/prompt-5-readmes-and-audit.md) | every `funding/**` README, PART I audit | `trial-ind/sub-prompts/prompt-4-final-ind.md` |

## PART II schedule - the summary paper

| # | Sub-prompt | Writes | Figures specified |
|:--|:--|:--|:--|
| 1 | [`prompt-1-mermaid.md`](part-ii/prompt-1-mermaid.md) | [`../mermaid/`](../mermaid) | 6 |
| 2 | [`prompt-2-plantuml.md`](part-ii/prompt-2-plantuml.md) | [`../plantuml/`](../plantuml) | 3 |
| 3 | [`prompt-3-d2.md`](part-ii/prompt-3-d2.md) | [`../d2/`](../d2) | 4 |
| 4 | [`prompt-4-diagrams-python.md`](part-ii/prompt-4-diagrams-python.md) | [`../diagrams-python/`](../diagrams-python) | 3 |
| 5 | [`prompt-5-graphviz.md`](part-ii/prompt-5-graphviz.md) | [`../graphviz/`](../graphviz) | 4 |
| 6 | [`prompt-6-draft-apply.md`](part-ii/prompt-6-draft-apply.md) | [`../draft-apply/`](../draft-apply) | 20 placeholders |
| 7 | [`prompt-7-full-apply.md`](part-ii/prompt-7-full-apply.md) | [`../full-apply/`](../full-apply) | 20 drawn |
| 8 | [`prompt-8-final-apply.md`](part-ii/prompt-8-final-apply.md) | [`../final-apply/`](../final-apply) | 20 polished |

The diagram-type split is deliberately uneven, because the master prompt
requires the type to be chosen by purpose rather than by quota. Mermaid takes
the largest share because the paper's spine is chronological; PlantUML takes the
smallest because only three subjects in the paper are formal enough to need it.

## What changed from the `trial-ind` sub-prompts

- Two schedules instead of one, because the master prompt asks for two
  deliverables with different outputs: ten application file sets and one paper.
- Five diagram stages instead of one, one per machine-readable platform, each
  with its own sub-commit directory.
- Figures are **colored** on the `patient-robot-advocacy` palette rather than
  grayscale, and **no black fill** is permitted, which removes the `padark`
  token that the parent style used sparingly.
- No `publication/` subdirectory in the final stage, and no PDFs are generated.

## Files used from other directories (Rule 5)

| Source | Used by |
|:--|:--|
| [`../prompts/prompt-apply.md`](../prompts/prompt-apply.md) | Every sub-prompt; the authority all thirteen answer to |
| [`../../science-golden-age/`](../../science-golden-age) | Part I sub-prompts 1 to 3, for recipient anchors and quoted mechanisms |
| [`../../RFA-RM-27-001-v2/`](../../RFA-RM-27-001-v2) | Part I sub-prompt 2, for trial synopsis numbers and bib entries |
| [`../../supplementary/source-files/`](../../supplementary/source-files) | All Part II diagram sub-prompts, for the palette and the five vocabularies |
| [`../../potential-partners/UC-San-Diego/`](../../potential-partners/UC-San-Diego) | Part I sub-prompt 3, application 10 |
| [`../../../trial-ind/sub-prompts/`](../../../trial-ind/sub-prompts) | The stage, commit, and README conventions all thirteen follow |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
