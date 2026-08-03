# sub-prompts/part-i - the PART I schedule (ten application file sets, v4.4.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stages](https://img.shields.io/badge/Stages-5-00417A.svg)](.)
[![Output](https://img.shields.io/badge/Output-10%20email%20file%20sets-3C7DB2.svg)](../../applications)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-blue.svg)](../../../../README.md)

The five sub-prompts that build PART I: ten Phase 1 pancreatic cancer trial
funding application email file sets, no DOIs, each unique to its recipient, each
dated August 3, 2026, each in Kevin Kawchak's name as an independent scientist,
each stating the intent to partner at UC San Diego Moores Cancer Center.

## The five stages

| # | File | Output | Commit floor |
|:--|:--|:--|:--|
| 1 | [`prompt-1-recipients-and-templates.md`](prompt-1-recipients-and-templates.md) | Recipient list, cover variants, style and bib contract | 3 |
| 2 | [`prompt-2-set-a-surgical.md`](prompt-2-set-a-surgical.md) | `app-01` to `app-05`, surgical perspective | 5 |
| 3 | [`prompt-3-set-b-medical-oncology.md`](prompt-3-set-b-medical-oncology.md) | `app-06` to `app-10`, medical oncology perspective | 5 |
| 4 | [`prompt-4-email-txt-and-attachments.md`](prompt-4-email-txt-and-attachments.md) | Ten `.txt` emails, ten Overleaf zips | 3 |
| 5 | [`prompt-5-readmes-and-audit.md`](prompt-5-readmes-and-audit.md) | Every `funding/**` README, plus the audit pass | 4 |

## Invariants every stage inherits

- Palette: `patient-robot-advocacy`, **no black fill**.
- Figure spacing: `\end{appfig}` then `\vspace{-0.7cm}` then `\figcaption`.
- Captions: centred, italic, three lines maximum, balanced line lengths.
- Tables: exactly `\textwidth`, every fixed column `>{\raggedright\arraybackslash}p{...}`.
- At most five compiled pages and at most five figures per application.
- Single dashes only; `\S` for codified references; clickable DOIs.

## Files used from other directories (Rule 5)

| Source | Stage that reads it |
|:--|:--|
| [`../../../science-golden-age/`](../../../science-golden-age) | 1 (anchors), 2 and 3 (quoted mechanisms) |
| [`../../../RFA-RM-27-001-v2/`](../../../RFA-RM-27-001-v2) | 1 (bib seed), 2 and 3 (trial numbers) |
| [`../../../supplementary/`](../../../supplementary) | 2 and 3 (evidence tables), 4 (attachment ledger) |
| [`../../../daraxonrasib-llm-story.md`](../../../daraxonrasib-llm-story.md) | 2 and 3 (chronology table) |
| [`../../../tripartisan-llm-support.md`](../../../tripartisan-llm-support.md) | 2 and 3 (model-role table) |
| [`../../../potential-partners/UC-San-Diego/`](../../../potential-partners/UC-San-Diego) | 3 (application 10), 5 (README expansion) |
| [`../../../pdfs/`](../../../pdfs) | 4 (manual-attachment ledger) |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
