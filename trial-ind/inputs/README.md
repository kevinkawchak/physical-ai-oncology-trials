# inputs - source materials for the IND build

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Template](https://img.shields.io/badge/Template-ReGARDD%20IND-3F3F3F.svg)](ReGARDD_IND_Template.docx)
[![Forms](https://img.shields.io/badge/Forms-FDA%201571%20%2F%203674-6C757D.svg)](FDA-1571_Instructions_R14_03-21-2023.md)
[![References](https://img.shields.io/badge/References-52%20entries-000000.svg)](references.bib)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-blue.svg)](../../README.md)

This directory holds the source materials the `trial-ind/` build draws on. None of
these files is modified by the build except `references.bib`, which is copied into
each stage and extended with new `@misc` entries in the identical format.

## Contents and how each is used (Rule 5)

| File | Used by | For |
|:--|:--|:--|
| [`ReGARDD_IND_Template.docx`](ReGARDD_IND_Template.docx) | every stage | The IND Table of Contents and section order, the Cover Letter / FDA 1571 placement rule, and the required content of each section. |
| [`FDA-1571_Instructions_R14_03-21-2023.md`](FDA-1571_Instructions_R14_03-21-2023.md) | `sec-01-fda-forms` | The field-by-field instructions for FDA Form 1571 (and the serial number / cover-sheet logic). |
| [`ReGARDD-Regulatory-Guidance-for-Academic-Research-of-Drugs-and-Devices.md`](ReGARDD-Regulatory-Guidance-for-Academic-Research-of-Drugs-and-Devices.md) | `sec-00`, `sec-03`, `sec-05` | Academic sponsor-investigator IND guidance (cover letter, pre-IND, content expectations). |
| [`references.bib`](references.bib) | `draft-ind`, `full-ind`, `final-ind` | The 52 author `@misc` references; copied into each stage's `references.bib` and extended. |
| `background/` | (optional) | Background research prompts; used only where proof of research is required. |

## Files from other directories used alongside these inputs

| Source | Used for |
|:--|:--|
| [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | The Phase 1 LLM-directed PDAC robotic daraxonrasib trial: clinical content, quantitative tables, formatting methods. |
| [`../../trial-documents/final-paper/publication`](../../trial-documents/final-paper/publication) | The acceleration argument, the figures adapted in context, and the back matter (`sec-08-references-backmatter.tex`). |
| [`../../trial-documents/inputs/llm-adoption`](../../trial-documents/inputs/llm-adoption) | The principal-investigator guidance for authoring large trial documents. |
| [`../../regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.sty`](../../regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.sty) | The paper template the `indstyle.sty` adapts (plus a back-matter section). |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
