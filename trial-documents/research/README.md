# trial-documents/research - Background AI sources for the paper

[![Sources](https://img.shields.io/badge/AI%20sources-4-2F5D7C.svg)](.)
[![Topics](https://img.shields.io/badge/Topics-DocTypes%20%2B%20Workflow-8B2E3F.svg)](.)
[![Paper](https://img.shields.io/badge/Paper-v1.0-D08770.svg)](../draft-paper)
[![Repository](https://img.shields.io/badge/Repository-v4.2.0-2F5D7C.svg)](../../README.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

This directory holds the four AI research sources that ground the paper *Phase 1
Pancreatic Cancer Trial Efficient LLM Document Generations* (paper v1.0, repository
v4.2.0). Two subdirectories cover two questions: which Phase 1 steps depend on long
documents (decision making), and how principal investigators (PIs) and medical
writers actually create those documents before, during, and after Phase 1.

The author has authority to over-rule any of the four AI sources to generate
accurate and relevant context, as stated in the master prompt
([`../prompts/prompt-paper.md`](../prompts/prompt-paper.md)).

## Subdirectories

| Subdirectory | Question | AI sources |
|:--|:--|:--|
| [`document-types/`](document-types) | Which Phase 1 steps and phase transitions rely on long documents, and whether faster authoring speeds the trial | ChatGPT 5.5 Thinking Extended; Gemini 3.1 Pro |
| [`industry-workflow/`](industry-workflow) | How PIs and medical writers collect data and create large documents before, during, and after Phase 1 | ChatGPT 5.5 Thinking Extended; Gemini 3.1 Pro |

## Where these sources are used in the paper

| Source | Used in | What it supplies |
|:--|:--|:--|
| `document-types/ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md` | Methods, Results | The six ACCELERATION targets; the hard / protocol-defined / decision gate taxonomy; the IND, clinical-hold, cohort-review, and CSR/NDA-BLA clocks |
| `document-types/Gemini-3-1-Pro-DocTypes-2026-06-26.md` | Introduction, Discussion | The three timeline buckets (clinical/operational, administrative/prep, regulatory review); the white-space-between-phases argument |
| `industry-workflow/ChatGPT-5-5-Thinking-Extended-Workflow-2026-06-26.md` | Methods | The detailed before/during/after Phase 1 document creation workflow |
| `industry-workflow/Gemini-3-1-Pro-Workflow-2026-06-26.md` | Methods | The data-collection pipeline (EHR -> EDC -> eCRF -> eCOA/ePRO -> SDV -> lock) and the medical-writer authoring roles |

## Color scheme of derived figures

The figures derived from these sources use the five-step palette deep maroon
`#8B2E3F`, steel blue `#2F5D7C`, terracotta `#D08770`, light blue `#BFD7EA`, and
near-white `#F4F7F9`, plus grayscale (see [`../mermaid`](../mermaid)).

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice. The two AI
sources per subdirectory are reproduced for research grounding and are attributed
to ChatGPT 5.5 Thinking Extended and Gemini 3.1 Pro.
