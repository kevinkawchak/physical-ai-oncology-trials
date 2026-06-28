# trial-documents/research/industry-workflow - Phase 1 document creation workflow

[![AI sources](https://img.shields.io/badge/AI%20sources-2-2F5D7C.svg)](.)
[![Topic](https://img.shields.io/badge/Topic-PI%20%2B%20medical%20writer%20workflow-8B2E3F.svg)](.)
[![Paper](https://img.shields.io/badge/Paper-v1.0-D08770.svg)](../../draft-paper)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

Two AI sources answering the prompt in [`prompt-workflow.md`](prompt-workflow.md):
exactly how oncology trial principal investigators (PIs) and medical writers collect
patient information and create all large documents before, during, and after a Phase
1 trial.

## Files

| File | Author model | Key contribution |
|:--|:--|:--|
| [`ChatGPT-5-5-Thinking-Extended-Workflow-2026-06-26.md`](ChatGPT-5-5-Thinking-Extended-Workflow-2026-06-26.md) | ChatGPT 5.5 Thinking Extended | Detailed before/during/after document creation, the medical-writer and PI roles, and regulatory timing |
| [`Gemini-3-1-Pro-Workflow-2026-06-26.md`](Gemini-3-1-Pro-Workflow-2026-06-26.md) | Gemini 3.1 Pro | The digital data ecosystem (EHR, EDC, eCRF, eCOA/ePRO, CTCAE, RECIST, SDV, lock), the eDMS authoring flow (Veeva Vault, TransCelerate), and a bibliography (ICH E3, ICH E2F, TransCelerate, medical-writing references) |
| [`prompt-workflow.md`](prompt-workflow.md) | (prompt) | The exact research prompt used to generate the two sources |

## The four workflow stages (sections A-D of the sources)

| Stage | Activity | Large documents |
|:--|:--|:--|
| A. Data collection | EHR extraction; EDC entry (Medidata Rave, Oracle Clinical); RECIST baseline; eCRF, eCOA/ePRO; CTCAE-graded AE logs; SDV by CRAs; database lock | (feeds all documents) |
| B. Before trial | PI rationale, biostatistician rules, pharmacology; eDMS authoring | IND application, Clinical Trial Protocol, Investigator's Brochure, ICF (6th-8th grade reading level) |
| C. During trial | Safety-driven authoring; PI causality assessment | SAE/SUSAR narratives (7/15-day), DSUR, protocol amendments, dose-escalation minutes |
| D. After trial | TLF generation; narrative interpretation; PI sign-off | CSR (ICH E3), manuscripts/abstracts, lay summaries |

## Where used in the paper

These sources feed the paper Methods (before/during/after workflow, data pipeline)
and the figures `fig-06`, `fig-07`, `fig-08`, `fig-09` in
[`../../mermaid`](../../mermaid). See
[`../../draft-paper/sections/sec-03-methods.tex`](../../draft-paper/sections/sec-03-methods.tex).

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice. Sources
attributed to ChatGPT 5.5 Thinking Extended and Gemini 3.1 Pro.
