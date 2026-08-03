# RFA-RM-27-001-v2 - Clinical Trial Funding Application v2.0 (source)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Version](https://img.shields.io/badge/Application-v2.0-00417A.svg)](.)
[![Opportunity](https://img.shields.io/badge/Opportunity-RFA--RM--27--001-3C7DB2.svg)](.)
[![Format](https://img.shields.io/badge/Format-LaTeX%20source%20zip-6C757D.svg)](.)
[![Sections](https://img.shields.io/badge/Sections-13-6C757D.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../README.md)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21317266-blue.svg)](https://doi.org/10.5281/zenodo.21317266)

The **LaTeX source** of the second NIH funding application. The compiled PDF is
in [`../pdfs/`](../pdfs).

> **7/12: (Clinical Trial Funding Application v2.0)** *RFA-RM-27-001, Kawchak K.
> The application proposes a first-in-human, combined drug-device investigation
> of perioperative daraxonrasib and an eight-arm robotic pancreaticoduodenectomy,
> with an on-premises, repository-pinned LLM acting only as a second-opinion
> advisory system.*
>
> Kawchak, K. (2026). Clinical Trial Funding Application v2.0, RFA-RM-27-001,
> Kawchak K. Zenodo. https://doi.org/10.5281/zenodo.21317266

## Contents of `LaTeX Source Files.zip`

| File | What it is |
|:--|:--|
| `main.tex` | Document root; one `\input` per section, with a template-appendix switch |
| `clinicaltrialgrant.sty` | Style: fillable form fields, `\cell` and `\field` macros, page furniture |
| `clinicaltrialgrant-unsrtnat.bst` | Bibliography style |
| `references.bib` | 58 entries: the opportunity, NIH policy, the author's prior works, clinical evidence |
| `sections/00-cover-and-transmittal.tex` | Cover page and transmittal |
| `sections/01-opportunity-and-organization.tex` | Opportunity and applicant organization |
| `sections/02-project-summary.tex` | Project summary and narrative |
| `sections/03-research-strategy.tex` | Significance, innovation, approach |
| `sections/04-clinical-trial-synopsis.tex` | The trial synopsis: phase, population, enrollment, endpoints, follow-up |
| `sections/05-human-subjects-and-regulatory.tex` | Human subjects, inclusion, regulatory pathway |
| `sections/06-statistics-safety-and-operations.tex` | Statistics, safety, operations |
| `sections/07-data-management-and-sharing.tex` | Data management and sharing plan |
| `sections/08-facilities-and-resources.tex` | Facilities and resources |
| `sections/09-budget-milestones-and-sustainability.tex` | Budget: $700,000 per year, $3,500,000 total, $0 cost share |
| `sections/10-investigator-profile.tex` | Investigator profile |
| `sections/11-assurances-and-attachments.tex` | Assurances and attachments |
| `sections/99-template-notes.tex` | Drafting notes, switchable off |
| `sections/ai-peer-review-context.tex` | Centralized AI provenance and peer-review wording |

## What v4.4.0 takes from this directory (Rule 5)

| Item taken | Where it is used |
|:--|:--|
| Trial synopsis: Phase 1, open-label, single-arm, up to 18 treated participants, 28-day screening, 30-day and 90-day safety, 24-month OS | §3 or §4 of all ten application file sets in [`../pdac-funding-applications/applications`](../pdac-funding-applications/applications) |
| Co-primary endpoint and estimand wording, including retention of all treated participants in safety summaries | §4 of applications 01, 03, 07, 08; §2 of application 08 |
| Budget frame: $700,000 per year, $3,500,000 total, $0 cost share, $0 program income | §5 of applications 01, 03, 04, 06, 08, 09; split across two phases in application 05; scoped to 36 months in application 02 |
| `references.bib` entries `phase1ind`, `onpremwhippl`, `rasolute302`, and the author-work keys | The seed for [`../pdac-funding-applications/applications/references.bib`](../pdac-funding-applications/applications/references.bib) |
| The centred form-field cover theme | Deliberately **varied from**, not reused: the summary paper's cover and all ten application covers use different furniture, per the master prompt |

## Compiling

`pdflatex main` then `bibtex main` then `pdflatex main` twice. The form fields
require a PDF viewer that supports AcroForm.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
