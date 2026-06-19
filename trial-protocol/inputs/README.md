# inputs - source materials for the protocol build

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Main documents](https://img.shields.io/badge/Main%20documents-3-00417A.svg)](.)
[![Author works](https://img.shields.io/badge/Author%20works-43%20BibTeX-6C757D.svg)](author_works.bib)

This directory holds every source the protocol build reads from. Three of them
are the designated **main documents**; one is the author's published-works
bibliography; the auto-bill-02 tree is also the **processing-workflow exemplar**
this build adapts (mermaid -> draft -> full -> final).

## Contents and how each is used (Rule 5)

| Item | Role | Used in |
|:--|:--|:--|
| [`2030-pdac-1min-final-paper/`](2030-pdac-1min-final-paper) | **Main document 1.** The author's *2030: 60 Second PDAC Robotic Whipple & Daraxonrasib Simulation* (DOI 10.5281/zenodo.20196639). Supplies the clinical subject, the eight-arm PancreSpeed platform, the 640-channel sensor stack, force caps, vessel safety zones, anastomosis ring tensions, the 32-iteration sweep, and the daraxonrasib advisory data. | All quantitative tables and figures; Study Intervention; Assessments |
| [`21cfr312_adapt/`](21cfr312_adapt) | **Main document 2.** *Adaption: 21 CFR Part 312* - the Physical AI overlay on the IND regulations (Subpart J, USL thresholds, Phase 0 simulation validation, Physical AI AE reporting, clinical-hold grounds). | Statement of Compliance; Study Intervention; Discontinuation; Oversight; Statistics |
| [`auto-bill-02/`](auto-bill-02) | **Main document 3** and the **workflow exemplar.** H. R. 9510 Bill v5.0 build; supplies the VVUQ financial/legislative framing and the mermaid/draft/full/final processing pipeline, the `mermaidfig` TikZ primitive, the table-column and ASCII conventions, and the auto-commit/PR schedule this build adapts. | Background; Oversight; every LaTeX style and figure convention |
| [`author_works.bib`](author_works.bib) | The author's 43 published LLM/oncology works (Aug 2024 - Jun 2026), evidencing established LLM trust for oncology trials. | Introduction (brief descriptions); references where directly necessary |

## The three main documents at a glance

1. **PDAC Robotic Whipple Procedure** - the clinical trial being submitted for
   funding here. The expedited 2030 sixty-second medicine-plus-surgery goal is
   reserved for a later trial phase; this Phase 1 protocol is the first step.
2. **21 CFR Part 312 adaptation** - the section-by-section physical-AI overlay
   on current IND processes that this protocol conforms to.
3. **H. R. 9510 Bill v0.5.0 / v5.0 (2026)** - the most current Physical AI VVUQ
   clinical-trial legislative and financial initiative.

## License

Released under CC BY 4.0; reproduced U.S. Government regulatory text is used
under 17 U.S.C. § 105. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
