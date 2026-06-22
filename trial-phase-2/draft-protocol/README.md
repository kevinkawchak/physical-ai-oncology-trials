# draft-protocol - Stage 2 (scaffold) (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20draft%20(scaffold)-800020.svg)](../sub-prompts/prompt-2-draft-protocol.md)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-800020.svg)](sections)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6B6B6B.svg)](main.tex)
[![Repo](https://img.shields.io/badge/Repo-v4.1.0-800020.svg)](../../)

## What this stage is

This directory is the output of **Stage 2** (sub-prompt
[`../sub-prompts/prompt-2-draft-protocol.md`](../sub-prompts/prompt-2-draft-protocol.md)):
the **draft (scaffold)** of the *Phase 2, Multicenter, Randomized, Controlled*
protocol. It is the earlier skeleton from which the full and final stages were
rendered. Every NIH section is present and ordered, but the body is not full prose:
each content slot carries a bracketed `[DRAFTING INSTRUCTION]` (the `\draftinstr`
macro) that names (a) the Phase 2 publication section and the Phase 1 model file to
follow, (b) the exact `physical-ai-oncology-trials` source files to use, and (c)
the figure number and table the full stage must render there. The project compiles
in Overleaf as committed, with the Burgundy `#800020` Phase 2 palette.

The locked Phase 2 design that the scaffold points to is: multicenter (eight
high-volume academic HPB centers), randomized 1:1, parallel-group, controlled,
open-label with blinded independent central review; n = 220 (110 per arm); primary
endpoint progression-free survival (HR 0.60, 85 percent power, two-sided alpha
0.05, about 140 events, one group-sequential interim); a key secondary hierarchy
of overall survival, R0 rate, ISGPS grade B/C fistula, major pathologic response,
and ctDNA clearance; Arm A perioperative daraxonrasib at the RP2D of 300 mg once
daily plus the on-premises LLM-directed eight-arm robotic Whipple, Arm B modified
FOLFIRINOX plus standard pancreaticoduodenectomy; the upgraded Phase 0 gate (USL
greater than or equal to 8.0, at least 5000 sims, at least 3 frameworks,
sim-to-real less than 1.5 mm and less than 0.4 N); and the Patient-Aligned
Co-Investment Facility behind a capital firewall.

## Files

```
draft-protocol/
  main.tex                  DRAFT cover (Draft scaffold, v1.1.0), clickable TOC, one \input per section
  protostyle.sty            Phase 2 palette (Burgundy #800020, Charcoal #2E2E2E, Slate #6B6B6B,
                            Mist #C9C9C9, Cloud #F5F5F5); mermaidfig, tabularx L/Y/C/R columns,
                            TikZ ORCID, \draftinstr marker, senior-author formatting
  references.bib            daraxonrasib (5), three main documents, Phase 1 predicate, author works,
                            clinical refs, randomized-trial / reporting methods, FDA / CFR / standards
  sections/                 sec-00 .. sec-12 (13 NIH sections, scaffold)
  prompt-draft-protocol.md  this stage's sub-prompt, verbatim
  output-draft-protocol.md  narrative output
```

## NIH sections (one `sections/*.tex` per section)

| File | NIH section | Figures / tables it points to |
|:--|:--|:--|
| `sec-00-compliance.tex` | Statement of Compliance | Figure 1 |
| `sec-01-summary.tex` | Protocol Summary (Synopsis, Schema, SoA) | Figure 2; `tab:soa` |
| `sec-02-introduction.tex` | Introduction (Rationale, Background, Risk/Benefit) | Figures 3-6; `tab:concerns`, `tab:coinvest` |
| `sec-03-objectives.tex` | Objectives and Endpoints | Figure 7; `tab:objend` |
| `sec-04-design.tex` | Study Design | Figures 8, 9 |
| `sec-05-population.tex` | Study Population | Figure 10 |
| `sec-06-intervention.tex` | Study Intervention | Figures 11-14; `tab:arms`, `tab:sensors` |
| `sec-07-discontinuation.tex` | Intervention and Participant Discontinuation/Withdrawal | (none) |
| `sec-08-assessments.tex` | Study Assessments and Procedures | Figures 15-17 |
| `sec-09-statistics.tex` | Statistical Considerations | Figure 18; `tab:power`, `tab:secendpts` |
| `sec-10-oversight.tex` | Regulatory, Ethical, and Oversight Considerations | Figures 19-21 |
| `sec-11-additional.tex` | Additional Considerations, Abbreviations, Amendment History | Figure 22; `tab:jurisdictions`, `tab:amend`, `tab:abbrev` |
| `sec-12-references-backmatter.tex` | References and Back Matter | bibliography (ieeetr) |

## Files from other directories used here

| Source | Used for |
|:--|:--|
| `../final-protocol/publication/protostyle.sty` | shared Phase 2 style, copied here |
| `../final-protocol/publication/references.bib` | shared bibliography, copied here |
| `../final-protocol/publication/sections/sec-00 .. sec-12` | the Phase 2 publication sections each scaffold slot renders to |
| `../mermaid/fig-01 .. fig-22` | bracketed TikZ-figure pointers (Figures 1-22) in every section |
| `../../trial-protocol/draft-protocol/sections/sec-00 .. sec-12` | the Phase 1 draft model files whose conventions each section follows |
| `../../trial-protocol/nih-protocol/01 .. 10` | NIH section order and required content |
| `../../trial-protocol/inputs/2030-pdac-1min-final-paper` | clinical, device, and telemetry data and tables |
| `../../trial-protocol/inputs/21cfr312_adapt` | Subpart J overlay, IND content, AE reporting, holds, oversight |
| `../../trial-protocol/inputs/auto-bill-02` | the H.R. 9510 / co-investment capital-firewall and ten-gate logic |
| `../../trial-protocol/inputs/author_works.bib` | directly relevant author works |

## Compile (Overleaf, pdfLaTeX)

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
