# La Jolla Move-In: Pancreatic Oncology Clinical Trial Site Complete Documentation Package (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](final-move-in)
[![Site](https://img.shields.io/badge/Site-La%20Jolla%2C%20San%20Diego-3C7DB2.svg)](final-move-in/sections/sec-05-san-diego-municipal.tex)
[![Staff](https://img.shields.io/badge/Staff-CEO%20%2B%2010%20coworkers-00417A.svg)](final-move-in/sections/sec-14-staffing-and-roles.tex)
[![Award](https://img.shields.io/badge/Award-%24700%2C000%20%C3%97%205%20years-6C757D.svg)](final-move-in/sections/sec-15-funding-and-lobbying.tex)
[![Trial](https://img.shields.io/badge/Trial-Phase%201%20PDAC%20robotic%20Whipple-00417A.svg)](../../trial-protocol)
[![Stages](https://img.shields.io/badge/Stages-draft%20%E2%86%92%20full%20%E2%86%92%20final-6C757D.svg)](sub-prompts)
[![Diagrams](https://img.shields.io/badge/Diagrams-none%2C%20by%20Rule%203-9AA1A8.svg)](sub-prompts)
[![Compiler](https://img.shields.io/badge/Compiler-pdfLaTeX%20%2B%20BibTeX-6C757D.svg)](final-move-in)
[![Template DOI](https://img.shields.io/badge/Template%20DOI-10.5281%2Fzenodo.19176370-blue.svg)](https://doi.org/10.5281/zenodo.19176370)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI%20v1.0-10.5281%2Fzenodo.22216519-blue.svg)](https://doi.org/10.5281/zenodo.22216519)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-6C757D.svg)](https://orcid.org/0009-0007-5457-8667)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../README.md)

**Fifteen documents that stand up California's first pancreatic ductal
adenocarcinoma large language model oncology clinical trial site, in La Jolla,
for a staff of eleven, against a federal award of $700,000 per year for five
years.**

The predecessor package, *Physical AI Oncology Clinical Trial Site Complete
Documentation Package* (v2.9.0, March 24, 2026, DOI
[10.5281/zenodo.19176370](https://doi.org/10.5281/zenodo.19176370)), described a
24-hour San Francisco site serving 168 patients across 15 cancer types with 29
robot instances. This package is narrower and harder: one disease, one Phase 1
protocol, one building, eleven named people, and a funded budget that has to
close. Everything the site does has to satisfy conventional pancreatic cancer
trial requirements first, and the large language model and robotic workflow
second.

---

## 1. What is in this directory

```
funding/move-in/
  README.md                    this build hub
  inputs/                      the three source artifacts the prompt names
    READMES/                   one README per artifact, plus the abstract corpus
  prompts/                     prompt-move-in.md (verbatim) + output-move-in.md
  sub-prompts/
    draft-move-in/             stage 1 schedule, 5 sub-prompts
    full-move-in/              stage 2 schedule, 5 sub-prompts
    final-move-in/             stage 3 schedule, 5 sub-prompts
  draft-move-in/               main.tex, movestyle.sty, references.bib, sections/, zip
    sections/                  sec-00 front, sec-01..sec-15, sec-16 back matter
  full-move-in/                the same set, fully written
    sections/
  final-move-in/               the same set, senior-author polished (no publication/)
    sections/
```

## 2. Milestone schedule (one pull request, updated as each lands)

| Milestone | Stage | Output directory | Status |
|:--|:--|:--|:--|
| M0 | Bootstrap: prompt of record, sub-prompt schedule, input READMEs | [`prompts/`](prompts), [`sub-prompts/`](sub-prompts), [`inputs/`](inputs) | complete |
| M1 | Stage 1, draft-move-in | [`draft-move-in/`](draft-move-in) | complete |
| M2 | Stage 2, full-move-in | [`full-move-in/`](full-move-in) | complete |
| M3 | Stage 3, final-move-in | [`final-move-in/`](final-move-in) | complete |
| M4 | Release v4.7.0 | root `README.md`, `CHANGELOG.md`, `releases.md` | complete |

## 3. The fifteen documents, and why there are fifteen

The predecessor carried eleven. Four were added, because the master prompt asks
for three things the San Francisco package never had to answer: conventional
pancreatic cancer trial requirements, a lobbying and federal funding position,
and a named staff of eleven.

| # | Document | Part | Instrument |
|:--|:--|:--|:--|
| 01 | SB 1188, California PDAC LLM Oncology Clinical Trial Site Authorization and Site Establishment Act of 2026 | I | State bill |
| 02 | AB 3162, California LLM Oncology Patient Rights and Robotic Safety Act of 2026 | I | State bill |
| 03 | SB 964, California Oncology Model Transparency and Clinical Data Protection Act of 2026 | I | State bill |
| 04 | H. R. 10412, Independent Investigator Pancreatic Cancer Trial Site Act of 2026 | I | Federal bill |
| 05 | San Diego Municipal Code Update, La Jolla Community Plan Area | II | City regulation |
| 06 | California Code of Regulations, Title 22, Division 5, Chapter 15 | II | State regulation |
| 07 | FDA LLM and Robotic Workflow National Compliance Guide | II | Federal guidance map |
| 08 | Building Code Standards for a PDAC LLM Trial Facility | III | Building code |
| 09 | Premises Code | III | Premises code |
| 10 | Parking and Patient Transportation Standards | III | Premises code |
| 11 | Emergency Preparedness and Business Continuity Plan | III | Operations |
| 12 | Site Activation Checklist and Standard Operating Procedures | IV | Operations |
| 13 | Conventional Pancreatic Cancer Clinical Trial Requirements Manual | IV | Operations |
| 14 | Staffing, Role Delineation, and Move-In Plan | IV | Operations |
| 15 | Federal Funding Stewardship, Lobbying, and Legislative Engagement Plan | IV | Operations |

Document 13 is the load-bearing addition. A reader who deletes every large
language model and every robot from the site still holds a complete conventional
Phase 1 pancreatic cancer requirements manual. That is the order the master
prompt sets: conventional requirements first, then the workflow that has to earn
its place inside them.

## 4. The eleven people

| # | Role | Award FTE | Why the site cannot open without it |
|:--|:--|:--|:--|
| 1 | Chief Executive Officer and sponsor representative | 0.20 | Holds the investigational new drug application and signs every regulatory submission |
| 2 | Site principal investigator, hepatopancreatobiliary surgery | 0.10 | Performs the pancreaticoduodenectomy and owns participant safety |
| 3 | Sub-investigator, gastrointestinal medical oncology | 0.10 | Owns perioperative daraxonrasib dosing and toxicity management |
| 4 | Director of clinical operations | 0.40 | Owns the activation checklist and the standard operating procedure index |
| 5 | Lead clinical research coordinator | 1.00 | The only full-time award-funded role; runs consent, visits, and source documents |
| 6 | Regulatory affairs and quality manager | 0.45 | Owns institutional review board correspondence, safety reporting, and the archive |
| 7 | Investigational drug pharmacist | 0.20 | Owns accountability under 21 CFR 312.57 and 312.62 |
| 8 | Robotics and physical AI systems engineer, site safety officer | 0.55 | Owns the robot envelopes, interlocks, and the emergency stop chain |
| 9 | LLM verification and model governance lead | 0.40 | Owns the advisory-only boundary, the audit trail, and model change control |
| 10 | Data manager and biostatistician | 0.30 | Owns the 3+3 escalation decisions and the data lock |
| 11 | Research nurse and participant navigator | 0.25 | Owns the participant-facing pathway and transportation |
| | **Total** | **3.95** | |

The chief executive is not a clinical investigator. He is chief executive of the
sponsor and holds 100 percent of it, so every trigger in 21 CFR 54.2 would fire
if he held both roles. The investigator role sits wholly with the site.

## 5. The money

| Line | Per year | Five years |
|:--|:--|:--|
| Personnel, 3.95 award-funded full-time equivalents across eleven roles | $521,000 | $2,605,000 |
| Premises and occupancy, La Jolla clinical suite | $96,000 | $480,000 |
| Physical AI and on-premises compute | $38,000 | $190,000 |
| Regulatory, institutional review board, and archive | $21,000 | $105,000 |
| Participant costs not covered by standard of care | $14,000 | $70,000 |
| Verification, monitoring, and audit | $10,000 | $50,000 |
| **Total direct cost** | **$700,000** | **$3,500,000** |

The frame is reused verbatim from
[`../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`](../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex)
and is not re-derived. Drug supply, operating theater time, pathology, and
bioanalytical support sit in a contributed non-federal column that carries no
dollar figure, because no agreement exists and an invented cost-share number is
worse than none.

## 6. Measured result

| Metric | Draft | Full | Final |
|:--|:--|:--|:--|
| Errors | 0 | 0 | 0 |
| Overfull boxes | 0 | 0 | 0 |
| Underfull boxes | 0 | 0 | 0 |
| Undefined citations and references | 0 | 0 | 0 |
| Bibliography entries printed | 2 | 76 | 76 of 76 |
| Pages | 27 | 71 | 67 |
| Contents pages | 4 | 4 | 3 |
| Pages under twelve body lines | 5 | 4 | 0 |
| Pages ending on a heading | not measured | 5 | 0 |
| Tables at the body measure | 17 shells | 56 | 56 |
| Fixed columns carrying the ragged prefix | 51 of 51 | 130 of 130 | 130 of 130 |
| Source characters | 60,992 | 167,972 | 175,256 |

Every stage was compiled with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`
before its commit, and each `.zip` was unpacked into an empty directory and
compiled again, so the author opens any of the three in Overleaf and fixes
nothing. The predecessor's `all_documents.tex` is 150,972 characters; the final
stage is 175,256, a ratio of 1.16, and the difference is structural: this
package carries 56 full-width tables where the predecessor carries none.

## 7. Build method

Three stages, five sub-prompts each, one directory per stage. No diagram
platform stages, because Rule 3 forbids diagrams; where the parent build would
have drawn a figure, this one writes a table.

| Stage | Directory | What it produces | Distinguishing feature |
|:--|:--|:--|:--|
| 1 | [`draft-move-in/`](draft-move-in) | The compiling skeleton | Every subsection carries a bracketed drafting instruction naming an exact repository path |
| 2 | [`full-move-in/`](full-move-in) | The full package | Every instruction is answered from the file it names; every table is populated and set to the body measure |
| 3 | [`final-move-in/`](final-move-in) | The polished package | `\clearpage` discipline, the spacing vocabulary, the dialect audit, and two context verification passes |

Each stage emits `main.tex`, `movestyle.sty`, `references.bib`,
`sections/*.tex`, two READMEs, and its own Overleaf `.zip`, rebuilt from the
same sources in the same pass as the compile.

## 8. Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`inputs/Physical-AI-Oncology-Clinical-Trial-Site-Complete-Documentation-Package.zip`](inputs) | The cover theme, the table of contents form, the `\part` per document convention, and the 150,972-character budget the package is written against |
| [`inputs/ChemicalQDevice_Accomplishments.docx`](inputs) | The author record in document 15 and the front matter: the California limited liability company date, the 2026 count of AI-generated papers, and the chronology |
| [`inputs/LLM-Pancreatic-Oncology-Clinical-Trial-System-...pptx`](inputs) | The twenty-paper readiness chronology in documents 07 and 13 |
| [`../pdac-funding-applications/final-apply/applystyle.sty`](../pdac-funding-applications/final-apply/applystyle.sty) | The typography invariants, the column types, `\apptable`, `\tabcap`, `\bmhead`, `\keywords`, `\orcidicon`, and the `\UrlBreaks` re-assertion |
| [`../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex`](../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex) | The four-source evidence table with each author's stated limitation, in document 13 |
| [`../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`](../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex) | The $700,000 per year budget frame, in documents 14 and 15 |
| [`../capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex`](../capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex) | The company record and the $36,330 figure, described as projected |
| [`../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`](../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex) | The three cost benchmark rows |
| [`../capitalization-plan/final-capital/capstyle.sty`](../capitalization-plan/final-capital/capstyle.sty) | The `unsrturl` bibliography style, which prints and links DOI and URL fields |
| [`../potential-partners/UC-San-Diego/`](../potential-partners/UC-San-Diego) | The feasibility sequence, the required positioning, and the sponsor-investigator determination |
| [`../science-golden-age/`](../science-golden-age) | The independent scientist and novel performer policy basis cited in documents 04 and 15 |
| [`../../regulatory/`](../../regulatory) | The three adapted frameworks implemented by documents 07 and 13 |
| [`../../trial-protocol/`](../../trial-protocol), [`../../trial-ind/`](../../trial-ind) | The Phase 1 protocol and the investigational new drug filing the site is built to run |
| [`../../new-trial/`](../../new-trial) | The 24-hour simulation the predecessor package was built on, cited where a La Jolla requirement derives from it |

## 9. What this package is not

Nothing here is enacted, filed, or agreed. SB 1188, AB 3162, SB 964 and
H. R. 10412 are independent drafts and no bill by those numbers is before any
legislature. The La Jolla site is not leased, permitted, or built. No
institutional review board has reviewed anything in this package. No agreement
of any kind exists with UC San Diego, with Moores Cancer Center, with a drug
developer, or with a robotic surgery vendor. Daraxonrasib is investigational and
already in Phase 3 evaluation, and is nowhere described as first in human; the
supportable novelty claim concerns the integrated surgical and advisory
workflow. The ten coworkers are roles, not hires.

## 10. License

Creative Commons Attribution 4.0 International (CC BY 4.0).
