# Stage 1, sub-prompt 3 - the seventeen section files

## Goal

One `.tex` file per section (Rule 6), one commit each. At this stage each file
carries its headings, its table shells, and a bracketed drafting instruction per
subsection that names the exact repository file stage 2 must read. No prose
argument is finished here; the point of the stage is that stage 2 never has to
guess where a number comes from.

## The roster

| File | Document | Part |
|:--|:--|:--|
| `sec-00-front.tex` | Front matter: why La Jolla, why fifteen documents, the author record, the funding position | - |
| `sec-01-sb-1188-authorization.tex` | SB 1188, California PDAC LLM Oncology Clinical Trial Site Authorization and Site Establishment Act of 2026 | I |
| `sec-02-ab-3162-patient-rights.tex` | AB 3162, California LLM Oncology Patient Rights and Robotic Safety Act of 2026 | I |
| `sec-03-sb-964-data-protection.tex` | SB 964, California Oncology Model Transparency and Clinical Data Protection Act of 2026 | I |
| `sec-04-hr-10412-federal.tex` | H. R. 10412, Independent Investigator Pancreatic Cancer Trial Site Act of 2026 | I |
| `sec-05-san-diego-municipal.tex` | San Diego Municipal Code Update, La Jolla Community Plan Area | II |
| `sec-06-title-22-regulations.tex` | California Code of Regulations, Title 22, Division 5, Chapter 15 | II |
| `sec-07-fda-compliance-guide.tex` | FDA LLM and Robotic Workflow National Compliance Guide | II |
| `sec-08-building-code.tex` | Building Code Standards for a PDAC LLM Trial Facility | III |
| `sec-09-premises-code.tex` | Premises Code | III |
| `sec-10-parking-transportation.tex` | Parking and Patient Transportation Standards | III |
| `sec-11-emergency-preparedness.tex` | Emergency Preparedness and Business Continuity Plan | III |
| `sec-12-site-activation-sops.tex` | Site Activation Checklist and Standard Operating Procedures | IV |
| `sec-13-conventional-trial-requirements.tex` | Conventional Pancreatic Cancer Clinical Trial Requirements Manual | IV |
| `sec-14-staffing-and-roles.tex` | Staffing, Role Delineation, and Move-In Plan | IV |
| `sec-15-funding-and-lobbying.tex` | Federal Funding Stewardship, Lobbying, and Legislative Engagement Plan | IV |
| `sec-16-backmatter.tex` | Abbreviations, availability, contributions, citation, references | - |

## Drafting instruction form

```
\draftnote{Populate the eleven-row roster from
\mvfile{funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex},
which fixes the \$700,000 per year and \$3,500,000 over five years frame. Do not
re-derive the budget; reuse it.}
```

Every instruction must name a path that exists. A path that does not resolve is
a stage 1 defect, not a stage 2 problem.

## Path inventory the instructions draw on

| Repository path | What stage 2 takes from it |
|:--|:--|
| `funding/pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex` | The four-source evidence table with each author's stated limitation, and the 2.4-fold against 2.0-fold chronology observation |
| `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` | The $700,000 per year, $3,500,000 over five years frame and its four layers |
| `funding/capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex` | The California limited liability company record and the $36,330 figure, described as projected |
| `funding/capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex` | The three cost benchmark rows against industry figures |
| `funding/potential-partners/UC-San-Diego/README.md` | The twelve-step feasibility sequence and the required-positioning list |
| `funding/potential-partners/UC-San-Diego/priority-steps.md` | The three positioning corrections and the sponsor-investigator determination |
| `funding/move-in/inputs/READMES/README-Physical-AI-Oncology-Clinical-Trial-Site-Complete-Documentation-Package.md` | The eleven-document San Francisco roster and its simulation evidence base |
| `funding/move-in/inputs/READMES/README-LLM-Pancreatic-Oncology-Clinical-Trial-System-Large-Documents-Funding-and-AI-Peer-Review.md` | The twenty-paper chronology with DOIs and the LLM Trust and LLM Benefit lines |
| `funding/move-in/inputs/ChemicalQDevice_Accomplishments.docx` | The author record from October 2021 forward, and the seventeen numbered references |
| `regulatory/adaption-ich-e6r3/`, `regulatory/Adaption-21-CFR-Part-50/`, `regulatory/Adaption-21-CFR-Part-312/` | The three adapted regulatory frameworks documents 07 and 13 implement |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | The Phase 1 protocol, the investigational new drug filing, and the Phase 2 successor |
| `new-trial/` | The 24-hour simulation the San Francisco package was built on, cited here only where a La Jolla requirement is derived from it |

## Acceptance

- Seventeen files exist, each with a `% sec-NN:` comment on line 1.
- Every `\draftnote` names at least one `\mvfile` path, and every path resolves.
- The stage compiles at 0 errors.

## Commit

Seventeen commits, one per section file, message
`move-in/draft: sec-NN <slug>`.
