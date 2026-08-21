# full-move-in/sections - the seventeen written section files (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Files](https://img.shields.io/badge/Files-17-00417A.svg)](.)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](.)
[![Tables](https://img.shields.io/badge/Tables-56-3C7DB2.svg)](.)
[![Instructions left](https://img.shields.io/badge/Drafting%20instructions-0-brightgreen.svg)](.)
[![Characters](https://img.shields.io/badge/Characters-167%2C972-6C757D.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../../README.md)

One `.tex` per section, one commit each. Every bracketed drafting instruction
from stage 1 has been answered and deleted.

## The roster, with what each file now carries

| File | Sections | Tables | The load-bearing content |
|:--|:--|:--|:--|
| `sec-00-front.tex` | 5 | 5 | The predecessor comparison, the fifteen-document index, the nineteen-row site parameter table, the author record, the seven-reader guide |
| `sec-01-sb-1188-authorization.tex` | 6 | 3 | Ten cited findings, nine definitions, the seven-step authorization clock with the consequence of departmental silence |
| `sec-02-ab-3162-patient-rights.tex` | 6 | 3 | Seven participant rights, the six-row disclosure evidence table, seven measurable safety values |
| `sec-03-sb-964-data-protection.tex` | 6 | 3 | Seven definitions, five transparency obligations, a uniform six-year retention across five record classes |
| `sec-04-hr-10412-federal.tex` | 6 | 3 | Eight operative sections, the appropriation arithmetic, the five-item review timeline report |
| `sec-05-san-diego-municipal.tex` | 6 | 3 | The five-zone table, six conditional use findings, seven operating standards, the nine-permit sequence |
| `sec-06-title-22-regulations.tex` | 6 | 3 | Nine application items, ten staffing minimums, the two registers |
| `sec-07-fda-compliance-guide.tex` | 6 | 5 | The six-component pathway table, the fourteen-row correction map, the four-part advisory-only boundary, the six-step pre-submission sequence |
| `sec-08-building-code.tex` | 6 | 5 | Six occupancy rows, six structural provisions, the seven-space air change table, seven machine room provisions, eight commissioning gates |
| `sec-09-premises-code.tex` | 6 | 3 | The four-zone table, eight envelope and interlock requirements, five waste streams |
| `sec-10-parking-transportation.tex` | 6 | 4 | The 46-stall schedule derived from visit concurrency, six geometry requirements, five post-procedure transport rules |
| `sec-11-emergency-preparedness.tex` | 6 | 4 | Five command functions, the six-system backup table, five incident classes |
| `sec-12-site-activation-sops.tex` | 6 | 5 | Five activation gates, a twenty-item checklist, a twenty-procedure index in three classes, eleven training rows |
| `sec-13-conventional-trial-requirements.tex` | 7 | 7 | Twelve eligibility criteria, four dose levels, four grading instruments, eight monitoring obligations, the four-source evidence table, the three-row cost benchmark |
| `sec-14-staffing-and-roles.tex` | 6 | 6 | The eleven-role roster at 3.95 full-time equivalents, qualifications and delegated tasks, the firewall, the cost table, the forty-six-week move-in |
| `sec-15-funding-and-lobbying.tex` | 6 | 6 | The six-line award, seven stewardship controls, the seven-row lobbying authority table, four engagement plans, twenty-four deposited works |
| `sec-16-backmatter.tex` | 5 | 1 | Thirty abbreviation pairs, availability, contributions and the 21 CFR part 54 position, citation, references |

## Cross-document consistency, verified

A package of fifteen documents fails when two of them state the same number
differently. Each value below is owned by exactly one section, and every other
section that uses it cites rather than restates it.

| Value | Owned by | Cited by | Verified |
|:--|:--|:--|:--|
| Deficiency and deviation classes, and their penalty ranges | `sec-01` §5 | `sec-03` §6, `sec-06` §5, `sec-12` §6 | yes |
| Record retention, six years from lock | `sec-03` §5 | `sec-06` §6, `sec-09` §2, `sec-13` §6 | yes |
| Robot envelope and interlock values | `sec-09` §3 | `sec-02` §5 | yes, seven values |
| Hours of operation | `sec-05` §5 | `sec-09` §2, `sec-12`, `sec-14` §4 | yes |
| Waste pickup windows | `sec-09` §5 | `sec-05` §5 | yes |
| Staffing minimums and qualifications | `sec-14` §1 and §2 | `sec-06` §3 | yes, word for word |
| Training hours | `sec-12` §5 | `sec-06` §3, `sec-11` §6 | yes |
| Air changes and pressure relationships | `sec-08` §3 | `sec-09` §4, `sec-13` §5 | yes |
| Accessible path of travel | `sec-10` §3 | `sec-09` §6 | yes, one route described once |
| The $700,000 budget frame | `sec-15` §1 | `sec-04` §4, `sec-14` §5 | yes, sums checked |
| Site parameters | `sec-00` §3 | `sec-05`, `sec-08`, `sec-09`, `sec-10`, `sec-13` | yes, nineteen rows |
| Activation and commissioning boundary | `sec-08` §6 gate C8 and `sec-12` §1 gate G1 | each other | yes, one boundary |

## Arithmetic checked in this stage

| Check | Result |
|:--|:--|
| Eleven full-time equivalent fractions sum to the stated total | 0.20 + 0.10 + 0.10 + 0.40 + 1.00 + 0.45 + 0.20 + 0.55 + 0.40 + 0.30 + 0.25 = 3.95 |
| Eleven charged salaries sum to the personnel line | $521,000 |
| Six budget lines sum to the annual direct cost | $521,000 + $96,000 + $38,000 + $21,000 + $14,000 + $10,000 = $700,000 |
| Five years of the annual direct cost | $3,500,000 |
| Six stall classes sum to the site parameter total | 22 + 12 + 4 + 3 + 3 + 2 = 46 |
| Eight robot types sum to the instance count | 2 + 2 + 1 + 2 + 2 + 1 + 2 + 2 = 14 |
| Three escalation levels at up to six participants | 18, the stated ceiling |
| Federal appropriation, five cohorts of five sites | 5 × $17,500,000 = $87,500,000 |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../../draft-move-in/sections/`](../../draft-move-in/sections) | The 96 drafting instructions, each answered and deleted here |
| [`../../../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex`](../../../pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex) | `sec-13` §7, the four-source evidence table and the 2.4-fold against 2.0-fold paragraph |
| [`../../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`](../../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex) | `sec-14` §5 and `sec-15` §1, the budget frame |
| [`../../../capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex`](../../../capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex) | `sec-00` §4 and `sec-15` §6, the company record and the projected $36,330 figure |
| [`../../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`](../../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex) | `sec-13` §7, the three cost benchmark rows |
| [`../../../potential-partners/UC-San-Diego/`](../../../potential-partners/UC-San-Diego) | `sec-00` §1 and `sec-15` §5, the feasibility posture and the three positioning corrections |
| [`../../inputs/READMES/`](../../inputs/READMES) | `sec-15` §6, the deposited work table; `sec-00` §1, the predecessor comparison |
| [`../../inputs/`](../../inputs) | `sec-00` §4 and `sec-15` §6, the author record from October 2021 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
