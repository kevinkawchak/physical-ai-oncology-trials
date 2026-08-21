# draft-move-in/sections - the seventeen section files (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Files](https://img.shields.io/badge/Files-17-00417A.svg)](.)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](.)
[![Rule 6](https://img.shields.io/badge/Rule%206-1%20commit%20per%20section-3C7DB2.svg)](../../sub-prompts/draft-move-in)
[![Instructions](https://img.shields.io/badge/Drafting%20instructions-96-6C757D.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../../README.md)

One `.tex` per section, one commit each, as Rule 6 requires. `sec-00` and
`sec-16` open with `\docfront` and are unnumbered; `sec-01` through `sec-15`
open with `\docpart` and are documents 1 through 15.

## The roster

| File | Document | Part | Sections | What stage 2 must resolve here |
|:--|:--|:--|:--|:--|
| `sec-00-front.tex` | Front matter | - | 5 | Why La Jolla, the document index, the site parameter table, the author record, the audience table |
| `sec-01-sb-1188-authorization.tex` | SB 1188, Site Authorization Act | I | 6 | Ten findings with citations, nine definitions, the authorization clock table |
| `sec-02-ab-3162-patient-rights.tex` | AB 3162, Patient Rights and Robotic Safety | I | 6 | Seven participant rights, the disclosure evidence table, the safety envelope numbers |
| `sec-03-sb-964-data-protection.tex` | SB 964, Model Transparency and Data Protection | I | 6 | Five transparency obligations, the retention periods, the breach clock |
| `sec-04-hr-10412-federal.tex` | H. R. 10412, Federal Act | I | 6 | The operative sections table, the appropriation arithmetic, the FDA reporting requirement |
| `sec-05-san-diego-municipal.tex` | San Diego Municipal Code Update | II | 6 | The zone table, the conditional use permit findings, the permit sequence |
| `sec-06-title-22-regulations.tex` | Title 22, Chapter 15 | II | 6 | The staffing minimum table, the register, the deficiency classes |
| `sec-07-fda-compliance-guide.tex` | FDA Compliance Guide | II | 6 | The correction map, the ICH E6(R3) mapping, the advisory-only boundary |
| `sec-08-building-code.tex` | Building Code Standards | III | 6 | The structural table, the mechanical provisions, the machine room |
| `sec-09-premises-code.tex` | Premises Code | III | 6 | The four-zone table, the robot envelopes, the five waste streams |
| `sec-10-parking-transportation.tex` | Parking and Transportation | III | 6 | The stall schedule, the accessible route, post-procedure transport |
| `sec-11-emergency-preparedness.tex` | Emergency Preparedness | III | 6 | The backup power table, the model fault response, the drill rotation |
| `sec-12-site-activation-sops.tex` | Activation and SOPs | IV | 6 | The five gates, the activation checklist, the procedure index |
| `sec-13-conventional-trial-requirements.tex` | Conventional Trial Requirements | IV | 7 | The protocol summary, the escalation table, the four-source evidence table |
| `sec-14-staffing-and-roles.tex` | Staffing and Roles | IV | 6 | The eleven-role roster, the firewall, the cost table, the move-in schedule |
| `sec-15-funding-and-lobbying.tex` | Funding and Lobbying | IV | 6 | The six-line award table, the lobbying authority table, the author record |
| `sec-16-backmatter.tex` | Back matter | - | 5 | The abbreviations table, availability, contributions, citation, references |

## The drafting instruction convention

```latex
\draftnote{Reuse the budget frame from
\mvfile{funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex}:
\$700,000 per year and \$3,500,000 over five years. Do not re-derive it.}
```

Every instruction names a path that exists. A path that does not resolve is a
stage 1 defect, not a stage 2 problem. `\mvfile` inserts a break opportunity
after every character, so a long path can wrap anywhere and can never overflow
the measure.

## Repository paths the 96 instructions name

| Path | Named by |
|:--|:--|
| `funding/pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex` | `sec-13` |
| `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` | `sec-04`, `sec-14`, `sec-15` |
| `funding/capitalization-plan/final-capital/sections/` | `sec-12`, `sec-13`, `sec-14`, `sec-15` |
| `funding/potential-partners/UC-San-Diego/` | `sec-00`, `sec-13`, `sec-15` |
| `funding/science-golden-age/` | `sec-04`, `sec-15` |
| `funding/move-in/inputs/READMES/` | `sec-00`, `sec-01`, `sec-02`, `sec-04`, `sec-07`, `sec-08`, `sec-15` |
| `funding/move-in/inputs/` accomplishments record | `sec-00`, `sec-15` |
| `regulatory/adaption-ich-e6r3/`, `Adaption-21-CFR-Part-50/`, `Adaption-21-CFR-Part-312/` | `sec-01`, `sec-02`, `sec-04`, `sec-07`, `sec-13` |
| `trial-protocol/`, `trial-ind/` | `sec-04`, `sec-07`, `sec-08`, `sec-10`, `sec-13` |
| `national-platform/` | `sec-03` |

## Cross-document consistency the instructions enforce

A package of fifteen documents fails when two of them state the same number
differently. Nineteen instructions exist only to prevent that, and each names
the section that owns the value.

| Value | Owned by | Cited by |
|:--|:--|:--|
| Deficiency and deviation classes | `sec-01` §5 | `sec-03` §6, `sec-06` §5, `sec-12` §6 |
| Record retention period | `sec-03` §5 | `sec-06` §6, `sec-09` §2, `sec-13` §6 |
| Robot operating envelope numbers | `sec-09` §3 | `sec-02` §5 |
| Hours of operation | `sec-05` §5 | `sec-12` §1, `sec-14` §4 |
| Waste pickup windows | `sec-09` §5 | `sec-05` §5 |
| Staffing minimums | `sec-14` §1 and §2 | `sec-06` §3 |
| Training hours | `sec-12` §5 | `sec-06` §3, `sec-11` §6 |
| Air pressure relationships | `sec-08` §3 | `sec-09` §4, `sec-13` §5 |
| Accessible path of travel | `sec-09` §6 | `sec-10` §3 |
| The $700,000 budget frame | `sec-15` §1 | `sec-04` §4, `sec-14` §5 |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../../sub-prompts/draft-move-in/prompt-3-fifteen-document-skeletons.md`](../../sub-prompts/draft-move-in/prompt-3-fifteen-document-skeletons.md) | The roster above, file for file |
| [`../../../pdac-funding-applications/final-apply/sections/`](../../../pdac-funding-applications/final-apply/sections) | The section header comment convention and the one-file-per-section discipline |
| [`../../inputs/`](../../inputs) | The codified drafting idiom taken from the predecessor package: findings, then definitions, then operative sections, with lettered subdivisions |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
