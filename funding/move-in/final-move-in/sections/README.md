# final-move-in/sections - the seventeen polished section files (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Files](https://img.shields.io/badge/Files-17-00417A.svg)](.)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](.)
[![Tables](https://img.shields.io/badge/Tables-56-3C7DB2.svg)](.)
[![Page--breaking tables](https://img.shields.io/badge/Page--breaking%20tables-14-6C757D.svg)](.)
[![Short pages](https://img.shields.io/badge/Short%20pages-0-brightgreen.svg)](.)
[![Characters](https://img.shields.io/badge/Characters-175%2C256-6C757D.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../../README.md)

One `.tex` per section, one commit each. Every file opens with a `% STAGE 3.`
comment naming exactly what changed in it and why, so a reader diffing this
directory against `../../full-move-in/sections/` never has to guess.

## What changed in each file

| File | Stage 3 change |
|:--|:--|
| `sec-00-front.tex` | Three column widths retuned; one paragraph tightened by a clause |
| `sec-01-sb-1188-authorization.tex` | Two widths retuned; finding (b) tightened; the seven-step authorization table moved to a page-breaking form |
| `sec-02-ab-3162-patient-rights.tex` | Two widths retuned; a `\needspace` binds the rights list to the paragraph that explains subdivision (a) |
| `sec-03-sb-964-data-protection.tex` | Two widths retuned; a `\needspace` binds the operative sentence of §3 to its table |
| `sec-04-hr-10412-federal.tex` | Two widths retuned; the eight-row operative sections table moved to a page-breaking form |
| `sec-05-san-diego-municipal.tex` | Three widths retuned, including the permit table's Relationship column |
| `sec-06-title-22-regulations.tex` | The ten-row staffing table moved to a page-breaking form; two widths retuned |
| `sec-07-fda-compliance-guide.tex` | The fourteen-row correction map moved to a page-breaking form; two widths retuned |
| `sec-08-building-code.tex` | The one in-document `\clearpage` in the package, before §6; two widths retuned |
| `sec-09-premises-code.tex` | The four-zone table rebalanced against measured wrap depth; two further widths retuned |
| `sec-10-parking-transportation.tex` | Three widths retuned, including the stall schedule's Dimension column |
| `sec-11-emergency-preparedness.tex` | Three widths retuned, including the backup power Runtime column |
| `sec-12-site-activation-sops.tex` | The training table moved to a page-breaking form; four widths retuned; `\needspace` before §6; five sentences tightened |
| `sec-13-conventional-trial-requirements.tex` | Two tables moved to a page-breaking form; five widths retuned; `\needspace` before the cost table; three paragraphs tightened |
| `sec-14-staffing-and-roles.tex` | Four long tables moved to a page-breaking form; six widths retuned; two move-in rows merged |
| `sec-15-funding-and-lobbying.tex` | The author record column widened 0.8 cm, recovering fourteen wrapped lines; two tables moved to a page-breaking form; four widths retuned |
| `sec-16-backmatter.tex` | The abbreviation table rebuilt mechanically from the body: 30 entries with 13 unused and 2 blank cells became 24 entries in 12 full rows |

## The fourteen page-breaking tables

A `tabularx` cannot break across a page, so a table taller than the space left
on the page either overflows or pushes its caption alone onto the next page.
Fourteen tables of ten rows or more are set in `xltabular` through the
`\mvltable` wrapper, each with a header that repeats on the continuation page.

| Section | Table | Rows |
|:--|:--|:--|
| `sec-01` | The authorization sequence | 7 |
| `sec-04` | The bill's operative sections | 8 |
| `sec-06` | Staffing minimums by position | 10 |
| `sec-07` | The correction map to five CFR parts | 14 |
| `sec-12` | The activation checklist | 20 |
| `sec-12` | The standard operating procedure index | 20 |
| `sec-12` | Initial training hours by role | 11 |
| `sec-13` | Inclusion and exclusion criteria | 12 |
| `sec-14` | The eleven-role roster | 12 |
| `sec-14` | Qualifications and delegated tasks | 11 |
| `sec-14` | The cost of the roster | 12 |
| `sec-14` | The forty-six-week move-in | 12 |
| `sec-15` | The award, line by line | 7 |
| `sec-15` | Lobbying activity, source and authority | 7 |
| `sec-15` | The deposited work record | 24 |
| `sec-16` | Abbreviations | 12 |

## Cross-document consistency, re-verified at this stage

Every value below is owned by exactly one section. Each was re-checked against
its owner after the stage 3 edits, because a tightened sentence is the easiest
place to lose a number.

| Value | Owner | Cited by | Re-verified |
|:--|:--|:--|:--|
| Deficiency and deviation classes and penalty ranges | `sec-01` §5 | `sec-03` §6, `sec-06` §5, `sec-12` §6 | yes |
| Six-year retention from database lock | `sec-03` §5 | `sec-06` §6, `sec-09` §2, `sec-13` §6 | yes |
| Seven robot envelope and interlock values | `sec-09` §3 | `sec-02` §5 | yes |
| Hours of operation | `sec-05` §5 | `sec-09` §2, `sec-12` §1, `sec-14` §4 | yes |
| Waste pickup windows | `sec-09` §5 | `sec-05` §5 | yes |
| Staffing minimums and qualifications | `sec-14` §1, §2 | `sec-06` §3 | yes, word for word |
| Training hours | `sec-12` §5 | `sec-06` §3, `sec-11` §6 | yes |
| Air changes and pressure relationships | `sec-08` §3 | `sec-09` §4, `sec-13` §5 | yes |
| Accessible path of travel | `sec-10` §3 | `sec-09` §6 | yes |
| The $700,000 budget frame | `sec-15` §1 | `sec-04` §4, `sec-14` §5 | yes |
| Nineteen site parameters | `sec-00` §3 | `sec-05`, `sec-08`, `sec-09`, `sec-10`, `sec-13` | yes |
| Commissioning gate C8 and activation gate G1 | `sec-08` §6, `sec-12` §1 | each other | yes, one shared boundary |

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../../full-move-in/sections/`](../../full-move-in/sections) | Every file here begins as its stage 2 counterpart |
| [`../../../pdac-funding-applications/final-apply/sections/`](../../../pdac-funding-applications/final-apply/sections) | The section header comment convention, and the practice of naming the stage change in the file itself |
| [`../../sub-prompts/final-move-in/prompt-1-clearpage-discipline.md`](../../sub-prompts/final-move-in/prompt-1-clearpage-discipline.md) | The fix hierarchy that prefers a sentence over a skip, followed in all six paragraph edits |
| [`../../sub-prompts/final-move-in/prompt-3-dialect-and-proofreading.md`](../../sub-prompts/final-move-in/prompt-3-dialect-and-proofreading.md) | The dialect and punctuation audits, run to zero |
| [`../../sub-prompts/final-move-in/prompt-4-context-verification.md`](../../sub-prompts/final-move-in/prompt-4-context-verification.md) | The two verification passes recorded in [`../README.md`](../README.md) |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
