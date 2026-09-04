# Sub-prompt 3 - 04Sep26, The Site and Partner Package (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-2F5D3A.svg)](../../04Sep26)
[![Accent](https://img.shields.io/badge/Accent-Cypress%20Green%20%232F5D3A-2F5D3A.svg)](../../04Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](../../04Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-2-6C757D.svg)](../../04Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](../../04Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../04Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../04Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../04Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

A funder buys a trial, and a trial needs a site. This day approaches the two La
Jolla institutions that could host one, the disease-specific foundations that
fund pancreatic work directly, and it does so on a Friday, which is the day of
the week a 45-minute feasibility meeting is most easily placed in the following
week's calendar.

## The single decision this day asks for

**Does the chief executive approve naming a target date for a feasibility meeting
and asking three foundations for the same thing on the same day?**

## Why the site approach is not a repeat of application 10

Application 10 in
[`../../../pdac-funding-applications/applications/app-10-ucsd-moores-engine`](../../../pdac-funding-applications/applications/app-10-ucsd-moores-engine)
asked UC San Diego Moores Cancer Center for a 45-minute feasibility meeting. That
letter still stands. This day does two things it did not: it escalates on the
published path in `../../../potential-partners/UC-San-Diego/priority-steps.md`
§4 rather than repeating the original request, and it opens the parallel Scripps
route in §6 of the Scripps plan, which is an inbound-visibility notice rather
than a request to host.

Both routes are run at once and both are described to the other. A site that
learns from a third party that it is one of two is owed the courtesy of learning
it from the letter instead.

## What this day produces

| # | Deliverable | Format | Recipient class |
|:--|:--|:--|:--|
| 1 | `emails/email-01-ucsd-moores-escalation.txt` | `.txt` | Moores Cancer Center leadership and surgical oncology |
| 2 | `emails/email-02-scripps-digital-trials-notice.txt` | `.txt` | Scripps Research Digital Trials Center and named research leads |
| 3 | `emails/email-03-lustgarten-foundation.txt` | `.txt` | Pancreatic cancer research foundation staff |
| 4 | `emails/email-04-pancan-research-grants.txt` | `.txt` | Pancreatic cancer network grants staff |
| 5 | `emails/email-05-actri-startup-support.txt` | `.txt` | Clinical trial start-up support services |
| 6 | `briefs/brief-01-site-feasibility-questions.md` | `.md` | A site principal investigator and a trials office |
| 7 | `briefs/brief-02-two-site-parallel-approach.md` | `.md` | Either institution, and the chief executive |
| 8 | `forms/form-01-ucsd-iit-concept-intake.md` | `.md` | Investigator-initiated trial concept intake |
| 9 | `forms/form-02-foundation-letter-of-intent.md` | `.md` | A foundation letter-of-intent portal |
| 10 | `investing/capital-03-site-startup-reserve.md` | `.md` | The chief executive and the broker |
| 11 | `diagrams/fig-07` .. `fig-09` | `.md` | The author, when a figure needs correction |
| 12 | `packet/` | `.tex`, `.pdf`, `.zip` | Every recipient above, as the attachment |

Two briefs rather than three, because this day's technical content is a question
list and a positioning statement, and a third brief would restate one of them.

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform and no other |
|:--|:--|:--|:--|
| 7 | Graphviz | Three dashed clusters | Obligations that belong to a site, a sponsor and a developer are three disjoint sets with edges between them, which is a clustered digraph |
| 8 | Diagrams | Clustered infrastructure with glyphs | Where a function physically sits across two campuses is an infrastructure drawing, and only the Diagrams vocabulary carries a glyph per node |
| 9 | Mermaid | Flowchart with gates | A foundation funnel is a sequence of gates with attrition at each, which is a flowchart with labeled edges |

D2 and PlantUML are unused on day 3 and appear on days 1, 2, 4 and 5.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 11 | The five feasibility questions and what a yes to each unlocks | 4.8 cm |
| 12 | The two candidate sites against seven capability criteria | 3.6 cm |
| 13 | Three foundations, their cycle, their ceiling, and their fit | 3.4 cm |
| 14 | Site start-up cost lines against the $700,000 annual direct cost | 3.8 cm |
| 15 | The three positioning corrections carried into every letter | 5.2 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Cypress Green `#2F5D3A`, with `#5E9370` and `#DFEAE1` as its two lighter shades |
| 2 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 3 | Caption lines | Two, balanced within a small character spread |
| 4 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 5 | Money | The frame in [`../../inputs`](../../inputs), not re-derived |
| 6 | Positioning | The three corrections travel with every letter: daraxonrasib is not first in human; the robotic configuration is specified at the site agreement; no agreement of any kind exists with any institution |
| 7 | Confidentiality | No engineering detail is sent before a confidentiality agreement is in place, per the Office of Clinical Trials Administration route |
| 8 | Dialect | American English, La Jolla usage |
| 9 | Rasters | None |

## Commit order

Identical in shape to days 1 and 2, with one fewer brief commit.

## Rule 5 source map

| Used | From | Where it appears in day 3 |
|:--|:--|:--|
| `UC-San-Diego/priority-steps.md` §3, §4 | [`../../../potential-partners`](../../../potential-partners) | Email 01's addresses, its escalation wording, and Table 11 |
| `UC-San-Diego/priority-steps.md` §6, §7, §8 | [`../../../potential-partners`](../../../potential-partners) | `forms/form-01-ucsd-iit-concept-intake.md`, invariant 7, and email 05 |
| `Scripps/priority-steps.md` §2, §6, §11 | [`../../../potential-partners`](../../../potential-partners) | Email 02's portfolio framing, its five addresses, and the meeting location |
| `UC-San-Diego/README.md` | [`../../../potential-partners`](../../../potential-partners) | The five success criteria reproduced as Table 11 |
| `applications/app-10-ucsd-moores-engine/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The prior request email 01 escalates rather than repeats |
| `applications/app-06-fnih-amp/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The consortium framing reused in emails 03 and 04 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | Table 14's start-up cost lines and Figure 8's campus split |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
