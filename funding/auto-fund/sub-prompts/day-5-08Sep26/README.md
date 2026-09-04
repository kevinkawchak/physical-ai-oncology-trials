# Sub-prompt 5 - 08Sep26, The Execution Record (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8A4B2A.svg)](../../08Sep26)
[![Accent](https://img.shields.io/badge/Accent-Ember%20Rust%20%238A4B2A-8A4B2A.svg)](../../08Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](../../08Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-2-6C757D.svg)](../../08Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-1-6C757D.svg)](../../08Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../08Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../08Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../08Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

The first open session after the holiday. The queue day 4 built is released, the
orders it staged are entered, the forms it prepared are submitted, and the week
hands a repeatable cadence to the next one. This is the day the four days before
it were written for.

## The single decision this day asks for

**Does the chief executive approve the release, and adopt the weekly cadence the
day proposes?**

A week that produces five days of correspondence and no recurring rhythm is a
week that has to be reinvented. The cadence in
`briefs/brief-02-weekly-cadence.md` is one page and fixes what happens on each
weekday from here forward, so that the next five-day block is a substitution of
content into a known frame rather than a fresh design.

## What this day produces

| # | Deliverable | Format | Depends on |
|:--|:--|:--|:--|
| 1 | `emails/email-01-release-cover-note.txt` | `.txt` | The day 4 release list |
| 2 | `emails/email-02-federal-followup-round-two.txt` | `.txt` | Day 1's five letters |
| 3 | `emails/email-03-developer-followup.txt` | `.txt` | Day 2's developer letter |
| 4 | `emails/email-04-site-meeting-confirmation.txt` | `.txt` | Day 3's site letters |
| 5 | `emails/email-05-broker-execution-instruction.txt` | `.txt` | Day 4's queued orders |
| 6 | `briefs/brief-01-thirty-day-pipeline.md` | `.md` | All four prior days |
| 7 | `briefs/brief-02-weekly-cadence.md` | `.md` | All four prior days |
| 8 | `forms/form-01-sbir-phase-i-submission-checklist.md` | `.md` | Day 1's registration packs |
| 9 | `investing/capital-05-execution-and-settlement.md` | `.md` | Day 4's queued orders |
| 10 | `diagrams/fig-13` .. `fig-15` | `.md` | The author, when a figure needs correction |
| 11 | `packet/` | `.tex`, `.pdf`, `.zip` | Everything above |

One form pack rather than two, because this day submits rather than registers,
and the single pack is the pre-submission checklist that gates the submission.

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform and no other |
|:--|:--|:--|:--|
| 13 | Mermaid | Gantt across one session | The subject is time-of-day sequencing inside a single market session, and a Gantt is the only form that shows both duration and overlap |
| 14 | Diagrams | Clustered topology with glyphs | A recurring cadence is a set of standing functions with a glyph each, grouped by weekday, which is what the Diagrams vocabulary draws |
| 15 | D2 | Layered stack with counts | A thirty-day pipeline is a stack of stages with a count on each layer, which is D2's layer construct |

Graphviz and PlantUML are unused on day 5 and appear on days 1, 2, 3 and 4. Over
the five days each platform is used exactly three times.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 21 | The release list with its actual send order and its dependency | 4.2 cm |
| 22 | The thirty-day pipeline by stage, count, and next action date basis | 3.4 cm |
| 23 | The weekly cadence by weekday, with owner and time budget | 3.0 cm |
| 24 | Execution and settlement with instrument, side, size, and settlement basis | 2.8 cm |
| 25 | The five-day record: what was sent, to whom, and what it asked | 4.4 cm |

Table 25 is the closing artifact of the whole block. It is the one page a person
who missed the week can read to know what the week did.

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Ember Rust `#8A4B2A`, with `#B57A55` and `#EFE1D8` as its two lighter shades |
| 2 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 3 | Caption lines | Two, balanced within a small character spread |
| 4 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 5 | Money | The frame in [`../../inputs`](../../inputs), not re-derived |
| 6 | Follow-up timing | No follow-up is sent inside three business days of the letter it follows, per the escalation rule the partner research sets |
| 7 | Record keeping | Table 25 is written so that nothing in it depends on a file being read; every row stands alone |
| 8 | Dialect | American English, La Jolla usage |
| 9 | Rasters | None |

## Commit order

Identical in shape to days 1 to 4, with two brief commits and one form commit.

## Rule 5 source map

| Used | From | Where it appears in day 5 |
|:--|:--|:--|
| `UC-San-Diego/priority-steps.md` §4 | [`../../../potential-partners`](../../../potential-partners) | Invariant 6, the three-business-day escalation rule |
| `UC-San-Diego/priority-steps.md` §13, §14 | [`../../../potential-partners`](../../../potential-partners) | `forms/form-01-sbir-phase-i-submission-checklist.md` |
| `applications/app-05-nih-sbir-seed/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The submission checklist's budget lines |
| `final-capital/sections/sec-05-twelve-milestones.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 22's stage definitions and Figure 15's layer counts |
| `final-capital/sections/sec-07-operating-plan.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 23's owner and time-budget columns and Figure 14's standing functions |
| Days 1 to 4 of this build | [`..`](..) | Table 21 and Table 25 in full |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
