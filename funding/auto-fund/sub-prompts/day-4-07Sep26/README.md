# Sub-prompt 4 - 07Sep26, The Staged Queue (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](../../07Sep26)
[![Accent](https://img.shields.io/badge/Accent-Slate%20Plum%20%235B3A5E-5B3A5E.svg)](../../07Sep26/packet)
[![Federal offices](https://img.shields.io/badge/Federal%20offices-closed-9AA1A8.svg)](#why-this-day-sends-nothing)
[![Markets](https://img.shields.io/badge/NYSE%20%2F%20Nasdaq-closed-9AA1A8.svg)](#why-this-day-sends-nothing)
[![Emails](https://img.shields.io/badge/Emails-4%20queued-6C757D.svg)](../../07Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../07Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](../../07Sep26/forms)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../07Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../07Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

Labor Day. Federal offices are closed, the New York Stock Exchange and Nasdaq are
closed, and no counterparty this program deals with can act. This day exists in
the schedule because a day on which nothing can be received is the only day on
which everything can be prepared without interruption.

## Why this day sends nothing

An email that lands in a program officer's inbox on a federal holiday arrives at
the bottom of a Tuesday stack. A market order entered against a closed session
either rejects or fills at an opening price nobody has seen. Both are avoidable
by writing on the holiday and releasing on the next session, so that is what this
day does. Every item it produces carries a **HOLD FOR RELEASE** line naming the
next open session as its earliest send or entry time, and day 5 is the release.

The single approval this day asks for is therefore different in kind from the
other four.

## The single decision this day asks for

**Does the chief executive approve the release list, so that day 5 opens with the
queue already authorized?**

One approval on a holiday replaces five separate approvals under time pressure on
the following session.

## What this day produces

| # | Deliverable | Format | Release condition |
|:--|:--|:--|:--|
| 1 | `emails/email-01-congressional-delegation.txt` | `.txt` | Hold for the next open session |
| 2 | `emails/email-02-california-ibank-inquiry.txt` | `.txt` | Hold for the next open session |
| 3 | `emails/email-03-san-diego-economic-development.txt` | `.txt` | Hold for the next open session |
| 4 | `emails/email-04-fda-combination-products-pre-rfd.txt` | `.txt` | Hold for the next open session |
| 5 | `briefs/brief-01-data-room-index.md` | `.md` | Ready on completion, no hold |
| 6 | `briefs/brief-02-recognition-letters-use.md` | `.md` | Ready on completion, no hold |
| 7 | `briefs/brief-03-diligence-question-bank.md` | `.md` | Ready on completion, no hold |
| 8 | `forms/form-01-grants-gov-workspace-setup.md` | `.md` | Hold: portal support is closed |
| 9 | `forms/form-02-era-commons-profile-audit.md` | `.md` | Hold: portal support is closed |
| 10 | `investing/capital-04-queued-orders.md` | `.md` | Hold for the next open session |
| 11 | `diagrams/fig-10` .. `fig-12` | `.md` | No hold |
| 12 | `packet/` | `.tex`, `.pdf`, `.zip` | No hold |

## The recognition letters, and the discipline that governs them

Three presidential recognition letters have been issued to the chief executive.
`briefs/brief-02-recognition-letters-use.md` states what that is and what it is
not, and it is written to be read by a skeptical reviewer. A recognition letter
is a fact about correspondence. It is not an award, a grant, a review, an
endorsement of any document in this repository, or a commitment of any kind, and
it is never described as one in any file in
[`../../07Sep26`](../../07Sep26). The brief exists so that the letters can be
mentioned once, precisely, in a data room, rather than repeatedly and loosely in
correspondence.

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform and no other |
|:--|:--|:--|:--|
| 10 | PlantUML | Activity with a fork and a join | A closed market forces preparation and release onto two concurrent lanes that rejoin, which is exactly a fork and a join |
| 11 | D2 | SQL table records with a class column | A data room is a set of typed records with an access class on each, which is D2's SQL table shape |
| 12 | Graphviz | Fault tree with AND and OR gates | The question is what combination of failures makes the week produce nothing, and a fault tree is the only form that answers it by combination rather than by list |

Mermaid and Diagrams are unused on day 4 and appear on days 1, 2, 3 and 5.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 16 | The release list with its earliest send time and its owner | 4.0 cm |
| 17 | The data room by folder, with access class and what each answers | 4.6 cm |
| 18 | The three recognition letters: what each is, and what it is not | 5.0 cm |
| 19 | Twenty-two diligence questions with the file that answers each | 5.4 cm |
| 20 | Queued orders with instrument, side, type, limit, and time in force | 2.6 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Slate Plum `#5B3A5E`, with `#8B6790` and `#E7DEE9` as its two lighter shades |
| 2 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 3 | Caption lines | Two, balanced within a small character spread |
| 4 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 5 | Money | The frame in [`../../inputs`](../../inputs), not re-derived |
| 6 | Release discipline | Every item that reaches a counterparty carries a `HOLD FOR RELEASE` line and an earliest send or entry time |
| 7 | Recognition letters | Stated once, precisely, as facts about correspondence, and never as awards, agreements, or reviews |
| 8 | Dialect | American English, La Jolla usage |
| 9 | Rasters | None |

## Commit order

Identical in shape to days 1 to 3, with four email commits rather than five.

## Rule 5 source map

| Used | From | Where it appears in day 4 |
|:--|:--|:--|
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | `briefs/brief-02-recognition-letters-use.md`, Table 18, and the lobbying boundary in email 01 |
| `final-move-in/sections/sec-00-front.tex` | [`../../../move-in`](../../../move-in) | The company record paragraph reused in the data room index |
| `UC-San-Diego/priority-steps.md` §10 | [`../../../potential-partners`](../../../potential-partners) | Email 04's addresses, its subject line, and its eight component descriptions |
| `UC-San-Diego/priority-steps.md` §13 | [`../../../potential-partners`](../../../potential-partners) | Both form packs and the registration audit |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Figure 12's stop conditions and Table 19's hardest questions |
| `final-capital/sections/sec-10-build-method.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The custody rows of Table 17 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
