# 07Sep26 - Day 4, The Staged Queue (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-5B3A5E.svg)](#the-one-approval-step)
[![Federal offices](https://img.shields.io/badge/Federal%20offices-closed-9AA1A8.svg)](#nothing-can-be-received-today)
[![Markets](https://img.shields.io/badge/NYSE%20%2F%20Nasdaq-closed-9AA1A8.svg)](#nothing-can-be-received-today)
[![Emails](https://img.shields.io/badge/Emails-4%20held-6C757D.svg)](emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](forms)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Staged%20Queue-5B3A5E.svg)](packet)

Labor Day, the first Monday in September. Federal offices are closed. The New
York Stock Exchange and Nasdaq are closed. No counterparty this program deals
with can act today.

That is why this day exists in the schedule, and why it has a different shape
from the other four.

## Nothing can be received today

| Counterparty | Status | Consequence |
|:--|:--|:--|
| Federal program offices | Closed, federal holiday | A letter sent today lands at the bottom of a Tuesday stack |
| NYSE and Nasdaq | Closed, full market closure | An order entered against a closed session either rejects or fills at an opening price nobody has seen |
| Federal portal support desks | Closed or reduced | A submission that fails cannot be helped until the next session |
| Institutional offices | Most closed | A routing request lands unread |

Both of those first two problems are avoidable by writing today and releasing on
the next open session, so that is exactly what this day does. Every item that
reaches a counterparty carries a **`HOLD FOR RELEASE`** line naming the next open
session as its earliest send or entry time, and day 5 is the release.

## The one approval step

The approval this day asks for is different in kind from the other four. It is
not "send this"; it is "authorize the release list, so that the next session
opens with the queue already approved."

> **Approve the release list, so that day 5 opens with the queue authorized.**

One approval on a closed day replaces five separate approvals taken under time
pressure on the following session.

## The run order

Nothing on this list has a counterparty, which is the point.

| Order | Item | Where | Time |
|:--|:--|:--|:--|
| 1 | Read the release list and approve or amend it | [`packet/`](packet) §2 | 12 minutes |
| 2 | Read the recognition letters brief and confirm the wording | [`briefs/`](briefs) | 8 minutes |
| 3 | Assemble the data room from the index | [`briefs/`](briefs) | 90 minutes |
| 4 | Read the diligence question bank and mark any question with no answer | [`briefs/`](briefs) | 30 minutes |
| 5 | Set the queued orders and their limits, entered nowhere | [`investing/`](investing) | 15 minutes |
| 6 | Complete both form packs offline, submit neither | [`forms/`](forms) | 45 minutes |

## The four held letters

| # | File | To | Held until |
|:--|:--|:--|:--|
| 1 | [`email-01-congressional-delegation.txt`](emails/email-01-congressional-delegation.txt) | The San Diego congressional delegation's district health staff | The next open session |
| 2 | [`email-02-california-ibank-inquiry.txt`](emails/email-02-california-ibank-inquiry.txt) | The state small business finance center | The next open session |
| 3 | [`email-03-san-diego-economic-development.txt`](emails/email-03-san-diego-economic-development.txt) | The regional economic development corporation | The next open session |
| 4 | [`email-04-fda-combination-products-pre-rfd.txt`](emails/email-04-fda-combination-products-pre-rfd.txt) | The FDA Office of Combination Products | The next open session |

Letter 4 is the most consequential item in the whole five-day block and it is
written on the quietest day deliberately. A Pre-Request for Designation
determines which FDA center leads the review of a drug, a robotic platform, and
an advisory software component, and every regulatory assumption downstream of it
depends on the answer.

## The three recognition letters, and the discipline that governs them

Three presidential recognition letters have been issued to the chief executive.
[`briefs/brief-02-recognition-letters-use.md`](briefs/brief-02-recognition-letters-use.md)
states precisely what that is and what it is not, and it is written to be read by
a skeptical reviewer.

A recognition letter is a fact about correspondence. It is not an award, a grant,
a review, an endorsement of any document in this repository, or a commitment of
any kind, and it is never described as one in any file in this directory. The
brief exists so that the letters can be mentioned once, precisely, in a data
room, rather than repeatedly and loosely in correspondence.

## Directory contents

```
07Sep26/
├── README.md              this approval sheet
├── emails/                4 .txt letters, every one carrying HOLD FOR RELEASE
├── briefs/                3 .md briefs: data room index, recognition letters,
│                          diligence question bank
├── forms/                 2 .md packs, completed offline and submitted later
├── investing/             1 .md queued-order instruction, entered nowhere
├── diagrams/              3 .md figure specifications
└── packet/                The Staged Queue: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 07Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../move-in`](../../move-in) | `briefs/brief-02-recognition-letters-use.md`, the packet's Table 18, and the lobbying boundary in letter 1 |
| `final-move-in/sections/sec-00-front.tex` | [`../../move-in`](../../move-in) | The company record rows of the data room index |
| `UC-San-Diego/priority-steps.md` §10 | [`../../potential-partners`](../../potential-partners) | Letter 4's addresses, its subject line, and its eight component descriptions |
| `UC-San-Diego/priority-steps.md` §11 | [`../../potential-partners`](../../potential-partners) | Letter 4's parallel drug and device meeting note |
| `UC-San-Diego/priority-steps.md` §13 | [`../../potential-partners`](../../potential-partners) | Both form packs |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The packet's Figure 12 stop conditions and Table 19 |
| `final-capital/sections/sec-10-build-method.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The custody rows of Table 17 |
| `02Sep26/investing/capital-01-treasury-ladder.md` | [`../02Sep26`](../02Sep26) | The queued orders' instrument list |
| `03Sep26/investing/capital-02-corporate-reserve-allocation.md` | [`../03Sep26`](../03Sep26) | The branch the queued orders are conditional on |

## Positioning, carried into every file in this directory

Nothing here has been sent, filed, entered, or agreed, and nothing here may be
sent or entered before the next open session. The three recognition letters are
facts about correspondence and are never described as awards, agreements, or
reviews. No institution is described as a partner, sponsor, site, or endorser. No
lobbying is undertaken with federal award funds, and no federal award exists.
Daraxonrasib is approved in the metastatic setting and is nowhere described as
first in human; the perioperative use this program proposes remains
investigational.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
