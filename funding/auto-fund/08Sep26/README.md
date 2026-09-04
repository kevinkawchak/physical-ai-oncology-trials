# 08Sep26 - Day 5, The Execution Record (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8A4B2A.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-8A4B2A.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](emails)
[![Briefs](https://img.shields.io/badge/Briefs-2-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-1-6C757D.svg)](forms)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Execution%20Record-8A4B2A.svg)](packet)
[![Cadence](https://img.shields.io/badge/Weekly%20cadence-established-9AA1A8.svg)](briefs/brief-02-weekly-cadence.md)

The first open session after the holiday. The queue day 4 built is released, the
orders it staged are entered, the form it prepared is submitted, and the week
hands a repeatable rhythm to the next one.

This is the day the four before it were written for.

## The one approval step

> **Approve the release, and adopt the weekly cadence.**

The release is already authorized in outline by day 4's approval; what this day
asks is confirmation that nothing has changed overnight, plus a decision on the
cadence that governs every week after this one.

A week that produces five days of correspondence and no recurring rhythm is a
week that has to be reinvented. The cadence in
[`briefs/brief-02-weekly-cadence.md`](briefs/brief-02-weekly-cadence.md) is one
page and fixes what happens on each weekday from here forward, so that the next
five-day block is a substitution of content into a known frame rather than a
fresh design.

## The run order, timed against one market session

| Order | Item | Where | When | Depends on |
|:--|:--|:--|:--|:--|
| 1 | Confirm the market is open and re-read the auction calendar | [`investing/`](investing) | Before the open | Nothing |
| 2 | Write this morning's limit price into the broker instruction | [`emails/`](emails) | Before the open | Item 1 |
| 3 | Release letters 1 to 4 of the day 4 queue | [`../07Sep26/emails`](../07Sep26/emails) | Early session | Day 4 approval |
| 4 | Send the broker execution instruction | [`emails/`](emails) | Mid session, never the first or last fifteen minutes | Items 1, 2 |
| 5 | Send the three follow-ups | [`emails/`](emails) | Any time after item 3 | The three-business-day rule |
| 6 | Submit the Grants.gov workspace and the profile changes | [`../07Sep26/forms`](../07Sep26/forms) | Any time | Portal support open |
| 7 | Read the submission checklist and mark what is missing | [`forms/`](forms) | End of session | Items 3 to 6 |

## The five letters

| # | File | To | What it does |
|:--|:--|:--|:--|
| 1 | [`email-01-release-cover-note.txt`](emails/email-01-release-cover-note.txt) | Self, as the release record | Records what was released, at what time, and by whom |
| 2 | [`email-02-federal-followup-round-two.txt`](emails/email-02-federal-followup-round-two.txt) | The federal mechanisms written to on day 1 | A single short follow-up, sent only where three business days have passed |
| 3 | [`email-03-developer-followup.txt`](emails/email-03-developer-followup.txt) | The agent's developer | A short follow-up on the submission window question only |
| 4 | [`email-04-site-meeting-confirmation.txt`](emails/email-04-site-meeting-confirmation.txt) | Whichever institution replied | Confirms a meeting, names attendees, and sends the agenda |
| 5 | [`email-05-broker-execution-instruction.txt`](emails/email-05-broker-execution-instruction.txt) | The broker-dealer | Transmits the six queued orders with this morning's limit |

Letter 1 is addressed to the chief executive himself and is filed rather than
sent. A release with no record of what left and when is a release nobody can
reconstruct three weeks later when a recipient replies to something.

## The three-business-day rule, applied

No follow-up is sent inside three business days of the letter it follows. The
rule comes from the escalation guidance in the partner research and it is
applied to federal offices as well, because a program officer who receives a
follow-up on day two reads an applicant who is counting hours.

| Letter sent | Earliest follow-up |
|:--|:--|
| Day 1 federal letters | This day, which is the third business day |
| Day 2 developer letter | This day, on the window question only |
| Day 3 site and foundation letters | Not yet. Two business days have passed |

Letter 2 and letter 3 therefore go out; no follow-up to the day 3 recipients does.

## Directory contents

```
08Sep26/
├── README.md              this approval sheet
├── emails/                5 .txt letters, one of them a filed release record
├── briefs/                2 .md briefs: the thirty-day pipeline, the weekly cadence
├── forms/                 1 .md pack: the SBIR submission checklist
├── investing/             1 .md execution and settlement instruction
├── diagrams/              3 .md figure specifications
└── packet/                The Execution Record: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 08Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `UC-San-Diego/priority-steps.md` §4 | [`../../potential-partners`](../../potential-partners) | The three-business-day rule above, applied to every follow-up |
| `UC-San-Diego/priority-steps.md` §13, §14 | [`../../potential-partners`](../../potential-partners) | `forms/form-01-sbir-phase-i-submission-checklist.md` |
| `applications/app-05-nih-sbir-seed/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The checklist's budget and phase lines |
| `final-capital/sections/sec-05-twelve-milestones.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The packet's Table 22 stage definitions and Figure 15's layer counts |
| `final-capital/sections/sec-07-operating-plan.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The packet's Table 23 and Figure 14's standing functions |
| `07Sep26/investing/capital-04-queued-orders.md` | [`../07Sep26`](../07Sep26) | The six orders this day enters |
| `07Sep26/README.md` | [`../07Sep26`](../07Sep26) | The release list this day executes |
| Days 1 to 4 of this block | [`..`](..) | The packet's Table 21 and Table 25 in full |

## Positioning, carried into every file in this directory

Nothing in this directory changes what any prior day claimed. No agreement of any
kind exists with any institution or with the agent's developer. No offering
exists and no instrument has been selected. No order in
[`investing/`](investing) constitutes investment advice, and each is entered only
against the six checks that precede it. Daraxonrasib is approved in the metastatic
setting and is nowhere described as first in human; the perioperative use this
program proposes remains investigational.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
