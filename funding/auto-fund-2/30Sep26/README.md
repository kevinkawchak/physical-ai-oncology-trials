# 30Sep26 - Day 3, The Structured Team (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-7A1F2B.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-7A1F2B.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-4-6C757D.svg)](emails)
[![Web form](https://img.shields.io/badge/White%20House-contact%20form-6C757D.svg)](forms)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Structured%20Team-7A1F2B.svg)](packet)

On September 12, 2026 the chief executive wrote to President Trump. Several
communications from the White House followed. This day answers them, and it does
so by turning the three priorities the message named into rules that anyone can
check.

## The message, quoted as sent

> Dear Mr. Trump,
>
> I am thankful for some humbling life circumstances leading to priority changes for myself and the startup which include:
> 1) Obedience over hard work and responsibilities.
> 2) Adapted to existing authority, order, and structure.
> 3) Acclimated to a teamwork based early-rise, scheduled routine.
>
> Sincerely,
> CEO Kevin Kawchak
> ChemicalQDevice
> September 12, 2026

## The one approval step

> **Approve answering the White House through its contact form and the Office of
> Science and Technology Policy, and placing the company, by letter, inside four
> established structures as a contributor rather than a lead.**

## The three priorities, as rules

| Priority, as written | The operating rule it becomes | How a counterparty can check it |
|:--|:--|:--|
| Obedience over hard work and responsibilities | When an office that owns a rule answers, the company follows the answer before doing more work of its own | Every letter this block sends asks the owner of the rule first |
| Adapted to existing authority, order, and structure | The company takes the role an existing structure offers, even when it is smaller than the role it asked for | Letters 2, 3 and 4 each ask for a contributor's place |
| Acclimated to a teamwork-based early-rise, scheduled routine | The working day starts at 5:00 a.m. Pacific, so the first correspondence window matches Washington's morning | The weekday in [`briefs/brief-03-the-early-rise-weekday.md`](briefs/brief-03-the-early-rise-weekday.md) |

## The run order

| Order | Item | Where | Depends on |
|:--|:--|:--|:--|
| 1 | Submit the White House contact form | [`forms/form-01-white-house-contact.md`](forms/form-01-white-house-contact.md) | Nothing |
| 2 | Write to OSTP | [`emails/email-01-ostp-response.txt`](emails/email-01-ostp-response.txt) | Row 1, so the letter can say the form was submitted |
| 3 | Offer a contributor's place to the Genesis Mission | [`emails/email-02-genesis-mission-contributor.txt`](emails/email-02-genesis-mission-contributor.txt) | Nothing |
| 4 | Offer a contributor's role to the Moores Clinical Trials Office | [`emails/email-03-ucsd-contributor-role.txt`](emails/email-03-ucsd-contributor-role.txt) | Nothing |
| 5 | Follow up in the CTEP thread | [`emails/email-04-nci-ctep-structure-followup.txt`](emails/email-04-nci-ctep-structure-followup.txt) | Row 4 sent |
| 6 | Adopt the early-rise weekday | [`briefs/brief-03-the-early-rise-weekday.md`](briefs/brief-03-the-early-rise-weekday.md) | Nothing |
| 7 | Run the quarter-end policy review | [`investing/capital-03-quarter-end-policy-review.md`](investing/capital-03-quarter-end-policy-review.md) | The quarter's last session |

## A note on the date

September 30 is the last day of the federal fiscal year. If federal
appropriations lapse at midnight, some federal offices written to this week may
not answer for a time. That is a reason to expect silence, never a reason to
follow up sooner: the three-business-day rule is counted in days the office is
open.

## Directory contents

```
30Sep26/
├── README.md              this approval sheet
├── emails/                4 .txt letters, each with at least three verified addresses
├── forms/                 1 .md pack: the White House contact form
├── briefs/                3 .md briefs: the priorities as rules, the structure map, the weekday
├── investing/             1 .md capital instruction: the quarter-end policy review
├── diagrams/              3 .md figure specifications
└── packet/                The Structured Team: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 30Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| [`../prompts/prompt-auto-fund-2.md`](../prompts/prompt-auto-fund-2.md) | This block | The message, quoted verbatim |
| `02Sep26/emails/email-06-nci-ctep-gore-reply.txt` | [`../../auto-fund`](../../auto-fund) | Letter 4 and the contributor-not-sponsor position |
| `04Sep26/emails/email-01-ucsd-moores-escalation.txt` | [`../../auto-fund`](../../auto-fund) | Letter 3's earlier contact |
| `08Sep26/briefs/brief-02-weekly-cadence.md` | [`../../auto-fund`](../../auto-fund) | The weekday themes in brief 3 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../move-in`](../../move-in) | The sponsor-investigator firewall in Table 15 |
| `science-golden-age/` | [`../../science-golden-age`](../../science-golden-age) | The policy position letter 1 is written against |

## Positioning, carried into every file in this directory

The White House communications are facts about correspondence. They are never
described as an award, an appointment, an endorsement, or a commitment, and they
are not quoted, because their text is not in the repository. No agreement of any
kind exists with the Department of Energy, UC San Diego, the National Cancer
Institute, or any other institution. Rasonque is approved in the metastatic
setting, and the perioperative use this program proposes remains
investigational. Nothing in [`investing/`](investing) is investment advice.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
