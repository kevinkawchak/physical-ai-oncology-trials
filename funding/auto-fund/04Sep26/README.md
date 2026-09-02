# 04Sep26 - Day 3, The Site and Partner Package (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-2F5D3A.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-2F5D3A.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](emails)
[![Briefs](https://img.shields.io/badge/Briefs-2-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](forms)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Site%20and%20Partner%20Package-2F5D3A.svg)](packet)
[![Agreements](https://img.shields.io/badge/Agreements%20in%20place-none-9AA1A8.svg)](#positioning-carried-into-every-file-in-this-directory)

A funder buys a trial, and a trial needs a site. This day approaches the two La
Jolla institutions that could host one, the disease-specific foundations that
fund pancreatic work directly, and the clinical trial support service that would
have to say yes before either of the first two could.

It falls on a Friday, which is the day of the week a 45-minute meeting is most
easily placed in the following week's calendar.

## The one approval step

> **Approve naming a target week for a feasibility meeting, and approve asking
> two institutions and two foundations for related things on the same day.**

Naming a week is the whole of the decision. A request with no proposed date
becomes a thread; a request with a proposed week becomes a calendar entry or a
clear no, and both are more useful than a thread.

Nothing in this directory has been sent, filed, or agreed.

## Why the site approach is not a repeat of a letter already sent

Application 10 in
[`../../pdac-funding-applications/applications/app-10-ucsd-moores-engine`](../../pdac-funding-applications/applications/app-10-ucsd-moores-engine)
asked UC San Diego Moores Cancer Center for a 45-minute feasibility meeting. That
letter still stands and is not resent.

This day does two things it did not. It escalates on the published path in
`../../potential-partners/UC-San-Diego/priority-steps.md` §4, which is a
different set of recipients asking a different question. And it opens the
parallel Scripps route in §6 of the Scripps plan, which is an inbound-visibility
notice rather than a request to host.

Both routes run at once and **both are told about the other**. An institution
that learns from a third party that it is one of two is owed the courtesy of
learning it from the letter instead.

## The run order

| Order | Item | Where | Time | Depends on |
|:--|:--|:--|:--|:--|
| 1 | Choose the target week and write it into all five letters | [`emails/`](emails) | 5 minutes | Nothing |
| 2 | Send the Moores escalation | [`emails/`](emails) | 6 minutes | Item 1 |
| 3 | Send the Scripps visibility notice | [`emails/`](emails) | 6 minutes | Item 1 |
| 4 | Send the clinical trial support services request | [`emails/`](emails) | 5 minutes | Item 1 |
| 5 | Send the two foundation letters | [`emails/`](emails) | 10 minutes | Item 1 |
| 6 | Complete the concept intake pack, hold it until a site responds | [`forms/`](forms) | 40 minutes | Nothing |
| 7 | Read the letter-of-intent pack against each foundation's cycle | [`forms/`](forms) | 20 minutes | Nothing |

## The five letters

| # | File | To | The ask |
|:--|:--|:--|:--|
| 1 | [`email-01-ucsd-moores-escalation.txt`](emails/email-01-ucsd-moores-escalation.txt) | Moores leadership and surgical oncology, with the trials office copied | The right site principal investigator and trial operations contact |
| 2 | [`email-02-scripps-digital-trials-notice.txt`](emails/email-02-scripps-digital-trials-notice.txt) | Scripps Research Digital Trials Center and four named research leads | A read of the public portfolio, and a view on whether it is of interest |
| 3 | [`email-03-lustgarten-foundation.txt`](emails/email-03-lustgarten-foundation.txt) | Pancreatic cancer research foundation staff | Which mechanism, if any, fits a company-sponsored Phase 1 |
| 4 | [`email-04-pancan-research-grants.txt`](emails/email-04-pancan-research-grants.txt) | Pancreatic cancer network grants staff | The same question, asked of a different cycle |
| 5 | [`email-05-actri-startup-support.txt`](emails/email-05-actri-startup-support.txt) | Clinical trial support services | A sponsor-investigator determination, which gates everything else |

Letter 5 is the one a reader might skip and should not. The determination of
which party is capable of serving as sponsor-investigator has to be settled
before a protocol is finalized, because the regulatory strategy, the site
responsibilities and the budget must all be internally consistent with it.

## Directory contents

```
04Sep26/
├── README.md              this approval sheet
├── emails/                5 .txt letters, each with its own pre-send checklist
├── briefs/                2 .md technical briefs
├── forms/                 2 .md form packs
├── investing/             1 .md site start-up reserve instruction
├── diagrams/              3 .md figure specifications
└── packet/                The Site and Partner Package: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 04Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `UC-San-Diego/priority-steps.md` §3, §4 | [`../../potential-partners`](../../potential-partners) | Letter 1's five addresses and its escalation wording |
| `UC-San-Diego/priority-steps.md` §6, §7 | [`../../potential-partners`](../../potential-partners) | `forms/form-01-ucsd-iit-concept-intake.md` and the confidentiality constraint |
| `UC-San-Diego/priority-steps.md` §8 | [`../../potential-partners`](../../potential-partners) | Letter 5 in full, including the ten support items |
| `UC-San-Diego/README.md` | [`../../potential-partners`](../../potential-partners) | The five success criteria, reproduced as the packet's Table 11 |
| `Scripps/priority-steps.md` §2, §6, §11 | [`../../potential-partners`](../../potential-partners) | Letter 2's framing, its five addresses, and the meeting location |
| `applications/app-10-ucsd-moores-engine/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The prior request letter 1 escalates rather than repeats |
| `applications/app-06-fnih-amp/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The consortium framing reused in letters 3 and 4 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../move-in`](../../move-in) | The packet's Table 14 and Figure 8's campus split |
| `03Sep26/briefs/brief-03-use-of-proceeds.md` | [`../03Sep26`](../03Sep26) | The $420,000 site start-up line this day spends against |

## Positioning, carried into every file in this directory

Nothing here is an agreement and no agreement of any kind exists with UC San
Diego, with Moores Cancer Center, with Scripps Research, with Scripps Health, or
with any other institution. Both institutions are named at the feasibility stage
only, and neither is described as a partner, sponsor, site, or endorser. No
engineering detail is sent before a confidentiality agreement is in place. No
robotic configuration is specified; that is settled at a site agreement and not
before. Daraxonrasib is approved in the metastatic setting and is nowhere
described as first in human; the perioperative use this program proposes remains
investigational.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
