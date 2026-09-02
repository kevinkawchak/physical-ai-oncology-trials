# 03Sep26 - Day 2, The Private Capital Bridge (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-1B3A5C.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-1B3A5C.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](forms)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Private%20Capital%20Bridge-1B3A5C.svg)](packet)
[![Offer](https://img.shields.io/badge/Offer%20or%20solicitation-none-9AA1A8.svg)](#the-securities-rule-that-governs-this-whole-day)

Day 1 asked the federal side for the $1,606,000 the SBIR route can supply. This
day addresses the $2,104,000 it cannot, the $5,900,000 private position that
would close it, and the one commercial relationship the approval makes newly
askable.

## The one approval step

> **Approve one instrument, one raise size, and one first approach to the drug
> developer.**

The three candidate instruments are set out side by side at the same raise size,
so the choice is a comparison rather than a judgment call. The developer letter
asks for a conversation and a submission window, and asks for nothing that a
closed window would have to refuse.

Nothing in this directory has been sent, filed, offered, or agreed.

## The securities rule that governs this whole day

Every file in this directory describes instruments **under consideration**. None
of it is an offer to sell or a solicitation of an offer to buy any security, and
none of it is investment advice. A public repository that describes a private
placement has to be readable as a plan and not as an offer, so that sentence
appears in every file here and on the cover of the packet.

Two consequences follow and both are honored throughout:

| Consequence | How it is honored |
|:--|:--|
| No terms are held out to the public as available | Valuation caps, discounts and sizes appear as candidate ranges under comparison, never as an offered term |
| No general solicitation | The outreach letters ask for an introduction and a conversation. None of them describes an offering, and Rule 506(b) permits no general solicitation at all |

## The run order

| Order | Item | Where | Time | Depends on |
|:--|:--|:--|:--|:--|
| 1 | Read the packet §3 and pick one instrument | [`packet/`](packet) | 12 minutes | Nothing |
| 2 | Approve the raise size against the use of proceeds | [`briefs/`](briefs) | 8 minutes | Item 1 |
| 3 | Send the developer letter | [`emails/`](emails) | 6 minutes | Nothing |
| 4 | Send the two investor introduction letters | [`emails/`](emails) | 10 minutes | Items 1, 2 |
| 5 | Send the counsel engagement letter | [`emails/`](emails) | 5 minutes | Items 1, 2 |
| 6 | Read the two filing packs, file nothing yet | [`forms/`](forms) | 15 minutes | Item 1 |
| 7 | Send the brokerage account letter | [`emails/`](emails) | 4 minutes | Item 1 |

The two filing packs are **read** on this day and filed only after a first sale.
Form D is due within fifteen days of the first sale, not before it, and filing
early creates a public record of an offering that has not happened.

## The five letters

| # | File | To | The ask |
|:--|:--|:--|:--|
| 1 | [`email-01-revolution-medicines-external-research.txt`](emails/email-01-revolution-medicines-external-research.txt) | Developer external research, grants, medical information, business development | A scientific feasibility conversation and the next submission window |
| 2 | [`email-02-san-diego-angel-syndicate.txt`](emails/email-02-san-diego-angel-syndicate.txt) | Regional angel and syndicate leads | A screening conversation, not a pitch slot |
| 3 | [`email-03-life-science-family-office.txt`](emails/email-03-life-science-family-office.txt) | Life science family offices and evergreen holders | A read of the repository, then a call |
| 4 | [`email-04-securities-counsel-engagement.txt`](emails/email-04-securities-counsel-engagement.txt) | Securities counsel | An engagement scoped to one instrument and two filings |
| 5 | [`email-05-brokerage-corporate-account.txt`](emails/email-05-brokerage-corporate-account.txt) | The broker-dealer corporate desk | A subscription account and the reserve re-cut |

## Why the developer letter is on day 2 and not day 1

`../../potential-partners/UC-San-Diego/priority-steps.md` §2 records the request
prepared for Revolution Medicines and notes that the public external-research
portal indicated submissions were closed. The approval changes this company's
posture, not that portal's calendar. So the letter asks for the next submission
window and a scientific conversation, and it asks for nothing a closed window
would have to refuse.

It sits after the federal re-contacts because a developer reasonably asks who is
funding the work, and day 1 is the answer to that question.

## Directory contents

```
03Sep26/
├── README.md              this approval sheet
├── emails/                5 .txt letters, each with its own pre-send checklist
├── briefs/                3 .md technical briefs
├── forms/                 2 .md filing packs, read now and filed after a first sale
├── investing/             1 .md reserve re-cut against the chosen instrument
├── diagrams/              3 .md figure specifications
└── packet/                The Private Capital Bridge: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 03Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The $5,900,000 position, the tranche table, Figures 4 and 5 |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The $2,104,000 delta in §1 and Table 8 |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The cost-of-disclosure paragraph in `briefs/brief-03-use-of-proceeds.md` |
| `UC-San-Diego/priority-steps.md` §2 | [`../../potential-partners`](../../potential-partners) | Letter 1's addresses, its seven requests, and its attachment limit |
| `UC-San-Diego/priority-steps.md` §12 | [`../../potential-partners`](../../potential-partners) | The sponsor and IND-holder questions in `briefs/brief-02-firewall-and-part-54.md` |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../move-in`](../../move-in) | The federal versus non-federal separation in `briefs/brief-03-use-of-proceeds.md` |
| `02Sep26/forms/form-02-sba-company-registry.md` | [`../02Sep26`](../02Sep26) | The ownership question every instrument in §3 is scored against |
| `02Sep26/investing/capital-01-treasury-ladder.md` | [`../02Sep26`](../02Sep26) | The reserve this day re-cuts |

## Positioning, carried into every file in this directory

Nothing here is an offer, a solicitation, a commitment, or an agreement. No term
sheet, SAFE, or subscription agreement exists. No drug supply agreement, letter
of authorization, or regulatory cross-reference is in place with the agent's
developer, and letter 1 is the first approach of any kind. No institution is
described as a partner, sponsor, site, or endorser. Daraxonrasib is approved in
the metastatic setting and is nowhere described as first in human; the
perioperative use this program proposes remains investigational.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
