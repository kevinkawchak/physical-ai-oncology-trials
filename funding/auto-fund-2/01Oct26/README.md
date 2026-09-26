# 01Oct26 - Day 4, The Pilot Inquiry (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-4B2A7B.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-4B2A7B.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-4-6C757D.svg)](emails)
[![Form packs](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](forms)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Pilot%20Inquiry-4B2A7B.svg)](packet)
[![Pilot](https://img.shields.io/badge/NSF%20pilot-%2420M%2C%20%241M%20to%20%243.75M%20awards-9AA1A8.svg)](#the-pilot-and-its-gate)

The National Science Foundation has asked whether ChemicalQDevice would apply to
a multi-million-dollar pilot program. This day answers the inquiry exactly, and
takes the first step on the only route that leads to it.

## The one approval step

> **Approve answering the NSF inquiry, testing the pilot's published eligibility
> gate in writing, and submitting the Project Pitch that opens the route.**

## The pilot and its gate

| Program | Amount | Who is eligible | The company today |
|:--|:--|:--|:--|
| Commercialization Readiness Pilot, run by NCTI at UCF | $20 million over two years; six to eight companies at $1 million to $3.75 million | Active or recent NSF SBIR/STTR Phase II or IIB awardees | No NSF award; not yet eligible |
| Strategic Breakthrough tier, NSF 26-510 | Up to $30 million, with a 1 to 1 match | Phase II awardees, after consulting the program officer | Not yet eligible |
| Scientific instrumentation pilot emphasis, NSF 26-511 | Part of a $40 million emphasis | SBIR/STTR proposers in scientific instrumentation | Outside the company's technology |
| SBIR Phase I, NSF 26-510 | Up to $305,000 over 6 to 18 months | Small businesses invited after a Project Pitch | Eligible to pitch today |

The inquiry is an invitation to consider applying. The honest answer to it is
that the company cannot enter any multi-million-dollar NSF pilot yet, and that
the published route to each one starts with the Project Pitch the company can
submit today. Letter 1 says exactly that, and asks whether the inquiry had a
different route in mind.

## What NSF funds, and what it does not

NSF's SBIR program funds technical innovation. It does not fund clinical trials.
The program's robotic Phase 1 therefore stays with the National Institutes of
Health and private capital, and the Project Pitch is written around the part of
the program NSF can fund: a governed, advisory-only language model layer for
robotic oncologic surgery, with verification by independent models and a public
audit trail.

## The run order

| Order | Item | Where | Depends on |
|:--|:--|:--|:--|
| 1 | Answer the NSF inquiry | [`emails/email-01-nsf-sbir-inquiry-reply.txt`](emails/email-01-nsf-sbir-inquiry-reply.txt) | Nothing |
| 2 | Confirm the pilot's gate with its principal investigator | [`emails/email-02-ncti-eligibility.txt`](emails/email-02-ncti-eligibility.txt) | Row 1 sent |
| 3 | Confirm the Research.gov organization record | [`forms/form-02-research-gov-organization.md`](forms/form-02-research-gov-organization.md) | An active SAM.gov registration |
| 4 | Submit the Project Pitch | [`forms/form-01-nsf-project-pitch.md`](forms/form-01-nsf-project-pitch.md) | Row 3 |
| 5 | Ask about NAIRR Pilot resources | [`emails/email-03-nairr-pilot-resources.txt`](emails/email-03-nairr-pilot-resources.txt) | Nothing |
| 6 | Disclose the NSF pitch to NIH | [`emails/email-04-nih-sbir-overlap-disclosure.txt`](emails/email-04-nih-sbir-overlap-disclosure.txt) | Row 4 submitted |
| 7 | Open the match ledger | [`investing/capital-04-match-ledger.md`](investing/capital-04-match-ledger.md) | Nothing |

## A note on the date

October 1 is the first day of the federal fiscal year. If appropriations have
lapsed, NSF staff may not answer for a time and the portals may accept
submissions they cannot yet process. The Project Pitch can still be submitted;
nothing else is followed up until three business days after the agency reopens.

## Directory contents

```
01Oct26/
├── README.md              this approval sheet
├── emails/                4 .txt letters, each with at least three verified addresses
├── forms/                 2 .md packs: the Project Pitch, Research.gov registration
├── briefs/                3 .md briefs: the gate, what NSF funds, the merit review map
├── investing/             1 .md capital instruction: the match ledger
├── diagrams/              3 .md figure specifications
└── packet/                The Pilot Inquiry: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 01Oct26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `inputs/README.md` | [`../inputs`](../inputs) | The NSF quantities in the table above |
| `applications/app-05-nih-sbir-seed/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The NIH route and letter 4 |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The private capital split |
| `tripartisan-llm-support.md` | [`../..`](../..) | The verification method in the Pitch |
| `02Sep26/forms/form-01-sam-gov-entity-validation.md` | [`../../auto-fund`](../../auto-fund) | The SAM.gov and UEI prerequisite |

## Positioning, carried into every file in this directory

An inquiry to apply is not an invitation to submit a proposal, an award, or an
assessment of eligibility, and no file here describes it as one. The company has
no NSF award and is not eligible for the Commercialization Readiness Pilot or the
Strategic Breakthrough tier today. NSF's SBIR program does not fund clinical
trials, and the Pitch does not ask it to. Rasonque is approved in the metastatic
setting, and the perioperative use this program proposes remains
investigational. Nothing in [`investing/`](investing) is investment advice.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
