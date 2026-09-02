# 02Sep26 - Day 1, Approval to Ask (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-0E5C63.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-0E5C63.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](forms)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Approval%20Dividend-0E5C63.svg)](packet)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Federal offices are open. The Treasury and equity markets are open. Five federal
mechanisms already hold an inquiry from this company, sent between July 10 and
August 8, 2026, and every one of those inquiries was written before the FDA
approved Rasonque (daraxonrasib) on August 26, 2026.

This day is the re-contact.

## The one approval step

Everything in this directory is finished. The addresses are filled, the subject
lines are written, the bodies are final, the attachments are named, and the
brokerage instruction carries order types and limits. One decision is open:

> **Approve sending the five letters in [`emails/`](emails), and approve the
> Treasury instruction in [`investing/`](investing).**

Nothing in this directory has been sent, filed, entered, or agreed.

## The run order, and how long each item takes

| Order | Item | Where | Time | Depends on |
|:--|:--|:--|:--|:--|
| 1 | Read the packet's §2, the action register | [`packet/`](packet) | 6 minutes | Nothing |
| 2 | Approve or amend the five letters | [`emails/`](emails) | 15 minutes | Item 1 |
| 3 | Compile the packet PDF and confirm it opens | [`packet/`](packet) | 3 minutes | Item 2 |
| 4 | Send letters 1 to 4 | [`emails/`](emails) | 10 minutes | Items 2, 3 |
| 5 | Confirm the entity record is current before letter 5 | [`forms/`](forms) | 20 minutes | Nothing |
| 6 | Send letter 5, the Treasury instruction | [`emails/`](emails) | 4 minutes | Items 2, 5 |
| 7 | File the two form packs | [`forms/`](forms) | 35 minutes | Item 5 |

The whole day is a little over an hour and a half of the chief executive's time,
and no item in it requires drafting.

## The five letters, and what each newly asks

Not one of these is a fresh introduction. Each names the earlier inquiry, states
the one fact that changed, and asks a question the earlier letter could not have
asked.

| # | File | To | The new question |
|:--|:--|:--|:--|
| 1 | [`email-01-nih-seed-sbir-recontact.txt`](emails/email-01-nih-seed-sbir-recontact.txt) | NIH SEED and SBIR program staff | Whether an approved agent changes the Phase I feasibility bar for a combined workflow |
| 2 | [`email-02-arpa-h-mission-office.txt`](emails/email-02-arpa-h-mission-office.txt) | ARPA-H mission office | Whether gate 1 can now be reduced in scope and cost, since drug risk left the program |
| 3 | [`email-03-nci-ctep-concept.txt`](emails/email-03-nci-ctep-concept.txt) | NCI Cancer Therapy Evaluation Program | Whether an approved agent used perioperatively is in or out of the concept pathway |
| 4 | [`email-04-nih-pioneer-eligibility.txt`](emails/email-04-nih-pioneer-eligibility.txt) | NIH Common Fund and CSR review contacts | Whether the science codes and the effort expectation change with the drug risk removed |
| 5 | [`email-05-brokerage-treasury-instruction.txt`](emails/email-05-brokerage-treasury-instruction.txt) | The company's broker-dealer | Execution of a Treasury ladder against a known nine-month Phase I horizon |

## What changed on August 26, 2026, stated once

The FDA approved Rasonque (daraxonrasib), Revolution Medicines' RAS(ON)
multi-selective inhibitor, as the first-in-class targeted therapy for metastatic
pancreatic cancer.

| Dimension | Before | After |
|:--|:--|:--|
| Regulatory status of the agent | Investigational, in Phase 3 evaluation | Approved, labeled, commercially supplied in the metastatic setting |
| What a funder underwrites | Drug risk, device risk, workflow risk | Device risk and workflow risk, in an approved-agent setting |
| The 2025 simulation | A hypothesis about a molecule | A dated public call on a molecule that later cleared the FDA |
| The supply question | Whether an investigational agent can be obtained at all | Whether a perioperative investigational use of an approved agent can be supported |

The fourth row is the honest one and it is stated in every letter. Approval in
the metastatic setting is not approval of the perioperative use this program
proposes. The approval removes a class of risk; it does not remove the
investigational new drug application.

## Directory contents

```
02Sep26/
├── README.md              this approval sheet
├── emails/                5 .txt letters, each with its own pre-send checklist
├── briefs/                3 .md technical briefs for plain-text readers
├── forms/                 2 .md field-by-field online form packs
├── investing/             1 .md capital instruction with order types and limits
├── diagrams/              3 .md figure specifications, one per figure
└── packet/                The Approval Dividend: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 02Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `applications/app-05-nih-sbir-seed/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | Email 01, and the $306,000 and $1,300,000 split in the packet §4 |
| `applications/app-02-arpa-h/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | Email 02's three-gate structure and its $2,100,000 frame |
| `applications/app-08-nci-ctep/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | Email 03's concept-submission framing |
| `applications/app-01-nih-pioneer-award/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | Email 04's two questions and its five-year $700,000 frame |
| `applications/emailed-source/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The July 10 to August 8 send dates every letter refers to |
| `UC-San-Diego/priority-steps.md` §13, §14 | [`../../potential-partners`](../../potential-partners) | Both form packs and the NIH addresses in email 04 |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../capitalization-plan`](../../capitalization-plan) | `briefs/brief-03-evidence-one-page.md` and packet Table 3 |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../capitalization-plan`](../../capitalization-plan) | Packet Table 5 and the money column of Figure 3 |
| `final-capital/capstyle.sty` | [`../../capitalization-plan`](../../capitalization-plan) | `packet/fundstyle.sty`, recolored and with the stick-figure macro deleted |
| `daraxonrasib-llm-story.md` | [`../..`](../..) | The packet §1 chronology, quoted rather than paraphrased |

## Positioning, carried into every file in this directory

Nothing here is a submission of record and nothing here is an agreement. No
letter describes daraxonrasib as first in human. No letter describes the approval
as covering the perioperative use this program proposes. No institution is
described as a partner, sponsor, site, or endorser. No order in
[`investing/`](investing) has been placed, and nothing in that file is investment
advice; the instruments named are candidates for a treasury policy the chief
executive sets.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
