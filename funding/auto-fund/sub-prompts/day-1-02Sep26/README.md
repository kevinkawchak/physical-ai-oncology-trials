# Sub-prompt 1 - 02Sep26, Approval to Ask (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-0E5C63.svg)](../../02Sep26)
[![Accent](https://img.shields.io/badge/Accent-Pacific%20Teal%20%230E5C63-0E5C63.svg)](../../02Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-7-6C757D.svg)](../../02Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../02Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](../../02Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../02Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../02Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../02Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

The first business day after the approval week. Federal offices are open, the
equity and Treasury markets are open, and the five federal mechanisms already
approached between July 10 and August 8, 2026 have a new fact in front of them
that none of them had when they read the original inquiry.

## The single decision this day asks for

**Does the chief executive approve sending five re-contact letters that lead with
the approval rather than with the company?**

Everything in [`../../02Sep26`](../../02Sep26) is built so that a yes to that
question requires no further drafting: the addresses are filled, the subject
lines are written, the bodies are final, the attachments are named, and the
brokerage instruction is a single page with order types on it.

## Why this day leads with the mechanisms already contacted

Nine of the ten application file sets in
[`../../../pdac-funding-applications/applications`](../../../pdac-funding-applications/applications)
were emailed between July 10 and August 8, 2026. A re-contact after a material
change is a normal and welcome event for a program officer; a fresh introduction
five weeks later is not. So this day writes no cold letter. Every one of the five
emails opens by naming the earlier inquiry, states the one fact that changed, and
asks a question the earlier letter could not have asked.

## What this day produces

| # | Deliverable | Format | Recipient class |
|:--|:--|:--|:--|
| 1 | `emails/email-01-nih-seed-sbir-recontact.txt` | `.txt` | NIH SEED and SBIR program staff |
| 2 | `emails/email-02-arpa-h-mission-office.txt` | `.txt` | ARPA-H mission office |
| 3 | `emails/email-03-nci-ctep-concept.txt` | `.txt` | NCI Cancer Therapy Evaluation Program |
| 4 | `emails/email-04-nih-pioneer-eligibility.txt` | `.txt` | NIH Common Fund and CSR review contacts |
| 5 | `emails/email-05-brokerage-treasury-instruction.txt` | `.txt` | The company's broker-dealer |
| 5a | `emails/email-06-nci-ctep-gore-reply.txt` | `.txt` | Chief, Investigational Drug Branch, CTEP |
| 5b | `emails/email-07-nih-pioneer-labosky-reply.txt` | `.txt` | Program Leader, High-Risk, High-Reward Research |
| 6 | `briefs/brief-01-approval-delta.md` | `.md` | Technical reviewers who read plain text |
| 7 | `briefs/brief-02-sbir-phase-i-readiness.md` | `.md` | An SBIR program officer's technical staff |
| 8 | `briefs/brief-03-evidence-one-page.md` | `.md` | Any reviewer who wants the numbers alone |
| 9 | `forms/form-01-sam-gov-entity-validation.md` | `.md` | SAM.gov entity record |
| 10 | `forms/form-02-sba-company-registry.md` | `.md` | SBA Company Registry |
| 11 | `investing/capital-01-treasury-ladder.md` | `.md` | The chief executive and the broker |
| 12 | `diagrams/fig-01` .. `fig-03` | `.md` | The author, when a figure needs correction |
| 13 | `packet/` | `.tex`, `.pdf`, `.zip` | Every recipient above, as the attachment |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform and no other |
|:--|:--|:--|:--|
| 1 | Mermaid | Flowchart with a branch | The subject is a decision that splits before and after a date, and a flowchart is the only one of the five that reads left to right as a change in state |
| 2 | Graphviz | Record nodes in a cluster | Five mechanisms each carrying four fields is a record table, which is what a Graphviz record node is for |
| 3 | D2 | Grid with a measure panel | Cash against three claims is a small table with a scale beside it, which is D2's grid plus an interval strip |

PlantUML and Diagrams are deliberately unused on day 1 and appear on days 2, 3,
4 and 5. No platform is used twice in one day.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 1 | What the approval changed, in four rows | 4.6 cm |
| 2 | The five mechanisms, their date of contact, and the question now asked | 3.4 cm |
| 3 | The six checkable quantities with their stated limitations | 4.2 cm |
| 4 | The Treasury ladder with maturity, yield basis, and settlement | 3.0 cm |
| 5 | The SBIR route against the five-year program, with the delta | 3.8 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Pacific Teal `#0E5C63`, with `#2F8A93` and `#DCEBEC` as its two lighter shades |
| 2 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 3 | Caption lines | Two, balanced within a small character spread |
| 4 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 5 | Money | The frame in [`../../inputs`](../../inputs), not re-derived |
| 6 | Dates in body text | The date appears on the cover and in the mechanism table only |
| 7 | Dialect | American English, La Jolla usage; no `programme`, no `centre`, no `organisation` |
| 8 | Punctuation | Single hyphens only; no em dash, no double dash, no triple dash |
| 9 | Rasters | None |

## Commit order

| Order | Commit |
|:--|:--|
| 1 | `02Sep26/README.md` |
| 2 | `02Sep26/emails/README.md` |
| 3 to 7 | The five `.txt` re-contact emails, one commit each |
| 8 | `02Sep26/briefs/` |
| 9 | `02Sep26/forms/` |
| 10 | `02Sep26/investing/` |
| 11 | `02Sep26/diagrams/` |
| 12 to 15 | `packet/main.tex`, `packet/fundstyle.sty`, `packet/references.bib`, `packet/README.md` |
| 16 to 22 | `packet/sections/sec-00` .. `sec-06`, one commit each |
| 23 | `packet/sections/README.md` |
| 24 | `packet/main.pdf` and `packet/02Sep26-packet-LaTeX.zip` |
| 25, 26 | Letters 6 and 7, added after two substantive replies came back. A reply is not scheduled in advance: it exists because a recipient answered, and it is committed when it is written |

## Rule 5 source map

| Used | From | Where it appears in day 1 |
|:--|:--|:--|
| `applications/app-05-nih-sbir-seed/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Email 01's opening reference and the $306,000 and $1,300,000 split in §4 |
| `applications/app-02-arpa-h/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Email 02's three-gate structure |
| `applications/app-08-nci-ctep/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Email 03's concept-submission framing |
| `applications/app-01-nih-pioneer-award/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Email 04's two questions, restated against the approval |
| `UC-San-Diego/priority-steps.md` §13 and §14 | [`../../../potential-partners`](../../../potential-partners) | The registration list in both form packs and the NIH addresses in email 04 |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 3 and `briefs/brief-03-evidence-one-page.md` |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 5 and Figure 3's money column |
| `daraxonrasib-llm-story.md` | [`../../..`](../../..) | §1 of the packet, quoted rather than paraphrased |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
