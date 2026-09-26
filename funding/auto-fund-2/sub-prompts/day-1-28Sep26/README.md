# Sub-prompt 1 - 28Sep26, The Unsolicited Applicants (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-2E5E4E.svg)](../../28Sep26)
[![Accent](https://img.shields.io/badge/Accent-Torrey%20Pine%20%232E5E4E-2E5E4E.svg)](../../28Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-3-6C757D.svg)](../../28Sep26/emails)
[![LinkedIn](https://img.shields.io/badge/LinkedIn%20replies-2-6C757D.svg)](../../28Sep26/linkedin)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../28Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](../../28Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../28Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../28Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../28Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

The first business day after the week in which two people wrote to the chief
executive on LinkedIn asking to work for ChemicalQDevice. The company had posted
no position and published no hiring information. Both found it through its
public work alone.

## The single decision this day asks for

**Does the chief executive approve the same reply to both applicants, and the
three letters that open routes to fund them?**

Everything in [`../../28Sep26`](../../28Sep26) is built so that a yes requires no
further drafting. The two replies are written, the three letters carry verified
addresses, the employer obligations are listed with their triggers, and the
payroll earmark is sized by formula rather than by guess.

## Why this signal is spent first

Of the four signals in the week, this is the only one that changes what a funder
can score. Every federal small-business reviewer asks whether the company can
build the team the budget pays for. Until now the answer was a staffing plan,
eleven roles at 3.95 award-funded full-time equivalents
([`../../../move-in`](../../../move-in)). Two unsolicited inquiries turn part of
that plan into a pipeline, and they arrived before the company asked anyone.

The day also comes first because the other three signals depend on it. The SEC
request, the White House correspondence and the NSF inquiry all ask, in
different words, whether this company can operate inside a structure. The first
two people who asked to join it are the first test of that answer.

## What this day produces

| # | Deliverable | Format | Recipient |
|:--|:--|:--|:--|
| 1 | `linkedin/linkedin-01-applicant-a-reply.txt` | `.txt` | The first applicant, in the LinkedIn conversation they opened |
| 2 | `linkedin/linkedin-02-applicant-b-reply.txt` | `.txt` | The second applicant, in the same way |
| 3 | `emails/email-01-nih-sbir-contingent-personnel.txt` | `.txt` | NIH SBIR/STTR Program Office, NCI SBIR Development Center, NIH SEED |
| 4 | `emails/email-02-first-hire-readiness.txt` | `.txt` | SBA Answer Desk, E-Verify Contact Center, California Competes |
| 5 | `emails/email-03-ucsd-research-staff-training.txt` | `.txt` | Moores Clinical Trials Office, ACTRI, UC San Diego IRB |
| 6 | `briefs/brief-01-why-two-inquiries-matter.md` | `.md` | A reviewer who scores the team |
| 7 | `briefs/brief-02-role-fit-and-structured-interview.md` | `.md` | The chief executive, before any conversation |
| 8 | `briefs/brief-03-contingent-personnel-in-applications.md` | `.md` | Whoever drafts the next federal application |
| 9 | `forms/form-01-edd-employer-registration.md` | `.md` | California EDD e-Services for Business, held until a first payroll |
| 10 | `forms/form-02-i9-and-new-hire-reporting.md` | `.md` | Form I-9 and the California new hire report, held until a start date |
| 11 | `investing/capital-01-contingent-payroll-earmark.md` | `.md` | The chief executive and the brokerage |
| 12 | `diagrams/fig-01` .. `fig-03` | `.md` | The author |
| 13 | `packet/` | `.tex`, `.pdf`, `.zip` | Attached to letters 1 and 3 |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform |
|:--|:--|:--|:--|
| 1 | Mermaid | Flowchart with decisions | A reply that can end in three different places is a decision flow, read left to right |
| 2 | Diagrams | Clustered glyph topology | The eleven roles have locations and license requirements, which is what clustered tiles show |
| 3 | Graphviz | Record nodes | Four funding routes carrying the same five fields is a record table |

PlantUML and D2 are not used on day 1. No platform is used twice in one day.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 1 | What the two inquiries show a funder, and what they do not | 5.0 cm |
| 2 | The day's action register | 4.6 cm |
| 3 | The eleven roles, their FTE, and whether an unsolicited applicant can fill each | 5.2 cm |
| 4 | Employer obligations before and after a first hire, with trigger and deadline | 4.0 cm |
| 5 | Four routes that can fund a first hire | 3.2 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Torrey Pine `#2E5E4E`, with `#5E8C7C` and `#E1ECE7` as its two lighter shades |
| 2 | Addresses | Three per email, all in [`../../inputs`](../../inputs) |
| 3 | Paste geometry | One paragraph per line in every letter and LinkedIn body |
| 4 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 5 | Caption lines | Two, balanced within a small character spread |
| 6 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 7 | Privacy | Neither applicant is named in any file; the names live only in the LinkedIn conversations |
| 8 | Equal treatment | The two replies are identical in substance, and the interview plan asks both the same questions |
| 9 | Dialect | American English, La Jolla usage; no `programme`, no `centre`, no `organisation` |
| 10 | Punctuation | Single hyphens only; no em dash, no double dash, no triple dash |
| 11 | Rasters | None |

## Commit order

| Order | Commit |
|:--|:--|
| 1 | This sub-prompt README |
| 2 | `28Sep26/README.md` |
| 3 | `28Sep26/emails/README.md` |
| 4 | `28Sep26/linkedin/README.md` |
| 5, 6 | The two LinkedIn replies, one commit each |
| 7 to 9 | The three letters, one commit each |
| 10 | `briefs/` |
| 11 | `forms/` |
| 12 | `investing/` |
| 13 | `diagrams/` |
| 14 to 17 | `packet/main.tex`, `fundstyle.sty`, `references.bib`, `packet/README.md` |
| 18 to 24 | `sec-00` through `sec-06`, one commit each |
| 25 | `packet/sections/README.md` |
| 26 | `main.pdf` and `28Sep26-packet-LaTeX.zip` |

## Rule 5 source map

| Used | From | Where it appears in day 1 |
|:--|:--|:--|
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | Table 3, Figure 2, and the role list linked from both LinkedIn replies |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | The separation of award funds from the company reserve in the payroll earmark |
| `applications/app-05-nih-sbir-seed/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | Letter 1's Phase I budget and the $306,000 in Table 5 |
| `02Sep26/emails/email-01-nih-seed-sbir-recontact.txt` | [`../../../auto-fund`](../../../auto-fund) | Letter 1's reference to the earlier re-contact |
| `04Sep26/emails/email-01-ucsd-moores-escalation.txt` | [`../../../auto-fund`](../../../auto-fund) | Letter 3's reference to the earlier feasibility contact |
| `02Sep26/investing/capital-01-treasury-ladder.md` | [`../../../auto-fund`](../../../auto-fund) | The ladder the payroll earmark sits inside |
| `02Sep26/forms/form-01-sam-gov-entity-validation.md` | [`../../../auto-fund`](../../../auto-fund) | The NAICS codes carried into the EDD form pack |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
