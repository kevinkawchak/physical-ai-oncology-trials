# Sub-prompt 5 - 02Oct26, The Week's Record (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8C5A12.svg)](../../02Oct26)
[![Accent](https://img.shields.io/badge/Accent-Sunset%20Cliffs%20Amber%20%238C5A12-8C5A12.svg)](../../02Oct26/packet)
[![Emails](https://img.shields.io/badge/Emails-3-6C757D.svg)](../../02Oct26/emails)
[![LinkedIn](https://img.shields.io/badge/LinkedIn%20replies-2-6C757D.svg)](../../02Oct26/linkedin)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../02Oct26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-1-6C757D.svg)](../../02Oct26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../02Oct26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../02Oct26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../02Oct26/packet)
[![Commits](https://img.shields.io/badge/Commits-25%2B-9AA1A8.svg)](#commit-order)

The day that spends no new signal. It sends only the follow-ups that have earned
their interval, advances the two applicant threads if the applicants have
answered, and closes the week with one record of what was asked of whom.

## The single decision this day asks for

**Does the chief executive approve sending the three earned follow-ups, answering
each applicant who has replied, and adopting the weekly cadence as the rule for
every week that follows?**

## Why this day is shaped the way it is

Four signals arrived in one week, and four days answered them with sixteen
letters, two LinkedIn replies and three submissions. A fifth day of new letters
would bury the first sixteen. The fifth day therefore applies the first of the
three priorities from the September 12 message to the company's own
correspondence: it waits for the offices it wrote to, and it follows up only
where the rule allows.

The rule is three business days, counted only on days the recipient's office was
open. By Friday, the Monday letters have had three full business days and the
Tuesday letters exactly three. Letters sent on Wednesday and Thursday have not
yet earned a follow-up, and a notice that asked for nothing never earns one. Of
the letters that have earned one, three are sent, each only if no reply has
arrived and each carrying one new fact.

## What this day produces

| # | Deliverable | Format | Recipient |
|:--|:--|:--|:--|
| 1 | `emails/email-01-nci-sbir-specific-aims.txt` | `.txt` | NCI SBIR Development Center, copied to the NIH SBIR/STTR program office and NIH SEED |
| 2 | `emails/email-02-sec-san-francisco-follow-up.txt` | `.txt` | SEC San Francisco Regional Office, copied to Los Angeles and the small business advocate |
| 3 | `emails/email-03-fda-oce-follow-up.txt` | `.txt` | FDA Oncology AI Program, copied to the Oncology Center of Excellence and Combination Products |
| 4 | `linkedin/linkedin-03-applicant-a-follow-up.txt` | `.txt` | Applicant A, only if Applicant A has replied |
| 5 | `linkedin/linkedin-04-applicant-b-follow-up.txt` | `.txt` | Applicant B, only if Applicant B has replied |
| 6 | `briefs/brief-01-the-week-in-one-record.md` | `.md` | Any reader who needs the whole week on one page |
| 7 | `briefs/brief-02-the-robotic-phase-1-in-numbers.md` | `.md` | A funder, and the program director letter 1 asks for |
| 8 | `briefs/brief-03-the-weekly-cadence.md` | `.md` | The chief executive and every future team member |
| 9 | `forms/form-01-sbir-gov-company-registry.md` | `.md` | The SBIR.gov company registry |
| 10 | `investing/capital-05-week-close-ledger.md` | `.md` | The chief executive |
| 11 | `diagrams/fig-13` .. `fig-15` | `.md` | The author |
| 12 | `packet/` | `.tex`, `.pdf`, `.zip` | Attached to letter 1 |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform |
|:--|:--|:--|:--|
| 13 | Graphviz | Record nodes in a chain, one rank per day | Each signal, letter and pending answer is a record with the same fields |
| 14 | Diagrams | Clustered topology with glyph tiles | The weekly cadence is a set of places and the paths letters take between them |
| 15 | Mermaid | Flowchart with decisions | The follow-up rule is a sequence of yes or no questions |

D2 and PlantUML are not used on day 5.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 21 | The week's four signals, each with its letters and what is pending | 4.6 cm |
| 22 | The day's action register | 6.4 cm |
| 23 | The follow-up ledger: every letter of the week against the rule | 4.7 cm |
| 24 | The robotic Phase 1 in numbers, for a funder | 7.3 cm |
| 25 | The funding routes opened this week, with amount and next step | 4.4 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Sunset Cliffs Amber `#8C5A12`, with `#B98A45` and `#F4EADB` as its two lighter shades |
| 2 | Addresses | Three per email, all in [`../../inputs`](../../inputs) |
| 3 | Follow-ups | Only after three open business days, only with no reply, and only with one new fact |
| 4 | The SEC office | No word attributed to it in the follow-up; the earlier letter's quotation is not repeated |
| 5 | Applicants | Never named; identical text for both; nothing sent to an applicant who has not replied |
| 6 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 7 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 8 | Dialect and punctuation | La Jolla usage; single hyphens only |
| 9 | Rasters | None |

## Commit order

| Order | Commit |
|:--|:--|
| 1 | This sub-prompt README |
| 2 | `02Oct26/README.md` |
| 3 | `emails/README.md` |
| 4 to 6 | The three letters, one commit each |
| 7 to 8 | `linkedin/`, the README and the two replies |
| 9 to 12 | `briefs/`, `forms/`, `investing/`, `diagrams/` |
| 13 to 16 | `packet/main.tex`, `fundstyle.sty`, `references.bib`, `packet/README.md` |
| 17 to 23 | `sec-00` through `sec-06`, one commit each |
| 24 | `packet/sections/README.md` |
| 25 | `main.pdf` and `02Oct26-packet-LaTeX.zip` |

## Rule 5 source map

| Used | From | Where it appears in day 5 |
|:--|:--|:--|
| Every `emails/` and `linkedin/` file of days 1 to 4 | [`../..`](../..) | The follow-up ledger, Table 23 |
| `08Sep26/briefs/brief-02-weekly-cadence.md` | [`../../../auto-fund`](../../../auto-fund) | The cadence, version 2 |
| `30Sep26/briefs/brief-03-the-early-rise-weekday.md` | [`../../30Sep26/briefs`](../../30Sep26/briefs) | The weekday blocks in Figure 14 |
| `trial-protocol/`, `trial-ind/` | repository root | Table 24 |
| `final-capital/sections/sec-03-gate-and-programme.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 25 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | The cost and staffing rows of Table 24 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
