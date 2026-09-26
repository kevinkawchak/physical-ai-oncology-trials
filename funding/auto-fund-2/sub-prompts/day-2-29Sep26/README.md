# Sub-prompt 2 - 29Sep26, The External Standard Request (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-34495E.svg)](../../29Sep26)
[![Accent](https://img.shields.io/badge/Accent-Harbor%20Slate%20%2334495E-34495E.svg)](../../29Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](../../29Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../29Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-1-6C757D.svg)](../../29Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../29Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../29Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../29Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

The day after the SEC San Francisco Regional Office wrote to the chief executive
that it is taking his request "very seriously." The request is that
ChemicalQDevice's oncology clinical trial research be listed as an external
standard for substantial oncology clinical trial research, on the basis that the
company's LLM papers have been used extensively by OpenAI and Anthropic.

## The single decision this day asks for

**Does the chief executive approve answering the SEC office with the full dated
record, routing the clinical-standard question to the FDA's oncology AI
program, and notifying the three companies the request names or concerns?**

## Why this day is shaped the way it is

A regulator's office that takes a request seriously has done the most it can do
in a first reply. What it needs next is a record it can evaluate, and the
cleanest record is a dated list of deposited works with persistent identifiers.
That is letter 1.

The SEC regulates securities markets; it does not set clinical research
standards. The federal body whose remit covers AI in oncology trials is the
FDA's Oncology Center of Excellence, which runs an Oncology AI Program. Writing
to it in parallel, and saying so to the SEC, is how a small company shows it
understands which office owns which question. That is letter 2.

The request names OpenAI and Anthropic, and the work it rests on concerns a
molecule owned by Revolution Medicines, a public company. A company that names
third parties in a request to a federal regulator tells them first. Those are
letters 3, 4 and 5, each of which offers to correct the record if the recipient
thinks any part of it is inaccurate.

## What this day produces

| # | Deliverable | Format | Recipient |
|:--|:--|:--|:--|
| 1 | `emails/email-01-sec-san-francisco-reply.txt` | `.txt` | SEC San Francisco, SEC Los Angeles, SEC small business advocate |
| 2 | `emails/email-02-fda-oce-oncology-ai.txt` | `.txt` | FDA OCE Oncology AI Program, FDA OCE, FDA Office of Combination Products |
| 3 | `emails/email-03-anthropic-notice.txt` | `.txt` | Anthropic press and legal, copied to the SEC office |
| 4 | `emails/email-04-openai-notice.txt` | `.txt` | OpenAI press and general inquiries, copied to the SEC office |
| 5 | `emails/email-05-revolution-medicines-notice.txt` | `.txt` | Revolution Medicines investor relations, media, medical information |
| 6 | `briefs/brief-01-the-request-and-the-record.md` | `.md` | The SEC office, if asked for more |
| 7 | `briefs/brief-02-what-an-external-standard-requires.md` | `.md` | The FDA program and any technical reviewer |
| 8 | `briefs/brief-03-the-openai-and-anthropic-statement.md` | `.md` | Anyone who asks what "used extensively" means |
| 9 | `forms/form-01-edgar-access-readiness.md` | `.md` | SEC EDGAR Filer Management, prepared and held |
| 10 | `investing/capital-02-trading-and-disclosure-guardrails.md` | `.md` | The chief executive and the brokerage |
| 11 | `diagrams/fig-04` .. `fig-06` | `.md` | The author |
| 12 | `packet/` | `.tex`, `.pdf`, `.zip` | Attached to letters 1 and 2 |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform |
|:--|:--|:--|:--|
| 4 | PlantUML | State machine with guards | A request moves through states, and the guards are the conditions the company controls |
| 5 | Mermaid | Gantt chart | Independence and simultaneity are claims about time, and a gantt shows two lanes on one axis |
| 6 | D2 | Grid | Eight criteria against what the record holds is a table with status cells |

Diagrams and Graphviz are not used on day 2.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 6 | The deposited record, with dates and identifiers | 6.4 cm |
| 7 | The day's action register | 6.4 cm |
| 8 | The developer's trials and the company's papers, side by side | 5.4 cm |
| 9 | Eight criteria for an external standard against the record | 5.0 cm |
| 10 | The six checkable quantities and the Phase 1 design | 3.6 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Harbor Slate `#34495E`, with `#6B7F94` and `#E3E8EE` as its two lighter shades |
| 2 | Addresses | Three per email, all in [`../../inputs`](../../inputs) |
| 3 | Paste geometry | One paragraph per line in every letter body |
| 4 | The office's words | "very seriously," quoted exactly, never extended, never called a finding |
| 5 | The OpenAI and Anthropic statement | The company's own statement, made as the basis of its request; never presented as either company's endorsement |
| 6 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 7 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 8 | Dialect and punctuation | La Jolla usage; single hyphens only |
| 9 | Rasters | None |

## Commit order

| Order | Commit |
|:--|:--|
| 1 | This sub-prompt README |
| 2 | `29Sep26/README.md` |
| 3 | `29Sep26/emails/README.md` |
| 4 to 8 | The five letters, one commit each |
| 9 to 12 | `briefs/`, `forms/`, `investing/`, `diagrams/` |
| 13 to 16 | `packet/main.tex`, `fundstyle.sty`, `references.bib`, `packet/README.md` |
| 17 to 23 | `sec-00` through `sec-06`, one commit each |
| 24 | `packet/sections/README.md` |
| 25 | `main.pdf` and `29Sep26-packet-LaTeX.zip` |

## Rule 5 source map

| Used | From | Where it appears in day 2 |
|:--|:--|:--|
| `tripartisan-llm-support.md` | [`../../..`](../../..) | Brief 3, letters 3 and 4, and §4 of the packet |
| `daraxonrasib-llm-story.md` | [`../../..`](../../..) | Table 6, Table 8 and Figure 5 |
| Root `README.md` release list | [`../../../../README.md`](../../../../README.md) | The deposit date of every work in Table 6 |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Table 10 |
| `03Sep26/forms/form-01-reg-d-506b-form-d.md` | [`../../../auto-fund`](../../../auto-fund) | The unfiled Form D that makes EDGAR access worth preparing |
| `02Sep26/investing/capital-01-treasury-ladder.md` | [`../../../auto-fund`](../../../auto-fund) | The Treasury-only rule the guardrails keep |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
