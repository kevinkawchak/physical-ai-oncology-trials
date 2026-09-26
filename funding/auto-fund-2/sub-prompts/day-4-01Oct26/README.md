# Sub-prompt 4 - 01Oct26, The Pilot Inquiry (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-4B2A7B.svg)](../../01Oct26)
[![Accent](https://img.shields.io/badge/Accent-Pancreatic%20Purple%20%234B2A7B-4B2A7B.svg)](../../01Oct26/packet)
[![Emails](https://img.shields.io/badge/Emails-4-6C757D.svg)](../../01Oct26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../01Oct26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](../../01Oct26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../01Oct26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../01Oct26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../01Oct26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

The day that answers an inquiry from the National Science Foundation about
applying to a multi-million-dollar pilot program. It is also the first day of
the federal fiscal year.

## The single decision this day asks for

**Does the chief executive approve answering the NSF inquiry, testing the
pilot's published eligibility gate in writing, and submitting the Project Pitch
that opens the only route to it?**

## Why this day is shaped the way it is

In September 2026 NSF announced a $20 million, two-year Commercialization
Readiness Pilot, run by the National Commercialization and Translation Institute
at the University of Central Florida, that will select six to eight companies for
milestone-based awards of $1 million to $3.75 million each. Its competition is
open to active or recent NSF SBIR/STTR Phase II and IIB awardees. ChemicalQDevice
has no NSF award yet. The same agency's Strategic Breakthrough tier, up to $30
million, is likewise open only to Phase II awardees.

An inquiry to apply is an honor, and the correct answer to it is exact. The
company thanks NSF, states plainly that it does not yet meet the gate, asks
whether the inquiry contemplates a route it has missed, and submits the Project
Pitch that is the first step on the published route: Pitch, then Phase I, then
Phase II, then eligibility for the pilot.

NSF's SBIR program does not fund clinical trials. The Pitch is therefore written
around the technology NSF can fund, the governed, advisory-only language model
layer and its verification, and not around the trial, which remains an NIH and
private-capital question.

## What this day produces

| # | Deliverable | Format | Recipient |
|:--|:--|:--|:--|
| 1 | `emails/email-01-nsf-sbir-inquiry-reply.txt` | `.txt` | NSF SBIR, NSF Policy Office, Research.gov help desk |
| 2 | `emails/email-02-ncti-eligibility.txt` | `.txt` | The NCTI principal investigator at UCF, copied to NSF SBIR and the Policy Office |
| 3 | `emails/email-03-nairr-pilot-resources.txt` | `.txt` | NAIRR Pilot, NAIRR Operations Center, NSF SBIR |
| 4 | `emails/email-04-nih-sbir-overlap-disclosure.txt` | `.txt` | NIH SBIR/STTR Program Office, NCI SBIR, copied to NSF SBIR |
| 5 | `forms/form-01-nsf-project-pitch.md` | `.md` | The NSF Project Pitch portal |
| 6 | `forms/form-02-research-gov-organization.md` | `.md` | Research.gov organization registration |
| 7 | `briefs/brief-01-the-eligibility-gate.md` | `.md` | NSF program staff, the NCTI team |
| 8 | `briefs/brief-02-what-nsf-funds.md` | `.md` | Any reviewer asking how the program is split between agencies |
| 9 | `briefs/brief-03-merit-review-map.md` | `.md` | An NSF reviewer |
| 10 | `investing/capital-04-match-ledger.md` | `.md` | The chief executive |
| 11 | `diagrams/fig-10` .. `fig-12` | `.md` | The author |
| 12 | `packet/` | `.tex`, `.pdf`, `.zip` | Attached to letters 1 and 2 |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform |
|:--|:--|:--|:--|
| 10 | D2 | Layered stack | The NSF funding ladder is a set of layers, each gated by the one below |
| 11 | PlantUML | Activity with a fork and a join | Registration and drafting run in parallel and must join before the Pitch is submitted |
| 12 | Diagrams | Clustered infrastructure | Where compute, data and review sit if NAIRR resources are used |

Mermaid and Graphviz are not used on day 4.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 16 | The NSF tiers and pilots, with eligibility and the company's position | 4.2 cm |
| 17 | The day's action register | 6.4 cm |
| 18 | How the program is split between NSF, NIH and private capital | 5.3 cm |
| 19 | The Phase I technical objectives, each with a measure and a threshold | 5.0 cm |
| 20 | NSF merit review criteria against the program's evidence | 5.4 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Pancreatic Purple `#4B2A7B`, with `#7E62A8` and `#EAE4F3` as its two lighter shades |
| 2 | Addresses | Three per email, all in [`../../inputs`](../../inputs) |
| 3 | The pilot | Named only with its published eligibility gate beside it |
| 4 | What NSF funds | Technology, not clinical trials; the Pitch says so |
| 5 | Character limits | Every Pitch field measured against its limit before commit |
| 6 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 7 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 8 | Dialect and punctuation | La Jolla usage; single hyphens only |
| 9 | Rasters | None |

## Commit order

| Order | Commit |
|:--|:--|
| 1 | This sub-prompt README |
| 2 | `01Oct26/README.md` |
| 3 | `01Oct26/emails/README.md` |
| 4 to 7 | The four letters, one commit each |
| 8 to 11 | `briefs/`, `forms/`, `investing/`, `diagrams/` |
| 12 to 15 | `packet/main.tex`, `fundstyle.sty`, `references.bib`, `packet/README.md` |
| 16 to 22 | `sec-00` through `sec-06`, one commit each |
| 23 | `packet/sections/README.md` |
| 24 | `main.pdf` and `01Oct26-packet-LaTeX.zip` |

## Rule 5 source map

| Used | From | Where it appears in day 4 |
|:--|:--|:--|
| `inputs/README.md`, NSF quantities | [`../../inputs`](../../inputs) | Table 16 and brief 1 |
| `applications/app-05-nih-sbir-seed/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The NIH route in Table 18 and letter 4 |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The private capital column of Table 18 |
| `trial-ind/`, `trial-protocol/` | repository root | The technical objectives in Table 19 |
| `tripartisan-llm-support.md` | [`../../..`](../../..) | The verification objective in Table 19 and the Pitch |
| `02Sep26/forms/form-01-sam-gov-entity-validation.md` | [`../../../auto-fund`](../../../auto-fund) | The UEI step in the Research.gov pack |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
