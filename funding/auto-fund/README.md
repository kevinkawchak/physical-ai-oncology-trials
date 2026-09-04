# auto-fund - Daily Funding Actions for Final CEO Approval (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../README.md)
[![Business days](https://img.shields.io/badge/Business%20days-5-00417A.svg)](#the-five-business-days)
[![Approval trigger](https://img.shields.io/badge/Trigger-Rasonque%20approved%208%2F26%2F26-3C7DB2.svg)](https://www.fda.gov/news-events/press-announcements/fda-approves-first-class-targeted-therapy-metastatic-pancreatic-cancer)
[![Emails](https://img.shields.io/badge/Emails-26%20.txt-6C757D.svg)](#what-a-day-contains)
[![Briefs](https://img.shields.io/badge/Technical%20briefs-13%20.md-6C757D.svg)](#what-a-day-contains)
[![Forms](https://img.shields.io/badge/Form%20packs-9%20.md-6C757D.svg)](#what-a-day-contains)
[![Capital](https://img.shields.io/badge/Capital%20instruction%20sets-5-6C757D.svg)](#what-a-day-contains)
[![Packets](https://img.shields.io/badge/LaTeX%20packets-5%20%C3%97%207%20sections-9AA1A8.svg)](#the-five-latex-packets)
[![Figures](https://img.shields.io/badge/Figures-15-9AA1A8.svg)](#the-fifteen-figures)
[![Tables](https://img.shields.io/badge/Tables-25-9AA1A8.svg)](#the-five-latex-packets)
[![Compiler](https://img.shields.io/badge/pdfLaTeX-0%20errors-9AA1A8.svg)](#compile-record)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-A6CE39.svg)](https://orcid.org/0009-0007-5457-8667)

**A constant stream of funding actions for CEO Kevin Kawchak to make final
decisions on.** Each business day directory holds every piece of information a
single day's funding actions need: recipient addresses, subject lines, complete
email bodies, technical briefs, field-by-field online form content, named
securities with order types and limits, and one compiled packet a recipient can
be sent. Nothing here is sent, filed, ordered, or agreed. Each day ends in one
approval step, and that step belongs to the chief executive.

## What changed, and why this directory exists now

On August 26, 2026 the FDA approved **Rasonque (daraxonrasib)**, Revolution
Medicines' RAS(ON) multi-selective inhibitor, as the first-in-class targeted
therapy for metastatic pancreatic cancer
([FDA press announcement](https://www.fda.gov/news-events/press-announcements/fda-approves-first-class-targeted-therapy-metastatic-pancreatic-cancer)).

ChemicalQDevice has used that molecule to prove large language model utility
since June 2025, when Google Gemini ranked a daraxonrasib combination as the
company's top funding candidate out of forty pancreatic ductal adenocarcinoma
meta-analyses totaling over 400,000 words
([10.5281/zenodo.15735068](https://doi.org/10.5281/zenodo.15735068)). The
approval converts fourteen months of the company's work from a **prediction**
into a **record**, and that changes the ask rather than the science:

| Before August 26, 2026 | After August 26, 2026 |
|:--|:--|
| A funder was asked to accept an investigational agent's forward risk | The agent is approved, labeled, and commercially supplied |
| The 2025 simulation was a hypothesis about a molecule | The 2025 simulation is a dated, public, correct call on a molecule that later cleared the FDA |
| The company's method was argued | The company's method has an external checkpoint a reviewer can date |
| The remaining risk was drug plus device plus workflow | The remaining risk is device plus workflow, in an approved-agent setting |

Every action in this directory is written from that one shift. The five days do
not repeat it; they spend it.

## What a day contains

| Subdirectory | Format | What it holds | Who receives it |
|:--|:--|:--|:--|
| `emails/` | `.txt` | Recipient addresses, CC list, subject, introduction, body, closing, attachment manifest, and a pre-send checklist | Program officers, foundation staff, clinicians, corporate development, brokers |
| `briefs/` | `.md` | Lightly formatted technical text, tables, no LaTeX furniture | Technical reviewers who read plain text and repository files |
| `forms/` | `.md` | Field-by-field answers for one online submission, character counts checked against each portal's stated limit | Portals: SAM.gov, eRA Commons, SBA, foundation LOI systems, state programs |
| `investing/` | `.md` | Named instruments, tickers, CUSIPs where public, order types, limits, sizes, settlement and tax notes | The chief executive and the brokerage |
| `diagrams/` | `.md` | One specification per figure: native source, TikZ coordinates, value provenance, exact caption | The author, when a figure needs correction |
| `packet/` | `.tex`, `.sty`, `.bib`, `.pdf`, `.zip` | The compiled document of the day, seven sections, ready for Overleaf | Anyone who receives an attachment that day |

The format rule is fixed and is not varied by convenience. Anything a person
receives as correspondence is `.txt`. Anything a technical reader reads inline is
`.md`. Anything that is compiled into a PDF is `.tex` inside a zip for Overleaf.

## The five business days

| Day | Directory | Theme | Federal offices | US equity market |
|:--|:--|:--|:--|:--|
| 1 | [`02Sep26/`](02Sep26) | Approval to ask: re-contact the federal mechanisms already approached, and reply to the two responses | Open | Open |
| 2 | [`03Sep26/`](03Sep26) | Private capital: instruments, angels, family offices, the developer relationship | Open | Open |
| 3 | [`04Sep26/`](04Sep26) | Site and partner: Moores, Scripps, foundations, and the disease-specific funders | Open | Open |
| 4 | [`07Sep26/`](07Sep26) | Labor Day: nothing can be sent or traded, so everything is staged and queued | **Closed** | **Closed** |
| 5 | [`08Sep26/`](08Sep26) | Execution: release the queue, place the orders, submit the forms, set the cadence | Open | Open |

Day 4 is a US federal holiday and the NYSE and Nasdaq are closed. That is not a
gap in the schedule; it is the reason day 4 exists in the shape it does. A day on
which no counterparty can act is the correct day to build the data room, draft
what will be sent, and queue what will be traded, so that day 5 opens with work
already finished. Day 4's own README states this in full.

## The five LaTeX packets

Each day compiles one packet. All five share a structure so that a recipient who
receives two of them reads the same document twice, and all five differ in accent
color so the author can tell one from another on a desk.

| Day | Packet title | Accent | Sections | Figures | Tables |
|:--|:--|:--|:--|:--|:--|
| 1 | The Approval Dividend | Pacific Teal `#0E5C63` | 7 | 3 | 5 |
| 2 | The Private Capital Bridge | Harbor Navy `#1B3A5C` | 7 | 3 | 5 |
| 3 | The Site and Partner Package | Cypress Green `#2F5D3A` | 7 | 3 | 5 |
| 4 | The Staged Queue | Slate Plum `#5B3A5E` | 7 | 3 | 5 |
| 5 | The Execution Record | Ember Rust `#8A4B2A` | 7 | 3 | 5 |

Every packet uses the same `fundstyle.sty`, which differs between days only in
the palette block at the head of the file and in the three metadata macros. The
structural code below that block is byte-identical across the five, so a
correction made once can be applied five times without reading five files.

## The fifteen figures

| # | Day | Platform | Perspective |
|:--|:--|:--|:--|
| 1 | 1 | Mermaid | What the approval changes in the ask, as a before and after flow |
| 2 | 1 | Graphviz | The five federal mechanisms as records, with the state of each |
| 3 | 1 | D2 | The day's cash position against three claims on it |
| 4 | 2 | D2 | Three capital instruments side by side at the same raise size |
| 5 | 2 | PlantUML | The financing state machine and the four guards on it |
| 6 | 2 | Mermaid | A sequence of who signs what, in what order, on a private round |
| 7 | 3 | Graphviz | Site, sponsor, and developer obligations as three clusters |
| 8 | 3 | Diagrams | Where the trial's functions physically sit across two campuses |
| 9 | 3 | Mermaid | The foundation funnel from letter of intent to award |
| 10 | 4 | PlantUML | The staging activity, with the fork that a closed market forces |
| 11 | 4 | D2 | The data room as typed records with access class |
| 12 | 4 | Graphviz | What has to fail for the week to produce nothing |
| 13 | 5 | Mermaid | The execution timeline across one market session |
| 14 | 5 | Diagrams | The recurring weekly cadence this week hands to the next |
| 15 | 5 | D2 | The thirty-day pipeline as layers with a count on each |

Five platforms, three figures a day, no platform twice in one day. Every figure
is TikZ compiled from source in this repository. No PNG or JPG is generated
anywhere in this directory.

## Directory structure

```
funding/auto-fund/
├── README.md                     this hub
├── inputs/README.md              the repository sources every day reads
├── prompts/
│   ├── README.md
│   ├── prompt-auto-fund.md       the master prompt, verbatim
│   └── output-auto-fund.md       the full build output
├── sub-prompts/
│   ├── README.md                 the five-day schedule and its invariants
│   ├── day-1-02Sep26/README.md
│   ├── day-2-03Sep26/README.md
│   ├── day-3-04Sep26/README.md
│   ├── day-4-07Sep26/README.md
│   └── day-5-08Sep26/README.md
├── 02Sep26/  03Sep26/  04Sep26/  07Sep26/  08Sep26/
│   ├── README.md                 the day's approval sheet
│   ├── emails/                   README.md + the day's .txt correspondence
│   ├── briefs/                   README.md + the day's .md technical text
│   ├── forms/                    README.md + the day's .md form packs
│   ├── investing/                README.md + the day's .md capital instructions
│   ├── diagrams/                 README.md + one .md specification per figure
│   └── packet/
│       ├── README.md  main.tex  fundstyle.sty  references.bib
│       ├── sections/README.md + sec-00 .. sec-06.tex
│       ├── main.pdf
│       └── <day>-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `final-capital/capstyle.sty` | [`../capitalization-plan`](../capitalization-plan) | The five TikZ diagram vocabularies, the figure frame, the rigid caption invariant, the clickable DOI machinery |
| `final-new-trial/trialstyle.sty` | [`../../new-trial-system`](../../new-trial-system) | The `-0.60cm` spacing invariant and the two-line caption convention, both required by this prompt |
| `final-capital/references.bib` | [`../capitalization-plan`](../capitalization-plan) | The base bibliography every packet extends |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../capitalization-plan`](../capitalization-plan) | The six checkable quantities, reused without re-derivation |
| `sec-04-capital-bridge.tex`, `sec-03-gate-and-programme.tex` | [`../capitalization-plan`](../capitalization-plan) | The $306K / $1.3M / $1.606M / $2.104M / $5.9M frame, reused verbatim |
| `applications/app-01` .. `app-10` email files | [`../pdac-funding-applications`](../pdac-funding-applications) | The email file format: `FROM / TO / CC / SUBJECT / BODY / ATTACHMENTS / BEFORE SENDING` |
| `applications/emailed-source/` | [`../pdac-funding-applications`](../pdac-funding-applications) | Which mechanisms were already contacted, and when, so no day re-introduces the company |
| `UC-San-Diego/priority-steps.md` | [`../potential-partners`](../potential-partners) | Named clinical, contracting, IRB, ACTRI and FDA addresses used in days 1, 3 and 5 |
| `Scripps/priority-steps.md` | [`../potential-partners`](../potential-partners) | The Digital Trials Center route and the four named research contacts used in day 3 |
| `final-move-in/sections/sec-14`, `sec-15` | [`../move-in`](../move-in) | The eleven-role roster and the federal versus non-federal funds separation |
| `daraxonrasib-llm-story.md` | [`..`](..) | The June 2025 to August 2026 chronology quoted in every packet's §1 |
| `tripartisan-llm-support.md` | [`..`](..) | The three-model division of labor cited in the method notes |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | repository root | The asset register and the evidence chain each packet cites |

## Compile record

Every packet was compiled with pdfLaTeX and BibTeX before its commit, in the
order `pdflatex` then `bibtex` then `pdflatex` then `pdflatex`, and no packet was
committed until it returned zero errors and zero overfull boxes. The measured
result for each day is recorded in that day's `packet/README.md`.

## Positioning constraints

Nothing in this directory is a submission of record and nothing here is an
agreement. Each day is a proposal to the chief executive and requires his
approval before any part of it leaves the repository. No order in
`investing/` has been placed and none constitutes investment advice; the
instruments named are candidates for a treasury policy the chief executive sets.
UC San Diego, Moores Cancer Center, Scripps Research and Scripps Health are named
as candidate partners at the feasibility stage only, and no agreement of any kind
exists with any of them. No drug supply agreement, letter of authorization, or
regulatory cross-reference is in place with Revolution Medicines. No robotic
configuration has been specified or cleared. No patient has been treated. The
approval of Rasonque is an approval of a drug in the metastatic setting and is
nowhere described as an approval of the perioperative use this program proposes.
The capitalization figures are a plan, not a completed raise: no term sheet,
SAFE, or subscription agreement exists.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).
The paper files are deposited in `/funding/auto-fund`.
