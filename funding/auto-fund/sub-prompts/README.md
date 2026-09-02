# sub-prompts - the five-business-day schedule (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Sub-prompts](https://img.shields.io/badge/Sub--prompts-5-00417A.svg)](#the-five-sub-prompts)
[![Commits per day](https://img.shields.io/badge/Commits%20per%20day-10%2B-3C7DB2.svg)](#the-commit-order-inside-one-day)
[![Publication dirs](https://img.shields.io/badge/Publication%20dirs-none-6C757D.svg)](#why-there-are-no-publication-directories)
[![Invariants](https://img.shields.io/badge/Shared%20invariants-9-6C757D.svg)](#the-nine-invariants-every-day-inherits)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One sub-prompt directory per business day, and no others. The parent build at
[`../../capitalization-plan/sub-prompts`](../../capitalization-plan/sub-prompts)
ran eight stages against one paper: five diagram-platform stages, then draft,
full, and final. This build has a different shape. It produces five independent
daily deliverables rather than three successive drafts of one document, so the
schedule is five parallel days rather than eight sequential stages, and each day
carries its own draft, full and final discipline internally.

## Why there are no publication directories

The parent build closed with `final-capital/publication/`, which held the
deposited PDF and the source zip for a work with a digital object identifier.
This build has no digital object identifier and asks for none. What it produces
is correspondence and instructions, and correspondence is deposited where it is
used, in the day directory. Each day therefore carries its own compiled
`main.pdf` and its own Overleaf zip inside `packet/`, and no `publication/`
directory exists anywhere under `../`.

## The five sub-prompts

| # | Sub-prompt | Day directory | Theme | Counterparties reachable |
|:--|:--|:--|:--|:--|
| 1 | [`day-1-02Sep26/`](day-1-02Sep26) | [`../02Sep26`](../02Sep26) | Approval to ask | Federal program staff, brokerage, registration portals |
| 2 | [`day-2-03Sep26/`](day-2-03Sep26) | [`../03Sep26`](../03Sep26) | Private capital bridge | Angels, family offices, the drug developer, counsel |
| 3 | [`day-3-04Sep26/`](day-3-04Sep26) | [`../04Sep26`](../04Sep26) | Site and partner | Clinical sites, disease foundations, research institutes |
| 4 | [`day-4-07Sep26/`](day-4-07Sep26) | [`../07Sep26`](../07Sep26) | Staged queue | None: federal holiday, markets closed |
| 5 | [`day-5-08Sep26/`](day-5-08Sep26) | [`../08Sep26`](../08Sep26) | Execution and cadence | All of the above, plus the queue day 4 built |

## The nine invariants every day inherits

Each of the five sub-prompts restates these in its own README with the values
specific to that day, and none of the five is permitted to relax one.

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Correspondence format | `.txt`, with `FROM`, `TO`, `CC`, `SUBJECT`, an introduction, a body, a closing, an attachment manifest, and a pre-send checklist |
| 2 | Technical reader format | `.md`, lightly formatted, tables allowed, no LaTeX furniture |
| 3 | Compiled format | `.tex` under `packet/`, zipped for Overleaf, with `main.pdf` beside it |
| 4 | Caption geometry | Two lines, balanced to a small character spread, at the body measure, centered |
| 5 | Caption spacing | `\vspace{-0.60cm}` between the float and its caption, giving 7.44 pt from rule to first caption line, identically for every figure and every table |
| 6 | Table geometry | `\begin{tabularx}{\textwidth}`, every fixed column declared `>{\raggedright\arraybackslash}p{...}`, widths cut to the longest real cell |
| 7 | Money | Reconciles to the frame in [`../inputs`](../inputs); nothing is re-derived |
| 8 | Dates | The directory name carries the date; the body does not repeat it, because the chief executive may act on a day outside its own date |
| 9 | Rasters | None. Every figure is TikZ compiled from source in this repository |

## The commit order inside one day

Rule 6 of the master prompt fixes part of this order and the rest follows from
it. A day is complete when all of the following have been committed and pushed,
each on its own commit unless stated:

| Order | Commit | Rule |
|:--|:--|:--|
| 1 | The day `README.md`, the approval sheet | Rule 5 |
| 2 | `emails/README.md` | Rule 5 |
| 3 to 7 | One commit per `.txt` email | Instruction A |
| 8 | `briefs/README.md` and the day's `.md` briefs | Instruction B |
| 9 | `forms/README.md` and the day's `.md` form packs | Instruction A and K |
| 10 | `investing/README.md` and the day's `.md` capital instruction | The prompt's opening paragraph |
| 11 | `diagrams/README.md` and the three figure specifications | Rule 3 |
| 12 | `packet/main.tex` | Rule 6 |
| 13 | `packet/fundstyle.sty` | Rule 6 |
| 14 | `packet/references.bib` | Rule 6 |
| 15 | `packet/README.md` | Rule 6 |
| 16 to 22 | One commit per section `.tex`, `sec-00` through `sec-06` | Rule 6 |
| 23 | `packet/sections/README.md` | Rule 5 |
| 24 | The compiled `main.pdf` and the Overleaf zip | Instruction L |

That is 24 commits at a minimum against a floor of 10, and no commit is held
back: each is pushed the moment the file it carries is written, so the branch can
be watched while the build runs.

## Where the two closing commits sit

The master prompt reserves the last two commits of the whole build, not of each
day. They come after day 5 and are:

| Commit | Scope |
|:--|:--|
| Second to last | Every error in every file, across all five days: compile defects, dialect, punctuation, symbol, link, caption balance, table width, and page-shape corrections |
| Last | Root `README.md`, `CHANGELOG.md`, `releases.md`, and `prompts/output-auto-fund.md` |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `sub-prompts/README.md` | [`../../capitalization-plan`](../../capitalization-plan) | The schedule table, the per-stage README convention, and the rule that a stage states what it may not relax |
| `sub-prompts/README.md` | [`../../move-in`](../../move-in) | The practice of listing invariants as a numbered table each stage restates |
| Rule 6 of [`../prompts/prompt-auto-fund.md`](../prompts/prompt-auto-fund.md) | This build | The commit order above, items 12 through 22 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
