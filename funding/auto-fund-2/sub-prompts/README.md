# sub-prompts - the five-business-day schedule (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Sub-prompts](https://img.shields.io/badge/Sub--prompts-5-00417A.svg)](#the-five-sub-prompts)
[![Commits per day](https://img.shields.io/badge/Commits%20per%20day-10%2B-3C7DB2.svg)](#the-commit-order-inside-one-day)
[![Publication dirs](https://img.shields.io/badge/Publication%20dirs-none-6C757D.svg)](#why-there-are-no-publication-directories)
[![Invariants](https://img.shields.io/badge/Shared%20invariants-11-6C757D.svg)](#the-eleven-invariants-every-day-inherits)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One sub-prompt directory per business day, and no others. The general template,
[`../../capitalization-plan/sub-prompts`](../../capitalization-plan/sub-prompts),
ran eight sequential stages against one paper. This block, like
[`../../auto-fund/sub-prompts`](../../auto-fund/sub-prompts) before it, produces
five independent daily deliverables, so the schedule is five parallel days, and
each day carries its own draft, full and final discipline internally.

## Why there are no publication directories

The template closed with `final-capital/publication/`, which held the deposited
PDF and source zip of a work with a digital object identifier. This block has no
identifier and asks for none. What it produces is correspondence and
instructions, and correspondence is deposited where it is used, in the day
directory. Each day therefore carries its own compiled `main.pdf` and its own
Overleaf zip inside `packet/`, and no `publication/` directory exists anywhere
under `../`.

## The five sub-prompts

| # | Sub-prompt | Day directory | Theme | Signal spent |
|:--|:--|:--|:--|:--|
| 1 | [`day-1-28Sep26/`](day-1-28Sep26) | [`../28Sep26`](../28Sep26) | The unsolicited applicants | Two LinkedIn inquiries, no posting |
| 2 | [`day-2-29Sep26/`](day-2-29Sep26) | [`../29Sep26`](../29Sep26) | The external standard request | The SEC San Francisco office's "very seriously" |
| 3 | [`day-3-30Sep26/`](day-3-30Sep26) | [`../30Sep26`](../30Sep26) | The structured team | The White House communications after September 12 |
| 4 | [`day-4-01Oct26/`](day-4-01Oct26) | [`../01Oct26`](../01Oct26) | The pilot inquiry | The NSF multi-million pilot inquiry |
| 5 | [`day-5-02Oct26/`](day-5-02Oct26) | [`../02Oct26`](../02Oct26) | The week's record | None alone; the earned follow-ups and the cadence |

## The eleven invariants every day inherits

Each sub-prompt restates these with its own values, and none may relax one.

| # | Invariant | Value |
|:--|:--|:--|
| 1 | Correspondence format | `.txt`, with `FROM`, `TO`, `CC`, `SUBJECT`, an introduction, a body, a closing, an attachment manifest, and a pre-send checklist |
| 2 | Address count | At least three verified addresses per email across `TO` and `CC`, each in [`../inputs`](../inputs) |
| 3 | Paste geometry | One paragraph per physical line in every letter body, so GitHub shows it unwrapped and iOS Mail wraps it to the screen |
| 4 | Technical reader format | `.md`, lightly formatted, tables allowed, no LaTeX furniture |
| 5 | Compiled format | `.tex` under `packet/`, zipped for Overleaf, with `main.pdf` beside it |
| 6 | Caption geometry | Two lines, balanced to a small character spread, at the body measure, centered |
| 7 | Caption spacing | `\vspace{-0.60cm}` between every float and its caption, giving 7.44 pt from rule to first caption line |
| 8 | Table geometry | `\begin{tabularx}{\textwidth}`, every fixed column `>{\raggedright\arraybackslash}p{...}`, widths cut to the longest real cell |
| 9 | Money and trial numbers | Reconcile to [`../inputs`](../inputs); nothing is re-derived |
| 10 | Dates | The directory name carries the date; letter bodies do not, so the chief executive may act on a day outside its own date |
| 11 | Rasters | None. Every figure is TikZ compiled from source in this repository |

## The commit order inside one day

A day is complete when all of the following are committed and pushed, each on
its own commit unless stated, and no commit is held back.

| Order | Commit | Rule |
|:--|:--|:--|
| 1 | The sub-prompt `README.md` | Instruction G |
| 2 | The day `README.md`, the approval sheet | Rule 5 |
| 3 | `emails/README.md` | Rule 5 |
| 4 onward | One commit per `.txt` letter or LinkedIn reply | Instruction A |
| then | `briefs/`, `forms/`, `investing/`, `diagrams/`, one commit each | Instructions B and K |
| then | `packet/main.tex`, `packet/fundstyle.sty`, `packet/references.bib`, `packet/README.md`, one commit each | Rule 6 |
| then | `sec-00` through `sec-06`, one commit each | Rule 6 |
| then | `packet/sections/README.md` | Rule 5 |
| last | The compiled `main.pdf` and the Overleaf zip | Instruction L |

That is more than twenty commits per day against a floor of ten.

## Where the two closing commits sit

The master prompt reserves the last two commits of the whole build. They come
after day 5.

| Commit | Scope |
|:--|:--|
| Second to last | Every error in every file across all five days: compile defects, dialect, punctuation, symbols, links, caption balance, table width, page shape, address count, and paste geometry |
| Last | Root `README.md`, `CHANGELOG.md`, `releases.md`, `funding/README.md`, and `prompts/output-auto-fund-2.md` |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `sub-prompts/README.md` | [`../../capitalization-plan`](../../capitalization-plan) | The schedule table and the rule that a stage states what it may not relax |
| `sub-prompts/README.md` | [`../../auto-fund`](../../auto-fund) | The five-parallel-days shape and the per-day commit order |
| [`../prompts/prompt-auto-fund-2.md`](../prompts/prompt-auto-fund-2.md) | This block | Invariants 2 and 3, and the commit order |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
