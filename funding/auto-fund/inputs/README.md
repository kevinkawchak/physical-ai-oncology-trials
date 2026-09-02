# inputs - the repository sources every business day reads (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Sources](https://img.shields.io/badge/Source%20directories-9-00417A.svg)](#the-nine-source-directories)
[![Copies](https://img.shields.io/badge/Copied%20files-none-3C7DB2.svg)](#why-nothing-is-copied-here)
[![Money frame](https://img.shields.io/badge/Frame-%24700K%20%C3%97%205%20years-6C757D.svg)](#the-money-frame-no-day-re-derives)
[![Evidence](https://img.shields.io/badge/Checkable%20quantities-6-6C757D.svg)](#the-six-checkable-quantities)
[![Contacts](https://img.shields.io/badge/Named%20addresses-40%2B-6C757D.svg)](#the-named-addresses-and-where-they-come-from)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

This directory is an index, not an archive. It names every repository source the
five business days read, states what each one supplies, and points at it where it
already lives.

## Why nothing is copied here

The parent build at [`../../move-in/inputs`](../../move-in/inputs) held three
source artifacts because they arrived from outside the repository and had nowhere
else to live. Every source this build reads is already deposited in this
repository under its own directory with its own README. Copying a file that
already has a home creates a second copy that will drift from the first, and the
first is the one the rest of the repository cites. So the sources are indexed and
linked, and the day directories cite the original path in every case.

## The nine source directories

| # | Source | What the five days take from it |
|:--|:--|:--|
| 1 | [`../../capitalization-plan`](../../capitalization-plan) | The whole capital frame: the $306K Phase I, the $1.3M Phase II, the $1.606M route, the $2.104M delta, the $5.9M private raise, and the 3.67 to 1 leverage. `capstyle.sty` supplies the five TikZ diagram vocabularies and the figure frame |
| 2 | [`../../pdac-funding-applications`](../../pdac-funding-applications) | The ten application file sets, the nine that were emailed, and the `.txt` email format every day here reuses: `FROM`, `TO`, `CC`, `SUBJECT`, body, attachment manifest, pre-send checklist |
| 3 | [`../../move-in`](../../move-in) | The eleven-role roster at 3.95 award-funded full-time equivalents, the $521,000 personnel line, and the separation of federal from non-federal funds that governs every lobbying and outreach action |
| 4 | [`../../potential-partners`](../../potential-partners) | Every named clinical, contracting, IRB, regulatory and research address used in days 1, 3 and 5, together with the escalation path and the three positioning corrections |
| 5 | [`../../science-golden-age`](../../science-golden-age) | The policy position the federal asks are written against, including the SBIR clause that appears in three separate chapters |
| 6 | [`../../supplementary`](../../supplementary) | The founding documents and the January 2026 competition baseline cited in the capital sections |
| 7 | [`../../../trial-protocol`](../../../trial-protocol), [`../../../trial-ind`](../../../trial-ind), [`../../../trial-phase-2`](../../../trial-phase-2) | The asset register a funder is buying: the Phase 1 protocol, the investigational new drug application, and the Phase 2 protocol |
| 8 | [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | The June 2025 to August 2026 chronology, quoted in the opening section of all five packets |
| 9 | [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | The three-model division of labor cited in each packet's method note |

## The money frame no day re-derives

Every packet reconciles to these figures and none of them is recalculated. Where
a day carries money, it carries one of these numbers or a stated subdivision of
one.

| Quantity | Value | Source |
|:--|:--|:--|
| Program, five years, direct | $3,500,000 | `../../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` |
| Program, per year, direct | $700,000 | Same |
| SBIR Phase I, total cost, 9 months | $306,000 | `../../pdac-funding-applications/applications/app-05-nih-sbir-seed` |
| SBIR Phase II, total cost, 24 months | $1,300,000 | Same |
| SBIR route, total cost, 33 months | $1,606,000 | Sum of the two above |
| Delta the SBIR route does not buy | $2,104,000 | `../../capitalization-plan/final-capital`, §3 |
| Private capital behind the firewall | $5,900,000 | `../../capitalization-plan/final-capital`, §4 |
| Private to federal leverage | 3.67 to 1 | Against the annex's 3 to 1 target |
| Personnel inside the annual direct cost | $521,000 across 3.95 FTE | `../../move-in/final-move-in/sections/sec-14-staffing-and-roles.tex` |
| Virtual trial cost, projected | $36,330 | `../../capitalization-plan/final-capital`, Table 17 |

The virtual trial figure is described as **projected** everywhere it appears in
this directory, and never as estimated.

## The six checkable quantities

Reused without re-derivation from
`../../capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`.
Every one carries the limitation its own authors stated, in the same row as the
result, in every table in this directory that reproduces it.

| Source | Test arm | Comparator | Tier |
|:--|:--|:--|:--|
| RASolute 302, May 2026 | 13.2 months median overall survival | 6.6 months | Trial |
| Ten-arm QSP simulation, 250 ODEs | 12.8 months, hazard ratio 0.25 | 5.4 months | In silico |
| Digital twin, 1000 patients | 12.1 months | Not applicable | Twin |
| Digital twin, progression-free survival | Hazard ratio 0.31 | Not applicable | Twin |
| VVUQ credibility, 55 tests | Score 81.9 | V and V 40 gate | Twin |
| Empirical triplicate, 100,000 records | Grade 3 plus, 8.0 percent | 25.0 percent | In silico |

## The named addresses, and where they come from

No address in this directory was invented. Every one is carried from a repository
file that already records it, and each day's email file names that file in its
own pre-send checklist.

| Address group | Count | Carried from |
|:--|:--|:--|
| UC San Diego clinical, trials office, contracting, IRB, ACTRI, coverage analysis | 18 | `../../potential-partners/UC-San-Diego/priority-steps.md` |
| Scripps Research and Scripps Health | 7 | `../../potential-partners/Scripps/priority-steps.md` |
| Revolution Medicines external research, grants, medical information, business development | 5 | `../../potential-partners/UC-San-Diego/priority-steps.md` §2 |
| FDA combination products, CDER, CDRH | 6 | `../../potential-partners/UC-San-Diego/priority-steps.md` §10 and §11 |
| NIH, Grants.gov, and the federal mechanisms already contacted | 12 | `../../pdac-funding-applications/applications/app-01` .. `app-10` |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `inputs/README.md` | [`../../move-in`](../../move-in) | The convention of a comprehensive inputs README with badges, adapted from an archive to an index |
| `README.md` | [`../../capitalization-plan`](../../capitalization-plan) | The single-arithmetic table, reproduced above as the money frame |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../capitalization-plan`](../../capitalization-plan) | The six checkable quantities and their limitations |
| `UC-San-Diego/priority-steps.md`, `Scripps/priority-steps.md` | [`../../potential-partners`](../../potential-partners) | Every named address in the table above |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
