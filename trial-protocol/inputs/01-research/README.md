# 01-research - Stage 1: automatic financial research (v5.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-1%20of%207-blue.svg)](.)
[![Focus](https://img.shields.io/badge/Focus-Financial%20data-darkgreen.svg)](.)

Stage 1 of Process B. Running
[`../sub-prompts/prompt-1-research.md`](../sub-prompts/prompt-1-research.md)
produced [`output-1-research.md`](output-1-research.md): the financial-data
research that gives H. R. 9510 Bill v5.0 its financial spine, current for the
119th Congress, 2d Session (2026). It covers the scoring framework (CBO,
Statutory PAYGO, House CutGo, earmark rules), the exact budget vocabulary
(budget authority, obligations, outlays, authorization versus appropriation),
the FDA funding structure (Salaries and Expenses appropriations plus the MDUFA
V device user-fee cycle, fiscal years 2023-2027, with the statutory fee
percentages), the financial-data law already attached to clinical
investigations (21 CFR part 54, Open Payments, Medicare reasonable and
necessary), the UMRA thresholds, the financial-data quality scaffolding (GAO
Green Book, GAO budget glossary, OMB Circular A-11), and the pre-introduction
financial documents. It closes with the money-map ASCII figure and the table
mapping each research area to its landing place in the bill.

## Files

| File | Role |
|:--|:--|
| [`output-1-research.md`](output-1-research.md) | The Stage 1 narrative output (tables, ASCII money map, sources) |
| `README.md` | This index |

## Sources used from other directories (Rule 6)

| No. | Asset used | Upstream source |
|:--|:--|:--|
| 01 | The research sub-prompt run here | `auto-bill-02/sub-prompts/prompt-1-research.md` |
| 02 | Prior research method and the seven-step structure | `cancer-automated/.../papers/VVUQ-05/update-bill/next-steps/prompt-1-next-steps.md`, `output-1-next-steps.md` |
| 03 | Carried 2026 process facts (eHopper, HOLC, Rule XII CAS, Energy and Commerce referral) | `single-prompt-bill/auto-bill-01/01-research/output-1-research.md` |
| 04 | PAYGO and fiscal-note conventions referenced for the landing map | `single-prompt-bill/auto-bill-01/final-bill/sections/g-paygo-cost.tex` |
| 05 | The operative measure the research targets | `cancer-automated/.../papers/VVUQ-05/final-bill` (no `/deliverables`) |

## Where this stage flows next

Stage 2 (template selection) chooses the LaTeX base; Stage 4 (figure selection)
turns the landing-map table into the figure and table plan; SEC. 5 and
Appendices A, C, and D of every bill stage implement the law identified here.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
