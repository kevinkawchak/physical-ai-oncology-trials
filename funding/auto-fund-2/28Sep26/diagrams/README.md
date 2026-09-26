# 28Sep26 / diagrams - three figure specifications (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-2E5E4E.svg)](..)
[![Figures](https://img.shields.io/badge/Figures-1%20to%203-2E5E4E.svg)](.)
[![Platforms](https://img.shields.io/badge/Platforms-Mermaid%2C%20Diagrams%2C%20Graphviz-6C757D.svg)](#the-three-figures)
[![Spacing](https://img.shields.io/badge/Caption%20spacing--0.60cm-6C757D.svg)](#the-caption-rule)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

One specification per figure. Each gives the figure's native source on its own
platform, the TikZ construction that reproduces it in the packet, the exact
caption, and the section that uses it. A reader who needs to correct a figure
edits the section file and checks it against the specification here.

## The three figures

| # | File | Platform | Native construct | Used in |
|:--|:--|:--|:--|:--|
| 1 | [`fig-01-applicant-flow.md`](fig-01-applicant-flow.md) | Mermaid | Flowchart with two decisions and two stops | `sec-01-the-signal.tex` |
| 2 | [`fig-02-roster-by-location.md`](fig-02-roster-by-location.md) | Diagrams | Clustered glyph tiles | `sec-03-the-roster.tex` |
| 3 | [`fig-03-funding-route-records.md`](fig-03-funding-route-records.md) | Graphviz | Record nodes under one root | `sec-05-funding-routes.tex` |

PlantUML and D2 are not used on day 1; they appear on days 2, 3 and 4. No
platform is used twice in one day, and across the five days each platform is
used exactly three times.

## The caption rule

Every caption is two lines at the body measure, centered, with the two lines
within a few characters of each other, and every caption sits `-0.60cm` below
its frame, which leaves exactly 7.44 pt from the frame's last rule to the first
caption line.

| Figure | Line 1 | Line 2 | Spread |
|:--|:--|:--|:--|
| 1 | 82 characters | 78 characters | 4 |
| 2 | 83 characters | 82 characters | 1 |
| 3 | 81 characters | 83 characters | 2 |

## Rules every figure here obeys

| Rule | How it is met |
|:--|:--|
| No stick figures | People are never drawn; roles are glyph tiles or labeled boxes |
| No rasters | Every figure is TikZ in the section file |
| No broken words | The style switches hyphenation off inside every node |
| No decimal node names | Loops over decimal coordinates name no nodes |
| Readable source | One node per line, positions stated as plain coordinates |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | The roles, FTE and license requirements in Figure 2 |
| `08Sep26/diagrams/README.md` | [`../../../auto-fund`](../../../auto-fund) | The specification layout and the rules table |
| `../packet/fundstyle.sty` | [`../packet`](../packet) | The `mm*`, `dg*` and `gv*` vocabularies the three figures use |
| `../investing/capital-01-contingent-payroll-earmark.md` | [`../investing`](../investing) | The $40,000 ceiling in Figure 3 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
