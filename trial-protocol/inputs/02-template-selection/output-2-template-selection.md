## output-2-template-selection

# Template selection for H. R. 9510 v5.0, the Financial Data Amendment

*Independent research draft. Not legal advice. Media are tables, ASCII, and
gray-scale Mermaid only; no images.*

Stage 2 chooses one LaTeX bill file set to base the v5.0 measure on. The two
candidates are the two finished ancestors of this bill line.

## The two candidates, inspected

| | (a) `cancer-automated/.../VVUQ-05/final-bill` (Bill v3.0) | (b) `auto-bill-01/final-bill` (Bill v4.0) |
|:--|:--|:--|
| Style file | `usctitle.sty` v3.1.0 | `usctitle.sty` v4.0.0 |
| Visual primitives | `asciifig`/`asciifigsmall` (centered, white-background, black-ruled monospace frames) + full-width `xltabular` tables | `mermaidfig` (gray-scale TikZ in the same centered frame) + `treelisting` + the same tables |
| Bill body | SECTION 1 + SEC. 2-4 + five non-operative appendices (A-E), eight section files | SECTION 1 + SEC. 2-4 + twelve deliverable appendices (A-L), fifteen section files |
| Carries the operative 515D text | Yes, with the comparative print | Yes, with the comparative print |
| Inspected files | `main.tex`, `usctitle.sty`, `references.bib`, `sections/s2-findings.tex`, `s3-amendment.tex`, `s4-comparative.tex`, `a5-a9` | `main.tex`, `usctitle.sty`, `sections/s2-findings.tex`, `g-paygo-cost.tex` |

## Decision: candidate (a), the VVUQ-05 final-bill (Bill v3.0)

**The Bill v3.0 file set is selected as the base**, for three reasons that
follow directly from the Master prompt:

1. **The Master prompt commands it.** "Build from
   `cancer-automated/tree/main/papers/VVUQ-05/final-bill` (no /deliverables)" -
   candidate (a) *is* that file set, minus its deliverables package. Its five
   lettered appendices (not the twelve deliverables) are the structural mold
   the five v5.0 financial appendices are poured into.
2. **It already carries two of the three required media natively.** Objective
   (b) requires tables, ASCII, and Mermaid together. The v3.0 style is the
   proven home of the ASCII figure frame and the full-width ragged-right table
   apparatus; only the Mermaid primitive must be imported.
3. **Its appendix skeleton matches the financial perspective.** The v3.0
   pattern - evidence appendix, package appendix, standard appendix, influence
   matrix, transparency appendix - maps one-to-one onto the v5.0 financial
   pattern (cost estimate, verification economics, financial-data standard,
   financial influence matrix, transparency), so the financial content drops
   into a proven page architecture.

## How the unselected candidate is still used

Candidate (b) is not discarded; it contributes exactly the pieces the base
lacks. The v5.0 `usctitle.sty` (v5.0.0) **merges both styles**:

| Primitive | Taken from | v5.0 role |
|:--|:--|:--|
| `asciifig`, `asciifigsmall` | v3.0 (a) | Ledgers, money flows, waterfalls, layering maps, timelines |
| `L{}`, `Y`, `ragcol` table columns; `xltabular` | v3.0 (a) | All full-width tables |
| `mermaidfig` + the `mm*` TikZ node styles + the five-step gray palette | v4.0 (b) | Process, sequence, architecture, and build diagrams |
| `\billbox` boxed statement | v4.0 (b) | The SEC. 5(e) budgetary-effects clause, set off |
| Amendment apparatus (`\billcaption` ... `\billtocline`), USC reproduction commands | identical in both | Carried once |
| **New for v5.0:** `R{<width>}` right-aligned numeric column | added here | Fiscal-year dollar columns that must align on the right |

## The medium genre each bill part leads with

| Bill part | Leads with | Why |
|:--|:--|:--|
| Cover (caption page) | ASCII overview | The v3.0 cover convention; a fixed-width process-and-money map |
| SEC. 2 Findings | Mermaid lineage + table + ASCII ledger | Topology (lineage), enumeration (four works), alignment (cost ledger) |
| SEC. 3 Amendment | Table (gate schedule with unit costs) + Mermaid BPMN + ASCII layering | Schedule is enumerable; decision rule is a process; layering is fixed-width |
| SEC. 4 Comparative print | Table (change map) + ASCII section order | Enumeration and fixed-width order |
| SEC. 5 Financial core | Tables (data dictionary, authorization) + Mermaid sequence + ASCII money flow | Money series enumerate; reporting is a sequence; flows align |
| App. A Cost estimate | Tables (BA/outlays, user fees, scorecard) + ASCII waterfall | CBO-style tables; the BA-to-outlay ramp is a waterfall |
| App. B Verification economics | Table (unit costs) + Mermaid component | Unit costs enumerate; the cost pipeline is a component flow |
| App. C Financial-data standard | Table (standard map) + Mermaid C4 | The standard enumerates; the system of record is an architecture |
| App. D Research influence matrix | Table only | A matrix is a table; no diagram belongs here |
| App. E Transparency | Tables (lineage, milestones) + Mermaid build flowchart | Schedules enumerate; the build is a process |

## How the choice flows into later stages

- Stage 3 selects the concrete gray-scale Mermaid diagrams (from
  `Clinical-AI-Demos/.../ai-outputs/output-01`) and the concrete ASCII forms
  (from `cancer-automated/.../VVUQ-05/update-bill/figures-bill` and the v3.0
  sections) that fill the genre slots above.
- Stage 4 fixes the numbered figure and table plan against these genres.
- Stages 5-7 build `main.tex`, the merged `usctitle.sty`, `references.bib`
  (the v3.0 set plus the financial authorities from Stage 1), and the nine
  section files on this base.
