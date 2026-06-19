## output-4-figure-selection

# Figure and table plan for H. R. 9510 v5.0

*Independent research draft. Figures are gray-scale Mermaid (Markdown)
reproduced as gray-scale TikZ (LaTeX) or monospace ASCII in centered
white-background frames; tables are full-width, left-aligned, ragged-right,
with right-aligned `R` columns for dollar series. No images.*

Stage 4 fixes the correct visual type for every slot in the bill, applying the
three-media rule from the figures-bill taxonomy and Stage 3: a **table** where
the content is enumerable financial data; **ASCII** where fixed-width
alignment carries the meaning; **Mermaid** where topology carries the meaning.
The result is deliberately balanced - six ASCII figures (counting the cover),
six Mermaid figures, fifteen tables - so the finished bill demonstrably
contains all three media (Master prompt objective (b)).

## Figure inventory (eleven numbered + the cover)

| Fig. | Title | Medium | Bill location | Descends from |
|:--|:--|:--|:--|:--|
| Cover | The v5.0 financial build and money frame | ASCII | `main.tex` | v3.0 cover; "building process" diagrams (figures-bill 01) |
| 1 | Evidence-to-law-to-cost lineage | Mermaid flowchart LR | SEC. 2 | lineage boxes (figures-bill 01, 04, 05) |
| 2 | Conventional vs autonomous cost ledger | ASCII side-by-side | SEC. 2 | "accelerated timeline" (figures-bill 01, 02), re-priced |
| 3 | Gate decision rule with cost-record checkpoint | Mermaid BPMN | SEC. 3 | gate funnel / decision surface (figures-bill 02, 03) |
| 4 | Statutory layering with the financial sections | ASCII layering map | SEC. 3 | "threads through Title 21" (figures-bill 05), + § 379j layer |
| 5 | Twelve sections in comparative-print order | ASCII ordered map | SEC. 4 | section-order chain (figures-bill 05), + s379j row |
| 6 | Annual financial-data reporting flow | Mermaid sequence | SEC. 5 | handoff/sequence family (Stage 3) |
| 7 | Appropriations and user fees to activities | ASCII money flow | SEC. 5 | repository/flow trees (figures-bill 01), re-themed to money |
| 8 | Budget authority to outlays waterfall | ASCII waterfall | App. A | timeline family (figures-bill 02), re-themed to fiscal years |
| 9 | Unit economics of a ten-gate run | Mermaid clustered component | App. B | clustered flow (Stage 3, Graphviz/DOT family) |
| 10 | The financial-data system of record | Mermaid C4 view | App. C | architecture view (Stage 3, Structurizr/C4 family) |
| 11 | The nine-milestone build process | Mermaid flowchart TB | App. E | "building process within the work" (figures-bill 03, 04) |

Media balance: ASCII 6 (Cover, 2, 4, 5, 7, 8); Mermaid 6 (1, 3, 6, 9, 10, 11);
no family repeats among the Mermaid six.

## Table inventory (fifteen, full-width)

| Tbl. | Title | Bill location | Financial data it enumerates |
|:--|:--|:--|:--|
| 1 | The four works at a glance, with the financial footprint | SEC. 2 | Role, releases, assurance result, illustrative build cost, DOI per work |
| 2 | Ten-gate threshold schedule with unit costs | SEC. 3 | Agreement, CV, hard predicate, governing standards, illustrative cost per gate run |
| 3 | Ten conforming amendments crosswalk | SEC. 3 | FD&C §, 21 U.S.C. §, change made |
| 4 | Comparative-print change map (twelve sections) | SEC. 4 | File, citations, change, including the § 379j fee row |
| 5 | Financial-data elements dictionary | SEC. 5 | Element, who files, where it flows, public visibility |
| 6 | Authorization of appropriations schedule | SEC. 5 | FY2027-FY2031 authorized amounts by activity (illustrative) |
| 7 | Estimated budget authority and outlays | App. A | BA, estimated outlays, spend-out by fiscal year (illustrative) |
| 8 | Statutory device user-fee structure | App. A | Fee class as percent of the standard PMA fee (21 U.S.C. § 379j) |
| 9 | Scorecard and mandates summary | App. A | PAYGO, CutGo, UMRA, earmark line items |
| 10 | Per-gate verification unit-cost schedule | App. B | Illustrative compute, records, and review cost per gate |
| 11 | Acceleration economics summary | App. B | Conventional vs autonomous program-year costs (illustrative) |
| 12 | The financial-data transparency standard | App. C | Four tests mapped to Green Book, GAO-05-734SP, A-11, part 11 |
| 13 | Research influence matrix (financial) | App. D | Authority, what it is, what it informs, placement |
| 14 | Bill version lineage | App. E | v1.0-v5.0 directories, roles, DOIs |
| 15 | The nine-milestone commit schedule | App. E | Milestone, files or action, role |

## Placement rules carried into the bill stages

1. Each operative section (SEC. 2-5) carries one to three figures and its
   tables; each appendix carries at most one figure, so visuals complement and
   never crowd the text.
2. Every figure - ASCII or Mermaid-as-TikZ - is centered on a white background
   inside a thin black frame, sized to its content, with a small italic
   caption; wide ASCII uses the smaller `asciifigsmall` frame so no line runs
   off the right margin.
3. Every table is set to `\textwidth` with
   `>{\raggedright\arraybackslash}p{...}` columns; **fiscal-year dollar
   columns use the right-aligned `R{...}` column** so amounts align on the
   right; every dollar figure is marked illustrative unless tied to a cited
   statute or notice; every fiscal-year series must sum to its stated total.
4. Figure and table numbering is continuous across the bill; each entry is
   reachable from the clickable SECTION 1(b) table of contents through its
   section.
5. Repository structure appears as a tree inside an ASCII frame (the
   figures-bill preference) rather than as file-name lists in prose.
