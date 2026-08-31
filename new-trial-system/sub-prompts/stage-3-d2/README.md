## Stage 3 sub-prompt - d2-type figures

[![Stage](https://img.shields.io/badge/Stage-3%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-3-d2)
[![Platform](https://img.shields.io/badge/Platform-D2-A32A3C.svg)](https://d2lang.com)
[![Figures](https://img.shields.io/badge/Figures-6-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/d2)
[![Output](https://img.shields.io/badge/Output-new--trial--system%2Fd2-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/d2)

### Instruction

Produce six d2-type figure specifications in
[new-trial-system/d2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/d2),
one file per figure, one commit per file, committed and pushed the moment each
file is written.

D2 is chosen wherever the paper's claim is about **nesting or tabulation**: a
thing inside a thing, or a value that only means something when read against
the row and column it sits in. Six of the paper's twenty-five figures are
d2-type because six of its claims are grids or containers rather than
sequences. Where a mermaid flowchart would force a comparison into a chain of
arrows, a d2 grid states it as a cell.

| Figure | Section | Construct | Perspective no other figure takes |
|:--|:--|:--|:--|
| 2 | §1 Introduction | grid | Prior system against new system on ten operating axes, each cell a measured value rather than an adjective |
| 8 | §3 IND | sql tables | Every Form FDA 1571 content item mapped to the generated section file and the repository path that satisfies it |
| 12 | §4 Trial Protocol | containers | What the Phase 2 document inherits unchanged from the Phase 1 document, what it replaces, and what it adds |
| 16 | §5 Legislation | layers | One verification requirement traced down four layers, from statute to a site standard operating procedure |
| 18 | §6 Funding Proposals | grid | The money grid: three asks, four cost layers, and the two overhead regimes the same direct work is priced under |
| 22 | §7 AI Peer Review | grid | Prior and new peer review on six economic axes, with the ratio in each cell |

### Required contents of each file

1. An H1 naming the figure number and its one-line perspective.
2. A **Type**, **Section**, **Perspective** paragraph stating what no other
   figure in the paper shows.
3. A caption block of exactly two lines within a four-character spread, opening
   with `Figure N. ` exactly as printed.
4. Valid D2 source in a fenced `d2` block using containers, `grid-rows`,
   `grid-columns`, `shape: sql_table`, or `layers` as the figure requires.
5. A TikZ construction table using the `d2*` vocabulary: `d2cont`, `d2box`,
   `d2key`, `d2mid`, `d2soft`, `d2gray`, `d2cell`, `d2cellh`, `d2sql`,
   `d2step`, with absolute coordinates and an explicit cell pitch.
6. A cell-value table listing every value the grid carries and its source, so
   no number in a figure is unattributable.
7. A repository-sources list naming exact files.

### Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white `#FFFFFF`.
Header cells take the Burgundy fill with white text. Charcoal is a stroke and a
text color only. **No black fill.**

### Anti-defect requirements

- **Grid regularity.** Every cell in a grid is the same height, and column
  widths are stated once in the construction table. A grid whose cells are
  sized by their content is a defect.
- **Container nesting depth.** No container may nest more than two deep. A
  third level is a sign the figure is really two figures.
- **Over-density.** No grid may exceed 6 columns by 11 rows including headers,
  and no container figure more than 18 leaf nodes.
- **Syntax hallucination.** No Mermaid or PlantUML keyword may appear in a d2
  fence. Labels containing a colon, a brace, or a pipe are quoted.
- **Edge overlap.** In a grid, edges are forbidden entirely; the grid's meaning
  is carried by position. In a container figure, an edge that must leave its
  container does so through a stated waypoint on the container boundary.
- **Layout instability.** Absolute coordinates only. A cell added in a later
  stage extends the grid downward or rightward and moves no existing cell.

### Prohibitions

Do not copy the money grid from
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan)
or the recipient strip from
[funding/pdac-funding-applications](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications).
Figure 18 carries three asks against four cost layers under two overhead
regimes, which is a different grid from either, and its cells are ratios where
theirs are totals.
