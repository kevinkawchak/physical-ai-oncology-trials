# 29 - Artifact Counts Treemap (NEW)

## Purpose

Add a NEW treemap chart in Section 6 (Conclusions, Headline Artifact
Counts) that visualizes the cumulative artifact set across the four
simulations (392 site-side artifacts from Sim 1, the 10 stage modules from
Sim 2, the 24 hourly scripts plus 53 agents plus 75 ASCII from Sim 3, and
the 168 hourly scripts plus 525 ASCII from Sim 4).

## Source Paper Section

`sections/conclusions.tex` Section 6 (headline artifact counts paragraph).

## Image Properties

- Filename: `images/29_artifact_treemap.png`
- DPI: 300
- Size: 10 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: Sim 1 navy (#1F4E79), Sim 2 teal (#2C7A7A), Sim 3 gold
  (#B45424), Sim 4 deep purple (#6A4C8C). Sub-tile fills lighter shades.

## Layout

- Treemap with four large blocks proportional to total artifact count per
  simulation, divided into sub-tiles for the artifact types.
- Sim 1 block: 392 total artifacts (168 ASCII diagrams plus 224 Markdown
  files).
- Sim 2 block: 12 Python modules plus 30 ASCII progress diagrams plus 6
  deliverable diagrams plus 6 regulatory tables = 54 artifacts.
- Sim 3 block: 24 Python plus 24 JSON plus 75 ASCII plus 53 agent files =
  176 artifacts.
- Sim 4 block: 168 Python plus 168 JSON plus 525 ASCII plus 7 daily
  summaries = 868 artifacts.
- Tile labels per simulation in bold; sub-tile labels in regular weight.
- Header: "Cumulative Artifact Counts Across the Four LLM Simulations
  (Total = 1,490 artifacts)."

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Implementation: use a manual rectangle layout (matplotlib does not have
  a built-in treemap) or compute a squarified treemap layout. Keep sub-
  tile minimum side at least 0.5 inches at 300 DPI to ensure label
  legibility.

## Suggested Caption

Figure 29: Cumulative artifact counts treemap across the four author
Physical AI oncology trial simulations totaling approximately 1,490 ASCII,
Markdown, Python, and JSON artifacts.
