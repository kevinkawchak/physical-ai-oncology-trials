# 11 - AI Baseline Comparison Chart (Full Page)

## Purpose

Replace the `tab:disc-baseline-vs-sim` table in Section 4.2 (Discussion, AI
Baseline) with a full-page four-column comparison chart that shows the
computational signature of supervised baselines, multimodal foundation
models, the Huang 2025 Cox vs ML null result, and the four LLM simulations.

## Source Paper Section

`sections/discussion.tex` lines 170 to 190 (the disc-baseline-vs-sim
table).

## Image Properties

- Filename: `images/11_ai_baseline_comparison.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: supervised models gray (#7C7C7C), multimodal foundation slate
  (#4A6A8A), Huang 2025 amber (#B45424), four simulations green (#2C7A4D),
  header navy (#1F4E79).

## Layout

- Top header: "Computational Signature Comparison: Supervised Baselines vs
  Foundation Models vs Huang 2025 Null Result vs Four LLM Simulations."
- Six-row property matrix: each row is a property name on the left and four
  cells on the right (one per column class). Property rows: Inputs, Output,
  Cadence, TRIPOD+AI compliant, Modalities, Real-time stream.
- Bottom band: a takeaway summary that reads "Computational signature
  comparison only. Not benchmarked AUROC. Repository scale plus narrative
  plus code is the qualitative shift the four simulations introduce."

## Property Data

| Property             | Supervised (Manz, SHIELD-RT, SCORPIO) | Multimodal (PROGPATH, AIM-LCpro)| Huang 2025 Cox vs ML | Four Simulations Here          |
| -------------------- | -------------------------------------- | ------------------------------- | -------------------- | ------------------------------ |
| Inputs               | Fixed feature set per task             | Multimodal images plus tabular  | Tabular features only| Repository plus narrative plus code |
| Output               | One probability per patient            | C-index per cohort              | Hazard ratio         | Per-hour narrative plus diagrams plus JSON |
| Cadence              | Per visit (months)                     | Per cohort                      | Retrospective        | Per hour or per stage          |
| TRIPOD+AI compliant  | Some                                   | Some                            | Yes (meta-analytic)  | No (synthetic)                 |
| Modalities           | 1                                      | 2-4                             | 1 (tabular)          | Text plus code plus ASCII plus JSON|
| Real-time stream     | No                                     | No                              | No                   | Yes (hourly commit)            |

## Style Rules

- Single dashes only, no em dashes or double dashes.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 11: Computational signature comparison versus Manz, SHIELD-RT,
SCORPIO, PROGPATH, AIM-LCpro, Huang 2025 across six properties.
