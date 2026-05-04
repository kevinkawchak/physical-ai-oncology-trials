# 13 - Code-Based vs Text-Only Simulations Comparison Chart

## Purpose

Replace the code-vs-text comparison ASCII block (Methods 2.7 and the
README) with a side-by-side practical comparison chart that shows the six
trade-off dimensions and which simulation falls into each category.

## Source Paper Section

`sections/methods.tex` lines 192 to 224 (code-vs-text simulations) and the
parallel block in `full-paper/README.md`.

## Image Properties

- Filename: `images/13_code_text_comparison.png`
- DPI: 300
- Size: 10 inches wide by 5.5 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: text-only orange (#B45424), code-based green (#2C7A4D), header
  navy (#1F4E79).

## Layout

- Top header: "Text-Only (Sim 1) vs Code-Based (Sims 2, 3, 4) Practical
  Trade-Offs."
- Six rows, one per dimension. Left text-only column in orange tint, right
  code-based column in green tint. Center cell is the dimension label.
- Right edge: a small "Optimal pairing" callout that reads "Sim 1 plus Sim
  4 paired together is the closest existing approximation."
- Bottom note: "A future Claude Code or competing local-AI instance that
  combines cloud (1M token context) plus local (specialist agents) would
  offer both the power of cloud compute and the security and flexibility
  of local computing."

## Dimension Data

| Dimension          | Text-Only (Sim 1)         | Code-Based (Sim 2, 3, 4)        |
| ------------------ | ------------------------- | ------------------------------- |
| Cloud compute use  | Light (text only)         | Moderate to heavy               |
| Local compute use  | None                      | Yes (Sim 2 light, Sim 4 4GB)    |
| Downstream agents  | Cannot consume Python     | Can fan out to local agents     |
| Auditability       | Markdown plus ASCII       | Markdown plus ASCII plus .py    |
| Re-run determinism | Re-run varies (LLM)       | Identical (.py is fixed)        |
| Verification cost  | Minimal                   | i5-6200U / 4 GB tested          |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.

## Suggested Caption

Figure 13: Code-based versus text-only simulation trade-offs for downstream
automation and clinical-reader auditability.
