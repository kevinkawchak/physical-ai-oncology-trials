# 22 - Multimodal Inputs Diagram (NEW)

## Purpose

Add a NEW multimodal inputs diagram in Section 2.6 (Methods, Code Snippet)
and Section 1.2 (Introduction, AI Baseline) that visualizes the six input
modalities the 1M token context can ingest in a single inference pass.

## Source Paper Section

`sections/methods.tex` Section 2.6 (snippet showing Python plus JSON
output) and `sections/introduction.tex` Section 1.2 (AI baseline note that
narrow models cannot ingest the full set).

## Image Properties

- Filename: `images/22_multimodal_inputs_diagram.png`
- DPI: 300
- Size: 9.5 inches wide by 6 inches tall (half-page landscape)
- Background: white (#FFFFFF)
- Palette: six input nodes in distinct hues (navy, teal, green, gold,
  purple, red wine), one central convergence node in dark slate, light
  arrow lines (#9CA3AF) from each input to the center.

## Layout

- Six labeled input nodes on the left (3 over 3 grid):
  1. Protocol text (LaTeX, Markdown).
  2. EHR records (JSON, FHIR).
  3. Robot telemetry (force feedback, gating efficiency).
  4. Regulatory adaptations (21 CFR 312, 21 CFR 50, ICH E6(R3)).
  5. ASCII facility diagrams.
  6. Python agent code (53 core agents).
- Central node on the right: "1M Token Repository Context (Claude Code
  Opus 4.7 Max)."
- Arrows from each input to the central node, with edge labels for the
  approximate token count contributed by each modality.
- Below the central node: an output strip listing the four artifact types
  that emerge from the context (per-hour Markdown narrative, ASCII
  diagrams, Python agent scripts, JSON sponsor-decision logs).
- Header: "Six Input Modalities Converge on 1M Token Context, Producing
  Four Output Artifacts per Inference Pass."

## Token Contribution Estimates

- Protocol text: 50,000 tokens.
- EHR records: 200,000 tokens.
- Robot telemetry: 80,000 tokens.
- Regulatory adaptations: 120,000 tokens.
- ASCII diagrams: 30,000 tokens.
- Python agent code: 150,000 tokens.
- Reserved for prompt and output: 370,000 tokens.

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- All arrows clearly visible against the white background.

## Suggested Caption

Figure 22: Six input modalities the 1M token context ingests in a single
inference pass, producing four output artifacts per pass.
