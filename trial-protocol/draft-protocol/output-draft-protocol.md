## output-draft-protocol

Stage 2 produced the draft (scaffold) of the Phase 1 protocol as a complete,
compiling LaTeX project under `trial-protocol/draft-protocol/`. The draft lays
out every NIH-FDA section in order and, for each content slot, carries a
bracketed `[DRAFTING INSTRUCTION]` (the `\draftinstr` macro) that names the exact
repository file the full stage must process and the figure or table medium each
slot will carry. It is a scaffold by design: the architecture, the section
order, the table-of-contents, and the back matter are final, while the prose,
the rendered TikZ figures, and the filled tables are deferred to the full stage.

The project is built from the recolored paper template and the auto-bill-02
primitives. `protostyle.sty` merges them: the Corporate Blue `#00417A` accent
replaces the template's color; the `mermaidfig` TikZ environment and the `mm*`
node styles are recolored to the protocol palette (blue goals, gray process,
white inputs, grayscale rules); the `asciifig` frame and the `L`/`Y`/`R`/`C`
table columns are carried at the body measure; and a PNG-free TikZ ORCID iD mark
replaces the template's `orcid_icon.png` (Rule 3). The senior-author formatting
is centralized here: RaggedRight with even interword spacing, maximal
widow/orphan/broken penalties, any-character URL breaking, single hyphens, and
the section symbol for codified references.

`main.tex` carries the cover (title, author with the ORCID mark, DOI placeholder
`10.5281/zenodo.xxxxxxxx`, June 20, 2026, v4.0.0, and the independent-research
disclaimer), a clickable table of contents, and one `\input` per section. The
thirteen section files map one-to-one onto the NIH template: Statement of
Compliance; Protocol Summary (Synopsis, Schema, Schedule of Activities);
Introduction (Rationale, Background including the eight Physical AI concerns,
Risk/Benefit); Objectives and Endpoints; Study Design; Study Population; Study
Intervention; Discontinuation/Withdrawal; Assessments and Procedures; Statistical
Considerations; Regulatory, Ethical, and Oversight Considerations; Additional
Considerations, Abbreviations, and Amendment History; and References and Back
Matter. Each was committed as its own file in real time (Rule 6).

The bracketed instructions wire the draft to its sources: the three counterfactual
PFS/OS scenarios and the eight Physical AI concerns are pointed at
`mermaid/fig-19` and `fig-20`; the combined IND/IDE spine at `fig-03`; the
intervention at `fig-04`, `fig-05`, `fig-09`, and the per-arm and sensor tables
of `inputs/2030-pdac-1min-final-paper`; the safety and reporting machinery at
`fig-07`, `fig-08`, `fig-15` and `inputs/21cfr312_adapt`; and the statistics at
`fig-22` and `nih-protocol/07`. `references.bib` assembles the five DARAXONRASIB
entries, the three main documents, the directly relevant author works, the
clinical references (Siegel 2025, the Dutch 2025 cohort, Conroy 2018, Bassi
2017, the daraxonrasib programs), and the FDA / CFR / consensus standards.

A static review confirmed balanced braces and environments, fully escaped
underscores and special characters, and no text running off the measure; the
second-to-last commit recorded that error pass, and the stage closed with the
directory README and the Overleaf zip bundle. The draft is the contract the full
stage executes instruction by instruction.
