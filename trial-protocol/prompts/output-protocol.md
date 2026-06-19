## output-protocol

This file records the Claude Code (Opus 4.8, 1M context) narrative output for the
autonomous, single-prompt build of the Physical AI Pancreatic Whipple plus
Daraxonrasib Phase 1 Clinical Trial Protocol (v4.0.0). It is the markdown
narrative of the run, not the generated source files; the LaTeX, Markdown, and
BibTeX deliverables live in the stage directories under `trial-protocol/`.

### Approach

I began by reading the workflow exemplar (`inputs/auto-bill-02`, its
`master-prompt.md`, the seven sub-prompts, and the final-bill `usctitle.sty` with
its `mermaidfig` TikZ primitive, `asciifig` frame, and `L`/`Y`/`R` table
columns), the NIH-FDA IND/IDE protocol template README and section chunks, the
recolorable paper template, and the three main documents. Four extraction agents
ran in parallel to pull the quantitative clinical data (the 2030 PDAC 60-second
Whipple simulation: eight arms, 56 DOF, 640 sensor channels, 3 N and 18 N force
caps, 10 kHz heartbeat, 100 microsecond watchdog, 3 ms E-stop, the five-vessel
1.0/1.5/3.0/5.0 mm safety zones, the PJ/HJ/GJ ring-tension bands, the 29/3/0 of
32 daraxonrasib restart distribution against the 0.5 ng/mL trough), the
regulatory overlay (21 CFR part 312 Subpart J, USL thresholds, Phase 0 simulation
validation, the 7-day and 15-day Physical AI reporting timelines, the clinical
hold grounds), the 2026 research framing (the IDE early-feasibility,
significant-risk, Phase Not Applicable device framing, and the zero approved
generative LLM devices against 1,016 to 1,450+ narrow authorizations), and the
author works and NIH section structure. I read the auto-bill-02 `usctitle.sty`
and a real `mermaidfig` example directly to replicate the figure conventions
exactly.

### What was built, milestone by milestone

M1 bootstrap. I created the branch tracking, opened one continuously updated pull
request, filed the master prompt verbatim in `prompts/prompt-protocol.md`,
generated the four stage sub-prompts (mermaid, draft, full, final) under
`sub-prompts/`, wrote the build-hub README and the `prompts/`, `sub-prompts/`,
`research/`, and `inputs/` directory READMEs, and recolored the paper template
accent to Corporate Blue `#00417A`. Every file was a separate commit pushed in
real time.

M2 Stage 1 mermaid. I authored 25 new, professionally colored Mermaid figures,
one commit per figure, none reused from prior author work. They carry the
protocol's quantitative spine: the trial schema, the CONSORT flow, the combined
IND/IDE pathway, the on-premises LLM advisory loop with second-opinion-oracle
isolation, the eight-arm platform, the eight operative phases, the five-vessel
safety gate, the heartbeat and E-stop architecture, the daraxonrasib advisory,
the 3+3 escalation, the three anastomoses, the VVUQ ten-gate, the
objectives-endpoints hierarchy, the Schedule of Activities, the AE and Physical
AI AE reporting, the governance structure, the informed-consent opt-out, the
staged-autonomy model, the three counterfactual PFS/OS scenarios, the eight
Physical AI concerns and mitigations, the audit trail, the analysis populations,
the author-works trust timeline, the risk-benefit framework, and the sensor-data
pyramid. The palette is Corporate Blue for goals and the system, Professional
Gray for process and oversight, Classic White for inputs, and black or grayscale
for rules and raw data. An error-fix commit escaped an ampersand in the VVUQ
figure, and the stage closed with a README figure inventory and a narrative.

M3 Stage 2 draft. I built the draft as a compiling LaTeX scaffold: `main.tex`
with a cover, the ORCID iD mark, the DOI placeholder, and a clickable table of
contents; `protostyle.sty` merging the recolored paper template with the
auto-bill-02 primitives (the `mermaidfig` TikZ environment recolored to the
protocol palette, the `asciifig` frame, the `L`/`Y`/`R`/`C` columns, a PNG-free
TikZ ORCID mark, and the senior-author RaggedRight, widow/orphan, and
URL-breaking rules); `references.bib`; and thirteen section files, one per NIH
section, each carrying bracketed `[DRAFTING INSTRUCTION]` pointers naming the
exact repository files and figure media the full stage would process. A static
review confirmed balanced braces, escaped specials, and an overlap-free running
header, and the stage shipped its zip.

M4 Stage 3 full. Four parallel section-writing agents executed the bracketed
instructions into finished senior-author prose, rendered 20 of the 25 Mermaid
figures as TikZ `mermaidfig` environments, and filled 11 full-width tables (the
Schedule of Activities, the three-column objectives table, the per-arm tooling
and sensor-channel inventories, the abbreviations glossary, and more). I verified
that all cite keys resolved, that the environments balanced 47 to 47, and that
the daraxonrasib dose ladder (160, 220, 300 mg) and the sample size (n up to 18)
were consistent across sections, then closed the stage with an error-fix pass, a
README, and a zip.

M5 Stage 4 final. I refined the full protocol to maximum quality, implementing
the corrections the full stage revealed: the counterfactual figure expanded to
all three scenarios and the Physical AI concerns figure expanded to all eight
concern-mitigation pairs, both overlap-free; `\clearpage` between every NIH
section so each is self-standing; `\raggedbottom` and `\frenchspacing` to remove
large inter-paragraph gaps and even the spacing; and a proof-reading pass that
confirmed pure-ASCII text with single hyphens, the section symbol for codified
references, and a ragged-right bibliography. The stage shipped its own zip.

M6 release. I updated the root `README.md` (fifteen badges, a 425-character
summary, a dedicated v4.0.0 section with a colored Mermaid pipeline, tables, and
a table of contents, and the `trial-protocol/` structure tree), added the v4.0.0
entries to `CHANGELOG.md` and `releases.md`, marked the build-hub status
complete, and verified the continuous-integration lint-and-format workflow stays
green (no Python or YAML files were added; `ruff check` and `ruff format` pass on
the whole repository).

### Result

The deliverable is a complete Physical AI Phase 1 oncology trial protocol that
addresses every NIH template section, carries the quantitative clinical and
regulatory data from the author sources, makes the case that withholding the
LLM-plus-robot-plus-medicine combination shortens progression-free and overall
survival for advanced pancreatic-cancer patients, and answers the eight Physical
AI concerns with concrete mitigations. Three LaTeX sets (draft, full, final),
each with its own zip, compile in Overleaf with pdfLaTeX. The whole build was a
single Claude Code Opus 4.8 session, one commit per file pushed in real time, in
one continuously updated pull request, with no shortcuts between stages.
