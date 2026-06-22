## output-protocol

This file records the narrative output of the `prompt-protocol.md` build for the
Phase II protocol (the Claude Code markdown narrative, not the LaTeX source
itself). The build grows the protocol in four stages, mermaid -> draft -> full ->
final, then adds the author-edited `final-protocol/publication` paper, exactly as
the Phase I `trial-protocol` build did, and recolors the entire document to the
five-step Phase II palette with Burgundy `#800020` as the document color.

### What was generated

1. **Bootstrap (Process A).** From this single master prompt, four stage
   sub-prompts were generated into [`../sub-prompts/`](../sub-prompts)
   (`prompt-1-mermaid`, `prompt-2-draft-protocol`, `prompt-3-full-protocol`,
   `prompt-4-final-protocol`), adapting the Phase I `trial-protocol` workflow to a
   Phase II, multicenter, randomized, controlled study. The recolored
   `protostyle.sty` (Burgundy `#800020`, Charcoal `#2E2E2E`, Slate `#6B6B6B`,
   Mist `#C9C9C9`, Cloud `#F5F5F5`) and the extended `references.bib` were
   authored as the shared foundation.

2. **Stage 1, mermaid.** A catalog of new, professionally colored Mermaid figures
   was authored into [`../mermaid/`](../mermaid), one file per figure, each opening
   with a real ```mermaid``` block that renders in GitHub and is reproduced as an
   identical TikZ `mermaidfig` in the LaTeX stages. None reuse a prior author
   figure. New Phase II figures include the randomized multicenter schema, the
   stratified randomization and BICR design, the co-investment to
   success-likelihood mapping, the capital-firewall governance, the
   group-sequential interim, ctDNA clearance, and federated learning across sites.

3. **Stage 2, draft-protocol.** The 13-section NIH scaffold was authored with
   bracketed `[DRAFTING INSTRUCTION]` pointers naming the exact repository files
   and directories each later stage consumes, plus a table of contents and back
   matter, and an Overleaf zip.

4. **Stage 3, full-protocol.** The 13 sections were rendered in full prose with
   every figure drawn as a TikZ `mermaidfig` and every table filled at the body
   measure, plus an Overleaf zip.

5. **Stage 4, final-protocol and publication.** The full protocol was polished to
   maximum quality (`\clearpage` per NIH section, `\raggedbottom`, ragged-right
   bibliography, the PNG-free TikZ ORCID iD mark, single hyphens, the section
   symbol for every codified reference), then copied to
   `final-protocol/publication` with author edits as the paper URL directory.

### The Phase II upgrade in one paragraph

The Phase I protocol was a single-arm, first-in-human, combined IND/IDE study
that established the daraxonrasib recommended Phase 2 dose (RP2D, 300 mg once
daily) and the feasibility and safety of the on-premises LLM-directed eight-arm
robotic Whipple. With those questions answered, genuine clinical equipoise now
exists, so this Phase II protocol is the randomized controlled efficacy study the
Phase I protocol explicitly deferred: 220 participants randomized 1:1 across eight
high-volume academic centers, with progression-free survival as the powered
primary endpoint (hazard ratio 0.60, 85 percent power, two-sided alpha 0.05, about
140 events, one group-sequential interim) and a fixed-sequence hierarchy of key
secondary endpoints (overall survival, R0 resection rate, ISGPS grade B/C fistula
rate, major pathologic response, and circulating-tumor-DNA clearance). The device
readiness bar is raised (Unified Safety Level greater than or equal to 8.0, at
least 5000 simulated procedures across at least 3 independent frameworks,
sim-to-real tightened to less than 1.5 mm and less than 0.4 N), with multicenter
fleet harmonization, a federated hash-chained audit trail, a single IRB of record,
and an independent DSMB.

### The co-investment model (A)

The protocol's distinctive Phase II addition is a Patient-Aligned Co-Investment
Facility (PACIF): wealthier individuals connected to cancer patients contribute
ring-fenced philanthropic and private capital to raise the trial's success
likelihood. A capital firewall structurally prevents any funder from influencing
randomization, endpoint definition, outcome adjudication, statistical analysis,
data access, or publication. Capital purchases only operational success-likelihood
levers: the eight-site network (accrual speed and generalizability), expanded
Phase 0 simulation compute (lower sim-to-real gap and a higher Unified Safety
Level), a robot fleet with redundancy (throughput and uptime), a Patient Access
and Equity Fund (travel, lodging, lost-wage, caregiver, and navigation support,
which raises retention and so protects statistical power while enabling equitable,
representative enrollment), central review and biomarker infrastructure (blinded
independent central review, central pathology, ctDNA, federated learning,
biostatistics), and independent oversight. The facility is disclosed and managed
under 21 CFR part 54 and aligned with the H. R. 9510 verification-before-generation
VVUQ financial-data standard. The message is precise: money raises power,
fidelity, retention, equity, and generalizability, and never buys a favorable
answer, because integrity is protected structurally rather than by trust alone.

### Commit and PR discipline

Every distinguishable file is a separate commit, pushed in real time, so branch
progress is monitored without user intervention. For each stage the second-to-last
commit fixes all errors across files, and the last commit performs the remaining
repository updates. The final repository update sets the version to v4.1.0 across
the root `README.md`, `CHANGELOG.md`, and `releases.md`, doubles the badge set,
adds a dedicated v4.1.0 section, and extends the repository structure tree with
`trial-phase-2/`. No PNG, Python, or YAML files are added, so the CI
lint-and-format workflow remains green.
