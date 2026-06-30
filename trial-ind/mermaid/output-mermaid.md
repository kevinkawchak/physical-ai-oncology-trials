## output-mermaid

Stage 1 narrative for the *Phase 1 PDAC IND: AI Generation* build (IND v1.0,
repository v4.3.0). Process B executed
[`../sub-prompts/prompt-1-mermaid.md`](../sub-prompts/prompt-1-mermaid.md) in
`trial-ind/mermaid/`.

### What was produced

- 22 new grayscale Mermaid figures (`fig-01` to `fig-22`), one commit per figure,
  each pushed to the working branch the moment it was generated so the branch is
  monitorable without intervention.
- Each figure opens with a native ```` ```mermaid ```` block (renders on GitHub),
  followed by a caption, its role in the IND, and the exact repository source
  files it draws from (Rule 5).
- A strictly grayscale eight-tone palette mapping one-to-one to the
  `indstyle.sty` `mm*` node styles, so each figure reproduces identically in the
  LaTeX stages. The body text of the IND keeps the template color (black).

### Perspectives covered (no overlap between figures)

The catalog blends two perspectives. The IND-acceleration process figures (1, 4,
5, 6, 7, 8, 9, 10, 11, 12, 13) adapt, in context and recolored to grayscale, the
acceleration argument of `trial-documents/final-paper/publication` Figures 6, 9,
11, 15, 16, 17, 19, 20, 23, and 24, reframed around the initial IND package. The
IND-content figures (2, 3, 14, 15, 16, 17, 18, 19, 20, 21, 22) render the clinical
and regulatory substance of the submission, the drug mechanism, the 3+3 dose
escalation, CMC, pharmacology and toxicology, the perioperative advisory, the
safety-reporting triggers, the governance, the objective-to-endpoint hierarchy,
and the previous human experience, carried from
`trial-protocol/final-protocol/publication`.

### Quantitative content carried into the figures

The figures carry the real numbers needed for Phase 1 review: the DL1 160 / DL2
220 / DL3 300 mg dose levels and the 3+3 / 28-day-DLT rule; the up-to-18 sample
size; the 7 and 15-day safety-reporting clocks and the six §312.32(g) Physical AI
triggers; the perioperative advisory sweep (29 / 3 / 0 of 32); the binding halt
rules; and the per-document time savings (initial IND 8 to 12 weeks to 1 to 4
days, and so on).

### Verification (Stage 1 closeout)

Every figure file was checked for a balanced ```` ```mermaid ```` fence, a present
`classDef` set, a `## Figure` heading, a named source-files block, and for every
`:::class` reference resolving to a `classDef` defined in the same file. All 22
files passed. The figures are improved further in the draft, full, and final
stages, where they are reproduced as TikZ and verified twice for overlaps, arrow
looseness, and box spacing.
