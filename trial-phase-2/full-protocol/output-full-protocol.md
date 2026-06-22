## output-full-protocol

This is the narrative output of the Stage 3 (full) sub-prompt. The bracketed
scaffold from Stage 2 was rendered into complete prose, with every figure drawn as
a TikZ `mermaidfig` and every table filled at the body measure.

### Full-stage actions

- Rendered all 13 NIH sections (Statement of Compliance through References and Back
  Matter) in full prose for the Phase 2, multicenter, randomized, controlled
  design (n = 220, 110 per arm; eight high-volume academic centers; primary
  progression-free survival at hazard ratio 0.60 with 85 percent power and about
  140 events; the fixed-sequence key secondary hierarchy of overall survival, R0
  rate, ISGPS grade B/C fistula, major pathologic response, and ctDNA clearance).
- Drew all 22 figures as TikZ `mermaidfig` from the Mermaid catalog, verifying
  twice that each figure has no text-box or arrow overlap, that curved arrows have
  the correct looseness, and that boxes are properly spaced.
- Filled all 11 full-width tables at the body measure, including the Schedule of
  Activities, the nine Physical AI concerns, the co-investment tranches, the
  objectives and endpoints, the per-arm tool and sensor tables, the power and
  secondary-endpoint tables, and the cross-jurisdiction, amendment-history, and
  abbreviations tables.
- Carried the experimental arm (perioperative daraxonrasib at the RP2D plus the
  on-premises LLM-directed eight-arm robotic Whipple), the control arm (modified
  FOLFIRINOX plus standard pancreaticoduodenectomy), the upgraded Phase 0 gate,
  the four counterfactual scenarios, the nine Physical AI concerns, and the
  Patient-Aligned Co-Investment Facility and capital firewall.

### Result

The full render compiles with pdfLaTeX. The senior-author pagination and
white-space polish (`\raggedbottom` and a `\clearpage` after each section) is
applied in the next stage, [`../final-protocol/`](../final-protocol).
