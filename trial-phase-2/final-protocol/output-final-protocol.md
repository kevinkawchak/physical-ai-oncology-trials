## output-final-protocol

This is the narrative output of the Stage 4 (final) sub-prompt. The full protocol
from Stage 3 was carried into its polished, final form and the author-edited
publication build was produced in [`publication/`](publication).

### Final-stage actions

- Re-enabled `\raggedbottom` in `protostyle.sty` so pages are not vertically
  stretched, removing the large inter-paragraph white gaps the full stage left.
- Added a `\clearpage` after every NIH section in `main.tex` so each of the 13
  sections is self-standing and no content is stranded across a boundary.
- Re-verified every TikZ `mermaidfig` for text-box and arrow overlaps, curved-arrow
  looseness, and box spacing, across all 22 figures.
- Confirmed the clickable, page-filling table of contents lists every NIH section
  and the back matter, and that the PNG-free TikZ ORCID iD mark and the DOI
  placeholder render correctly.
- Verified internal consistency of the locked constants (n = 220, 110 per arm; HR
  0.60; 85 percent power; about 140 progression-free-survival events; RP2D 300 mg
  once daily; eight sites; Phase 0 USL greater than or equal to 8.0; at least 5000
  simulated procedures across at least 3 frameworks; 3 N per-arm and 18 N
  cumulative force caps; 3 ms cross-arm emergency stop) across every section,
  figure, and table.
- Confirmed single hyphens only, the section symbol for every codified reference,
  no raster images, white background, and Burgundy `#800020` as the document color.

### Result

The 13 final sections (Statement of Compliance through References and Back Matter),
the recolored `protostyle.sty`, the `references.bib`, the polished `main.tex`, and
the Overleaf zip compile cleanly with pdfLaTeX. The author-edited `publication/`
build is the paper URL directory used for deposition.
