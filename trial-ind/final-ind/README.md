# final-ind - Stage 4 (polished) (IND v1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![IND](https://img.shields.io/badge/IND-Phase%201%20First--in--Human-000000.svg)](.)
[![Indication](https://img.shields.io/badge/Indication-KRAS%20PDAC-3F3F3F.svg)](.)
[![Intervention](https://img.shields.io/badge/Intervention-Robotic%20Whipple%20%2B%20Daraxonrasib-3F3F3F.svg)](.)
[![Template](https://img.shields.io/badge/Template-ReGARDD%20IND-6C757D.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication)
[![Figures](https://img.shields.io/badge/Grayscale%20figures-22-000000.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication)
[![Method](https://img.shields.io/badge/Method-mermaid%E2%86%92draft%E2%86%92full%E2%86%92final-6C757D.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-6C757D.svg)](https://orcid.org/0009-0007-5457-8667)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-6C757D.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21097442-blue.svg)](https://doi.org/10.5281/zenodo.21097442)

[Publication with Author Edits](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) (this directory). Stage 4 of the `trial-ind/` build: the final IND, at maximum context and formatting
quality. It carries over the full IND, deepens the regulatory and clinical prose
toward the ten-times character target, unifies the figure numbering into a single
document sequence (Figure 1 to Figure 22, so all 22 grayscale figures appear) and
the table numbering into section sequences (Table 1.1, Table 3.1, and so on), and
applies the senior-author polish learned from
[`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication):
`\clearpage` per self-standing section, tuned full-width table column widths,
`\vspace`/`\hspace`/`\needspace` to remove large empty space without overcrowding,
even `\RaggedRight` interword spacing with no right-margin overflow, no stranded or
one-to-two-word lines, single dashes only, and the section symbol for codified
references. There is no `publication` subdirectory under `final-ind`.

## Files

| File | Purpose |
|:--|:--|
| [`main.tex`](main.tex) | Cover page, ReGARDD ordering, numbered Table of Contents, `\input` of all sections, `\clearpage` per section. |
| [`indstyle.sty`](indstyle.sty) | The shared grayscale style. |
| [`references.bib`](references.bib) | Author `@misc` entries; `ieeetr`; clickable URLs and DOI text plus clickable DOI URLs; long links break on any character. |
| `sections/sec-00 .. sec-11` | The 12 polished IND sections (Rule 6). |

## What changed from the full IND (the final corrections)

- Figure numbering unified to a single sequence, Figure 1 to Figure 22, in
  document order; every figure from the Stage 1 catalog is rendered.
- Table numbering unified to section sequences (Table `<section>`.`<n>`), each
  table at the body measure with tuned ragged-right column widths.
- Prose deepened with additional regulatory detail, worked numeric examples, and
  21 CFR cross-references, toward the ten-times character target.
- White-space polish: `\needspace` before every float, no stranded lines, no
  one-to-two-word lines, no large empty gaps, single dashes, `\S` for sections.
- Each figure re-verified twice for text-box and arrow overlaps, curved-arrow
  looseness, and box spacing.

## Files from other directories used here (Rule 5)

| Source | Used for |
|:--|:--|
| [`../full-ind/`](../full-ind) | the full IND, refined here (not overwritten) |
| [`../mermaid/fig-01 .. fig-22`](../mermaid) | the grayscale figures at full fidelity |
| [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | the `\clearpage` / `\vspace` / table-width proof-reading techniques and the clinical data |
| [`../../trial-documents/final-paper/publication/sections/sec-08-references-backmatter.tex`](../../trial-documents/final-paper/publication/sections/sec-08-references-backmatter.tex) | the back matter adapted here |
| [`../inputs`](../inputs) | the ReGARDD IND template, FDA 1571 instructions, ReGARDD guidance, references |

## Compile (Overleaf, pdfLaTeX)

`pdflatex main` then `bibtex main` then `pdflatex main` twice. The packaged
`final-ind-LaTeX.zip` is a self-contained project.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
