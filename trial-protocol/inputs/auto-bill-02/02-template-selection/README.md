# 02-template-selection - Stage 2: template bill selection (v5.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20of%207-blue.svg)](.)
[![Selected](https://img.shields.io/badge/Selected-VVUQ--05%20final--bill%20(v3.0%20base)-darkblue.svg)](.)

Stage 2 of Process B. Running
[`../sub-prompts/prompt-2-template-selection.md`](../sub-prompts/prompt-2-template-selection.md)
produced [`output-2-template-selection.md`](output-2-template-selection.md):
the Bill v3.0 file set (`cancer-automated/.../papers/VVUQ-05/final-bill`,
without `/deliverables`) is **selected as the base** of H. R. 9510 v5.0, with
the gray-scale `mermaidfig` TikZ primitives and the `\billbox` device imported
from the unselected candidate (`auto-bill-01/final-bill`), and a new
right-aligned `R{<width>}` numeric column added for fiscal-year dollar tables.
The output also fixes the medium genre (table, ASCII, or Mermaid) each part of
the bill leads with.

## Files

| File | Role |
|:--|:--|
| [`output-2-template-selection.md`](output-2-template-selection.md) | The Stage 2 narrative output (candidates, decision, merge plan, genre map) |
| `README.md` | This index |

## Sources used from other directories (Rule 6)

| No. | Asset used | Upstream source |
|:--|:--|:--|
| 01 | The template sub-prompt run here | `auto-bill-02/sub-prompts/prompt-2-template-selection.md` |
| 02 | Candidate (a), selected: bill apparatus, ASCII and table primitives, eight sections | `cancer-automated/.../papers/VVUQ-05/final-bill` (`main.tex`, `usctitle.sty` v3.1.0, `references.bib`, `sections/`) |
| 03 | Candidate (b), merged from: gray-scale Mermaid TikZ primitives, `\billbox` | `single-prompt-bill/auto-bill-01/final-bill` (`usctitle.sty` v4.0.0, `main.tex`, `sections/`) |
| 04 | Financial vocabulary the genre map enforces | `auto-bill-02/01-research/output-1-research.md` |

## Where this stage flows next

Stage 3 fills the Mermaid and ASCII genre slots with concrete diagrams; Stage 4
numbers them; the three bill stages implement the merged `usctitle.sty` v5.0.0.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
