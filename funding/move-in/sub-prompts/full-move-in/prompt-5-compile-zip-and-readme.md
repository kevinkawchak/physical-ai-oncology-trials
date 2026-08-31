# Stage 2, sub-prompt 5 - compile, bundle, README, and the stage error pass

## Compile sequence

Four passes, `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. The stage is not
finished until the log shows 0 errors, 0 overfull boxes, 0 underfull boxes,
0 undefined citations and 0 undefined references.

## The stage error pass (Rule 7)

Stage 2 introduces prose and tables, so it introduces the defect classes stage 1
could not have. Each is checked mechanically:

| Check | Command form | Must return |
|:--|:--|:--|
| Overfull boxes | `grep -c 'Overfull' main.log` | 0 |
| Underfull boxes | `grep -c 'Underfull' main.log` | 0 |
| Undefined citations | `grep -c 'Citation.*undefined' main.log` | 0 |
| Surviving drafting instructions | `grep -rc 'draftnote' sections/` | 0 |
| Columns missing the ragged prefix | a paired grep over `p{` | 0 |
| Dialect words | a grep over the word list | 0 |
| Em dash, en dash, double hyphen | `grep -n -- '---\|--\|—\|–' sections/*.tex` | 0 outside `%` comments |
| Literal `SS` for `§` | `grep -n 'SS' sections/*.tex` | 0 |
| "estimated" beside $36,330 | a grep | 0 |
| Uncited bibliography entries | compare `\cite` keys against `references.bib` | equal sets |

## The bundle

`full-move-in-LaTeX.zip`, built from the same sources in the same pass as the
compile. `main.tex`, `movestyle.sty`, `references.bib`, `sections/*.tex`.

## README

`full-move-in/README.md` and `full-move-in/sections/README.md`, both
comprehensive, both with badges, both carrying the Rule 5 source map and the
measured page count and character count of the stage.

## Commit

Two commits: the error pass, then the READMEs and the bundle.
