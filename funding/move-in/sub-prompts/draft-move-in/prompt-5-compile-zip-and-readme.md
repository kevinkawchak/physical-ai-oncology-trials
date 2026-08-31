# Stage 1, sub-prompt 5 - compile, bundle, README, and the stage error pass

## Goal

Close stage 1 with a source set the author can open in Overleaf and compile
without touching anything.

## Compile sequence

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Four passes, not two. The third and fourth resolve the contents, the part
counter, and every cross-reference. A stage that leaves undefined references is
not finished.

## The bundle

`draft-move-in-LaTeX.zip` contains `main.tex`, `movestyle.sty`,
`references.bib` and `sections/*.tex`, and nothing else. No `.aux`, no `.log`,
no `.pdf`, no `README.md`. It is rebuilt from the same source in the same pass
that produced the compile log, so the zip can never be older than the sources
beside it.

## README

`draft-move-in/README.md` and `draft-move-in/sections/README.md`, both
comprehensive, both with badges, both carrying a Rule 5 table that says which
files from other directories were used and where.

## The stage error pass (Rule 7)

The second-to-last commit of the stage fixes every defect found in the stage.
Report each with its measured size rather than a description:

| Defect class | How it is detected |
|:--|:--|
| Overfull box | `grep -c 'Overfull' main.log`, with the point size of each |
| Underfull box | `grep -c 'Underfull' main.log` |
| Undefined citation | `grep -c 'undefined' main.log` |
| Table wider than the measure | Any `tabularx` whose column widths plus `\tabcolsep` exceed `\textwidth` |
| Column without `\raggedright\arraybackslash` | A grep over `p{` in every section file |
| Dialect word | A grep over the word list |
| Em dash, en dash, double hyphen | A grep over the section files |
| Literal `SS` where `§` belongs | A grep over the section files |

## Acceptance

- 0 errors, 0 overfull boxes, 0 underfull boxes, 0 undefined citations,
  0 undefined references.
- The zip opens and compiles in a clean directory.

## Commit

Two commits: the error pass, then `move-in/draft: README, sections README, and
the Overleaf bundle`.
