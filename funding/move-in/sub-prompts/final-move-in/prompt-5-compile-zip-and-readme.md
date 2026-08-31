# Stage 3, sub-prompt 5 - compile, bundle, README, and close

## Compile sequence

`pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. The stage closes only at 0 errors,
0 overfull boxes, 0 underfull boxes, 0 undefined citations and 0 undefined
references.

## No publication directory

The master prompt says not to generate `final-move-in/publication`. It is not
generated. The parent build at `funding/pdac-funding-applications/final-apply/`
carries one and this stage deliberately does not.

## The bundle

`final-move-in-LaTeX.zip`, rebuilt from the same sources in the same pass as the
compile, so it cannot be older than the files beside it. Contents: `main.tex`,
`movestyle.sty`, `references.bib`, `sections/*.tex`. Nothing else.

## READMEs

`final-move-in/README.md` and `final-move-in/sections/README.md`. Both
comprehensive, both with badges, both carrying the Rule 5 source map, the
measured page count, the measured character count against the 150,972-character
template budget, and the compile result.

## The repository close

The last commit of the whole build, not of this stage alone, performs the
remaining repository updates: the root `README.md` with one new section and the
updated repository structure, `CHANGELOG.md` at v4.7.0, `releases.md` in the
required format, and `prompts/output-move-in.md`.

## Acceptance

- The author opens `final-move-in-LaTeX.zip` in Overleaf, compiles with
  pdfLaTeX, and fixes nothing.
- The three `lint-and-format` checks pass, because no Python file is added.

## Commit

Two commits for the stage, then one for the repository close.
