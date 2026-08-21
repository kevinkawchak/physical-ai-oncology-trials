# draft-move-in - stage 1 of 3, the compiling skeleton (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-1%20of%203-00417A.svg)](../sub-prompts/draft-move-in)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](sections)
[![Section files](https://img.shields.io/badge/Section%20files-17-3C7DB2.svg)](sections)
[![Drafting instructions](https://img.shields.io/badge/Drafting%20instructions-96-6C757D.svg)](sections)
[![Compile](https://img.shields.io/badge/Compile-0%20errors%2C%2027%20pages-brightgreen.svg)](main.tex)
[![Overfull](https://img.shields.io/badge/Overfull-0-brightgreen.svg)](main.tex)
[![Underfull](https://img.shields.io/badge/Underfull-0-brightgreen.svg)](main.tex)
[![Bundle](https://img.shields.io/badge/Overleaf-draft--move--in--LaTeX.zip-6C757D.svg)](draft-move-in-LaTeX.zip)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI%20v1.0-10.5281%2Fzenodo.xxxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../README.md)

Stage 1 of the three-stage build. It is a compiling skeleton whose job is to
remove every open question before stage 2 begins. Each of the seventeen section
files carries its headings, its table shells, and one bracketed
`[DRAFTING INSTRUCTION]` per subsection. **Every instruction names an exact
repository file or directory**, set with `\mvfile`, so `full-move-in` resolves
instructions rather than inventing content.

## Files

| File | What it is |
|:--|:--|
| [`main.tex`](main.tex) | The cover page, the table of contents in the template's form, and one `\input` per section with a `\clearpage` before every document |
| [`movestyle.sty`](movestyle.sty) | The shared style. Copied unchanged into `full-move-in/` and `final-move-in/`, so a formatting fix is a fix to one file rather than three |
| [`references.bib`](references.bib) | 76 entries. Every entry that has a digital object identifier carries a paired `doi` and `url` field |
| [`sections/`](sections) | 17 files: front matter, fifteen documents, back matter |
| [`draft-move-in-LaTeX.zip`](draft-move-in-LaTeX.zip) | The Overleaf bundle, 21 files, rebuilt from these sources in the same pass as the compile |

## Measured result

| Metric | Value |
|:--|:--|
| Compile | `pdflatex` then `bibtex` then `pdflatex` twice |
| Errors | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 0 |
| Undefined citations | 0 |
| Undefined references | 0 |
| Pages | 27 |
| Source characters, `main.tex` plus `sections/` | 60,992 |
| Drafting instructions carried | 96 |
| Fixed-width table columns, all ragged-prefixed | 51 of 51 |

## What `movestyle.sty` changed, relative to the parent style

| Change | Detail |
|:--|:--|
| Deleted | All five TikZ diagram vocabularies (`mm*`, `uml*`, `d2*`, `dg*`, `gv*`), the `appfig` frame, the `appfloat` carrier, `figcaption`, `figslot`, and the vector glyph macros. Rule 3 forbids diagrams, so the audit is a grep for `appfig`, which returns nothing |
| Added | `\docpart` and `\docfront`, which open a document with a ruled block, restart its section numbering, write a part-level contents line, re-key the hyperref anchors, and set the running header |
| Added | `\mvcover`, `\mvstrip`, `\mvbadge`, `\mvbadgel`, `\mvrule` for the cover |
| Added | `\draftnote` and `\mvfile` for the stage 1 drafting instructions |
| Added | A numbered `\tabcap` and a labeled `\tabcapl`, because a table a funder is asked to read has to be referable by number |
| Kept | The typography block, the `L`, `C`, `R` and `Y` column types, `\mvtable`, `\thead`, `\bmhead`, `\keywords`, `\orcidicon`, the `\UrlBreaks` re-assertion after `url` and `hyperref`, and the compressed contents entries |

## Defects found and fixed in the stage error pass

Four, each with its measured size rather than a description.

| Defect | Size | Cause | Fix |
|:--|:--|:--|:--|
| Math shift error in `sec-00` and `sec-15` | 10 errors, 3 overfull boxes of 81.26 pt, 596.46 pt and 1622.11 pt | An unescaped underscore inside a `\mvfile` path put the rest of the paragraph into math mode | The path is written with an escaped underscore |
| `Undefined color 'ACCENTCOL'` | 3 errors | `\contentsname` held `\color{accentcol}`, and `\tableofcontents` passes the name through `\MakeUppercase` for the running mark | The color is taken from the `titlesec` section format instead |
| Underfull part lines in the contents | 3 boxes at badness 10000 | `\l@part` set `\rightskip \@pnumwidth` with no stretch, so a title that wrapped could not fill its first line | `\rightskip \@pnumwidth \@plus 4em` |
| A visible gap between every character of a printed link | a legibility defect rather than a box warning | `\Urlmuskip` at the parent's `0mu plus 3mu` takes almost all of a ragged-right line's available stretch | `0mu plus 0.45mu`, and the deposit paragraph set with unbounded `\raggedright` |

Two further style defects were found while the style was being written and are
recorded here because they would otherwise recur in any adaptation of the parent
style. The parent's `\AtBeginDocument` redefinition of `\thebibliography` takes a
parameter, and the LaTeX begin-document hook stores its argument in a macro body,
so the literal parameter is read as a parameter of that body and the compile
stops at `\begin{document}`; `movestyle.sty` uses `\apptocmd` instead. And
`\theHsection` does not exist until `hyperref` loads, so the three
document-keyed anchor redefinitions must follow the `\RequirePackage`, not
precede it.

## Files used from other directories (Rule 5)

| Source | Used where in this directory |
|:--|:--|
| [`../../pdac-funding-applications/final-apply/applystyle.sty`](../../pdac-funding-applications/final-apply/applystyle.sty) | `movestyle.sty`: the typography block, the column types, `\mvtable` from `\apptable`, `\thead`, `\bmhead`, `\keywords`, `\orcidicon`, `\mvfile` from `\appfile`, `\dlink`, `\mvrefs` from `\apprefs`, and the compressed `\l@section` |
| [`../../pdac-funding-applications/final-apply/main.tex`](../../pdac-funding-applications/final-apply/main.tex) | `main.tex`: the cover structure, the badge line, the keyword line, and the `\clearpage` commentary |
| [`../../capitalization-plan/final-capital/capstyle.sty`](../../capitalization-plan/final-capital/capstyle.sty) | The `unsrturl` bibliography style, which prints and links `doi` and `url` fields that plain `unsrt` drops |
| [`../inputs/Physical-AI-Oncology-Clinical-Trial-Site-Complete-Documentation-Package.zip`](../inputs) | `physical_ai_legislation.sty`: one part per document, document-keyed hyperref anchors, and the lettered subdivision idiom. `all_documents.bib`: four California artificial intelligence bills and the author entries carried into `references.bib` |
| [`../inputs/READMES/`](../inputs/READMES) | `references.bib`: the twenty deposited papers with their dates and identifiers, and the predecessor package entry |
| [`../inputs/`](../inputs) | `references.bib`: the three 2024 preprint entries and the 2025 clinical decision support entry, from the accomplishments record |
| [`../../pdac-funding-applications/final-apply/references.bib`](../../pdac-funding-applications/final-apply/references.bib) | `references.bib`: the four policy entries, and the practice of pairing every `doi` with a `url` |
| [`../sub-prompts/draft-move-in/`](../sub-prompts/draft-move-in) | The five sub-prompts this stage executes |

## Next stage

[`../full-move-in/`](../full-move-in) answers all 96 drafting instructions and
deletes them. A surviving instruction is a defect there, and the audit is a
recursive grep over `sections/`, which must return zero.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
