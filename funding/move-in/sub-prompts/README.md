# sub-prompts - the three-stage schedule that builds the La Jolla package (v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stages](https://img.shields.io/badge/Stages-3-00417A.svg)](.)
[![Sub--prompts](https://img.shields.io/badge/Sub--prompts-15-3C7DB2.svg)](.)
[![Documents](https://img.shields.io/badge/Documents-15-00417A.svg)](../final-move-in)
[![Section files](https://img.shields.io/badge/Section%20files-17%20per%20stage-6C757D.svg)](../final-move-in/sections)
[![Diagrams](https://img.shields.io/badge/Diagrams-none%2C%20by%20Rule%203-9AA1A8.svg)](.)
[![Compiler](https://img.shields.io/badge/Compiler-pdfLaTeX%20%2B%20BibTeX-6C757D.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../../../README.md)

The parent build at
[`funding/pdac-funding-applications/sub-prompts/`](../../pdac-funding-applications/sub-prompts)
ran two schedules across thirteen sub-prompts, because it had two deliverables
and five diagram platforms to produce. This build has one deliverable and Rule 3
forbids diagrams, so the schedule collapses to the three stages the master
prompt names, and each gets its own directory rather than a single flat list.

## The schedule

| Stage | Directory | Sub-prompts | Output | Commit floor |
|:--|:--|:--|:--|:--|
| 1 | [`draft-move-in/`](draft-move-in) | 5 | [`../draft-move-in/`](../draft-move-in): skeleton with bracketed drafting instructions naming exact repository paths | 10 |
| 2 | [`full-move-in/`](full-move-in) | 5 | [`../full-move-in/`](../full-move-in): every instruction resolved, all fifteen documents written, all tables populated | 10 |
| 3 | [`final-move-in/`](final-move-in) | 5 | [`../final-move-in/`](../final-move-in): the senior-author pass. No `publication/` subdirectory, by instruction | 10 |

Each stage emits the same five artifacts, so a reader can diff any two stages
file for file:

```
<stage>/
  README.md                  comprehensive, with badges and a Rule 5 source map
  main.tex                   cover, contents, and one \input per section
  movestyle.sty              the shared style, identical across the three stages
  references.bib             the shared bibliography, identical across the three
  sections/
    README.md                what each section file carries
    sec-00-front.tex         front matter
    sec-01 .. sec-15         one file per document
    sec-16-backmatter.tex    abbreviations, availability, citation, references
  <stage>-LaTeX.zip          the Overleaf bundle, rebuilt in the same pass
```

## Why fifteen documents

The San Francisco predecessor carried eleven. The La Jolla site adds four,
because the master prompt asks for three things the predecessor never had to
answer: conventional pancreatic cancer trial requirements (clause B), a lobbying
and federal funding position (clause C), and a named staff of eleven (clause A).

| Part | Documents | Added for La Jolla |
|:--|:--|:--|
| I, Legislation and lobbying | 01 to 04 | 04, the federal bill that succeeds H. R. 9510 |
| II, Regulations | 05 to 07 | none; 07 is rewritten around LLM and robotic workflow acceptance |
| III, Building and premises | 08 to 11 | none; all four are resized for a single Phase 1 trial |
| IV, Operations | 12 to 15 | 13, the conventional trial requirements manual; 14, staffing and roles; 15, funding stewardship and legislative engagement |

## Invariants every stage inherits

1. Every table is set to `\textwidth` with `tabularx`, and every fixed-width
   column is declared `>{\raggedright\arraybackslash}p{...}`. A column without
   the prefix is a defect, and the audit is a grep.
2. No raster image, no TikZ diagram, no `.png` and no `.jpg` anywhere. The only
   TikZ in the build is cover-page furniture: a badge, a rule, and the ORCID
   mark.
3. Single hyphens only. No em dash, no en dash used as punctuation, no double
   or triple hyphen.
4. The section symbol `§` for every codified reference. A literal `SS` is a
   defect.
5. American English, La Jolla register. A dialect audit runs before every
   commit and no word on the list survives.
6. `\RaggedRight` with a bounded `\RaggedRightRightskip`, so interword space is
   even and nothing overflows the right margin.
7. `\widowpenalty`, `\clubpenalty`, `\brokenpenalty` and `\displaywidowpenalty`
   at 10000, and a stretchable `\parfillskip`, so no paragraph ends in a one- or
   two-word line and no single line is stranded across a page break.
8. Every stage compiles with pdfLaTeX and BibTeX before its commit, and the zip
   is rebuilt from the same source in the same pass.

## Files used from other directories (Rule 5)

| Source | Used where |
|:--|:--|
| [`../../pdac-funding-applications/sub-prompts/part-ii/prompt-6-draft-apply.md`](../../pdac-funding-applications/sub-prompts/part-ii/prompt-6-draft-apply.md) | The form of the stage 1 sub-prompts: a skeleton carrying bracketed drafting instructions that name exact repository paths |
| [`../../pdac-funding-applications/sub-prompts/part-ii/prompt-7-full-apply.md`](../../pdac-funding-applications/sub-prompts/part-ii/prompt-7-full-apply.md) | The form of the stage 2 sub-prompts, including the column-width optimization method |
| [`../../pdac-funding-applications/sub-prompts/part-ii/prompt-8-final-apply.md`](../../pdac-funding-applications/sub-prompts/part-ii/prompt-8-final-apply.md) | The form of the stage 3 sub-prompts: `\clearpage` discipline and the senior-author proof pass |
| [`../../pdac-funding-applications/final-apply/applystyle.sty`](../../pdac-funding-applications/final-apply/applystyle.sty) | The typography invariants listed above, the `L`, `C`, `R` and `Y` column types, and the `\apptable` wrapper that removes the trailing space which otherwise reports every table as overfull |
| [`../../capitalization-plan/final-capital/`](../../capitalization-plan/final-capital) | The `unsrturl` bibliography style, which prints and links DOI and URL fields that plain `unsrt` silently drops |
| [`../inputs/`](../inputs) | The three input artifacts every stage reads |
| [`../prompts/prompt-move-in.md`](../prompts/prompt-move-in.md) | The master prompt these fifteen sub-prompts expand |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
