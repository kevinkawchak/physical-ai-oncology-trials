## output-move-in

I read the three input artifacts, the parent build at
`funding/pdac-funding-applications`, and the San Francisco predecessor package,
then built the La Jolla move-in documentation package in three stages under one
pull request, pushing every commit the moment it was made.

### What I decided before writing anything

**Fifteen documents, not eleven.** The master prompt asks me to identify the
correct number and type of documents. The San Francisco predecessor carries
eleven. Clause A asks for a chief executive and ten coworkers, clause B for
conventional pancreatic cancer trial requirements, and clause C for a lobbying
and federal funding position. None of those three has a counterpart in the
predecessor, so four documents were added and the roster became:

| Part | Documents | New for La Jolla |
|:--|:--|:--|
| I, Legislation and lobbying | 01 SB 1188, 02 AB 3162, 03 SB 964, 04 H. R. 10412 | 04, the federal bill succeeding the author's own H. R. 9510 |
| II, Regulations | 05 San Diego Municipal Code, 06 Title 22 Chapter 15, 07 FDA compliance guide | none; 07 is rewritten around LLM and robotic workflow acceptance |
| III, Building and premises | 08 Building code, 09 Premises code, 10 Parking, 11 Emergency plan | none; all four resized for a single Phase 1 trial |
| IV, Operations | 12 Activation and SOPs, 13 Conventional trial requirements, 14 Staffing, 15 Funding and lobbying | 13, 14 and 15 |

Document 13 is the load-bearing addition and it is written to pass one test: a
reader who deletes every large language model and every robot from the site
still holds a complete conventional Phase 1 pancreatic cancer requirements
manual.

**The site is sized from the protocol, not from the predecessor.** The San
Francisco site served 168 unique participants in twenty-four hours with
twenty-nine robot instances. A Phase 1 trial in resectable pancreatic ductal
adenocarcinoma treats up to eighteen participants across a 3+3 escalation, so
every dimension of the La Jolla site is derived from the visit schedule instead:
12,400 gross square feet, two robotic procedure suites, eight robot types across
fourteen instances, forty-six parking stalls, extended-day rather than
twenty-four-hour operation, and eleven roles at 3.95 award-funded full-time
equivalents.

**No diagrams.** Rule 3 forbids them, so where the parent build would have drawn
a figure this one writes a table. The package carries 56 full-width tables and
no raster image of any kind. The only TikZ in the style is cover furniture: a
badge, a rule, the fifteen-cell document strip, and the ORCID mark.

**The budget is reused, not re-derived.** The `$700,000` per year and
`$3,500,000` over five years frame comes verbatim from
`funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`.
I split it into six lines that sum to `$700,000` exactly, with personnel at
`$521,000` reconciling to the eleven-role cost table.

### The three stages

| Stage | Directory | Commits | Compile |
|:--|:--|:--|:--|
| 1, draft | `draft-move-in/` | 22 | 0 errors, 27 pages |
| 2, full | `full-move-in/` | 22 | 0 errors, 71 pages |
| 3, final | `final-move-in/` | 22 | 0 errors, 67 pages |

Every stage emits `main.tex`, `movestyle.sty`, `references.bib`, seventeen
section files, two READMEs and its own Overleaf zip. Each zip was unpacked into
an empty directory and compiled with `pdflatex`, `bibtex`, `pdflatex`,
`pdflatex` before its commit, so the author opens it in Overleaf and fixes
nothing.

### Defects found and fixed, with measured sizes

I record these rather than absorbing them, because a defect fixed silently
teaches the next build nothing.

**Stage 1, four defects.**

| Defect | Size | Cause |
|:--|:--|:--|
| Math shift error in two sections | 10 errors and 3 overfull boxes of 81.26 pt, 596.46 pt and 1622.11 pt | An unescaped underscore inside a repository path put the rest of the paragraph into math mode |
| `Undefined color 'ACCENTCOL'` | 3 errors | `\contentsname` held a color command and `\tableofcontents` passes the name through `\MakeUppercase` for the running mark |
| Underfull part lines in the contents | 3 boxes at badness 10000 | `\l@part` set `\rightskip` with no stretch, so a wrapped title could not fill its first line |
| A visible gap between every character of a printed link | a legibility defect, not a box warning | `\Urlmuskip` at the parent's `0mu plus 3mu` takes almost all of a ragged-right line's stretch |

Two further hazards were found while writing the style and are recorded because
they would recur in any adaptation of the parent. The parent's
`\AtBeginDocument` redefinition of `\thebibliography` takes a parameter, and the
LaTeX begin-document hook stores its argument in a macro body, so the literal
parameter is read as a parameter of that body and the compile stops at
`\begin{document}`; `movestyle.sty` uses `\apptocmd` instead. And `\theHsection`
does not exist until `hyperref` loads, so the three document-keyed anchor
redefinitions must follow the `\RequirePackage`, not precede it.

**Stage 2, fourteen overfull boxes and two uncited entries.**

| Defect | Size | Fix |
|:--|:--|:--|
| A bold `Part` header wider than its 0.7 cm column | 0.65 pt | Column widened to 1.0 cm |
| A twelve-character word in a 1.3 cm column, ten occurrences | 21.42 pt each | Column widened to 2.2 cm |
| A hyphenated compound in a 1.7 cm column | 4.55 pt | Column widened to 2.3 cm |
| A twenty-row table that cannot break across pages | Overfull vertical box, 196.14 pt | Moved to `xltabular` |
| A second twenty-row table | Overfull vertical box, 223.34 pt | Moved to `xltabular` |
| `13 CFR 121.702` present in the bibliography, never cited | - | Cited in document 15 §1 |
| The first funding application present, never cited | - | Added to the author record table |

**Stage 3, ten defects, all found by measuring the compiled PDF rather than by
reading the source.** The stage 2 PDF was converted to text, every page was
counted, and every page carrying fewer than twelve body lines or ending on a
heading was investigated at its cause.

| Defect | Fix |
|:--|:--|
| Contents ran to four pages, the fourth carrying two lines | Part lead 0.62 em to 0.42 em, section lead 0.12 em to 0.06 em, contents line spacing 0.94 to 0.92. Now three pages |
| A table caption alone on a page, three lines | `\clearpage` before document 08 §6, the only in-document barrier in the package |
| The close of document 13 alone on a page, two lines | `\needspace{18\baselineskip}` before the cost table, so table, caption and closing paragraph move as a block |
| The close of document 12 alone on a page, three lines | `\needspace{16\baselineskip}` before §6, so its four subdivisions move together |
| Five pages ending on a section heading | `\section` reservation raised from 3.4 to 9 baselines, because 3.4 is satisfied by space a full-width table cannot use |
| A caption separable from its table | Both table wrappers now close with `\nopagebreak` |
| Fourteen tables of ten rows or more unable to break | Moved to `xltabular` through a new `\mvltable` wrapper, each with a repeating header |
| The abbreviation table carried 30 entries, 13 never used in the body, and 2 blank cells | Rebuilt mechanically from the body: 24 entries, 12 full rows, no blank cell |
| Two bold header cells wider than their columns | Two columns widened |
| Thirty-eight fixed columns wrapping deeper than their neighbors | All 38 retuned against the longest token in each |

Six paragraphs were tightened by a clause each. In every case the instrument was
the first one in the fix hierarchy, a sentence, and never a skip: no `\vspace`
was added to the body anywhere in stage 3.

### Final measured state

| Metric | Value |
|:--|:--|
| Errors, overfull boxes, underfull boxes | 0, 0, 0 |
| Undefined citations, undefined references | 0, 0 |
| Bibliography entries, all cited | 76 of 76 |
| Pages | 67, of which 3 are contents |
| Pages under twelve body lines | 0 |
| Pages ending on a heading | 0 |
| Tables, all at the body measure | 56 |
| Fixed-width columns carrying the ragged prefix | 130 of 130 |
| Dialect word list hits | 0 of 37 |
| Em dashes, en dashes, double hyphens, triple hyphens | 0 |
| Literal `SS` where the section symbol belongs | 0 |
| Raster images anywhere in the subtree | 0 |
| Source characters, `main.tex` plus `sections/` | 175,256 |

### Arithmetic checked rather than asserted

| Check | Result |
|:--|:--|
| Eleven full-time equivalent fractions | 0.20 + 0.10 + 0.10 + 0.40 + 1.00 + 0.45 + 0.20 + 0.55 + 0.40 + 0.30 + 0.25 = 3.95 |
| Eleven charged salaries | $521,000 |
| Six budget lines | $521,000 + $96,000 + $38,000 + $21,000 + $14,000 + $10,000 = $700,000 |
| Five years | $3,500,000 |
| Six stall classes | 22 + 12 + 4 + 3 + 3 + 2 = 46 |
| Eight robot types | 2 + 2 + 1 + 2 + 2 + 1 + 2 + 2 = 14 |
| Three escalation levels at up to six | 18, the stated ceiling |
| Five cohorts of five sites | 5 x $17,500,000 = $87,500,000 |

### Two instructions I could not follow literally, and what I did instead

**The character budget.** The prompt asks for a similar total character count to
the predecessor. The predecessor's `all_documents.tex` is 150,972 characters;
this package's `main.tex` plus `sections/` is 175,256, a ratio of 1.16. The
difference is structural rather than verbose: this package carries 56 full-width
tables where the predecessor carries none, and a table row costs more source
characters per printed line than a paragraph does. Printed, the predecessor is a
denser document and this one is 67 pages.

**The root README limit.** The prompt asks for one additional section and limits
additions to two. The README gains one new section, and the existing badge row,
headline entry and repository structure tree are updated in place rather than
duplicated.

### Two things I recorded rather than smoothed over

**A commit-message count.** The stage 1 `references.bib` commit message says 62
entries. The file carries 76. The message was written before the last block of
codified and standards entries was added and was not corrected, because
rewriting a pushed commit message would rewrite history on a branch the author
is watching. Every README, badge and table in the package states 76, which is
the correct figure.

**One British spelling that stays.** The abbreviation table expands ICH as
International Council for Harmonisation, and the bibliography carries the title
of the author's own ICH E6(R3) adaptation with the guideline's own spelling.
Both are registered names of the bodies and documents concerned, not authored
prose, and changing them would make the citation wrong. Every word of authored
prose in the package is American English, and the dialect audit over a
thirty-seven word list returns zero.

### Deviations from the commit ledger

Rule 6 asks for one commit per section file. Stage 3 landed sections 01 through
04 in a single commit first, and each of those four then received its own
follow-up commit carrying a further improvement, so every section file in every
stage has at least one commit of its own. The total is 66 stage commits across
the three stages against a floor of 30, plus bootstrap, repository and release
commits.

### What is not claimed

Nothing in this package is enacted, filed, or agreed. SB 1188, AB 3162, SB 964
and H. R. 10412 are independent drafts and no bill by those numbers is before
any legislature. The La Jolla site is not leased, permitted, or built, and the
address is withheld until a lease is executed. No institutional review board has
reviewed anything here. No agreement of any kind exists with UC San Diego, with
Moores Cancer Center, with a drug developer, or with a robotic surgery vendor.
Daraxonrasib is investigational and already in Phase 3 evaluation and is nowhere
described as first in human; the supportable novelty claim concerns the
integrated surgical and advisory workflow. The ten coworkers are roles, not
hires. The favorable federal funder responses, the three presidential
recognition letters, and the industry responsiveness are facts about
correspondence and are stated as such, not as awards or agreements.
