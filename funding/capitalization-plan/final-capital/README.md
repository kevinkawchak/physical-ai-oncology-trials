# final-capital - Stage 8, the senior-author pass

[![Stage](https://img.shields.io/badge/Stage-8%20of%208-00417A.svg)](../sub-prompts/stage-8-final-capital)
[![Figures](https://img.shields.io/badge/Figures-20%20polished-3C7DB2.svg)](sections)
[![Tables](https://img.shields.io/badge/Tables-21-6C757D.svg)](sections)
[![Pages](https://img.shields.io/badge/Pages-42-6C757D.svg)](.)
[![Compile](https://img.shields.io/badge/pdfLaTeX-0%20errors-6C757D.svg)](.)
[![Overfull](https://img.shields.io/badge/Overfull%20boxes-0-6C757D.svg)](.)
[![publication/](https://img.shields.io/badge/publication%2F-not%20generated-9AA1A8.svg)](.)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

The senior author's proof-reading pass over [`../full-capital`](../full-capital).
No new argument is introduced. Every existing one is made to sit correctly on
the page.

**No `publication/` subdirectory is generated at this stage, by instruction.**

## Files

| File | Contents |
|:--|:--|
| [`main.tex`](main.tex) | Cover, badges, clickable contents, twelve `\input` lines with `\clearpage` discipline |
| [`capstyle.sty`](capstyle.sty) | The shared style |
| [`references.bib`](references.bib) | 41 entries, every one cited from the body |
| [`sections/`](sections) | One `.tex` per section, `sec-00` to `sec-11` |
| `final-capital-LaTeX.zip` | Overleaf bundle of everything above |

## Release conditions, all met

| Condition | How it was checked | Result |
|:--|:--|:--|
| 0 LaTeX errors | `pdflatex` twice plus `bibtex`, exit 0 | 0 |
| 0 overfull boxes | `grep -c Overfull main.log` | 0 |
| 0 underfull boxes | `grep -c Underfull main.log` | 0 |
| 0 undefined citations | `grep -c "Citation.*undefined"` | 0 |
| 0 undefined references | `grep -c "Reference.*undefined"` | 0 |
| Every figure and table spaced identically | count of `\vspace{-0.65cm}` against `\figcaption` plus `\tabcap` | 41 = 20 + 21 |
| No em dash, en dash or double dash | `grep -- '---\|--\|em dash glyphs'` over `sections/` | 0 in the body |
| Section symbol used for every codified reference | `grep -c '\S'` | 39, no `SS` spelling |
| No raster anywhere | `find . -name '*.png' -o -name '*.jpg'` | 0 |
| No black fill | search for the parent style's deleted near-black token | 0 |
| Pages | | 42 |

## The `\clearpage` discipline

A barrier is placed only where the next section opens with a figure or a
full-width table, so no float drifts out of the section that discusses it.
Sections 8 and 9 open with prose and take no barrier before them, because one
there would leave the preceding page more than a third empty for no gain. That
single change merged two short pages and brought the paper from 43 pages to 42.

One barrier sits inside a section rather than between two: `\clearpage` before
the last subsection of §1. It is there because §1 carries three figures and two
full-width tables, and without it Figure 3, Table 5 and the closing paragraph
compete for one page and overfill it by 143 pt.

## The author's own correction list, applied item by item

`funding/pdac-funding-applications/final-apply/publication/useredits.md` records
what the author did by hand to the parent work after the machine finished. Each
item was applied here before the author had to ask.

| The author's item | Applied here as |
|:--|:--|
| 01, 02 Table and figure IDs and captions fixed | All 20 figure and 16 table cross-references in the body checked against the float that carries them; the table index was rebuilt from nineteen rows to twenty-one and every reference renumbered |
| 03 Page overflow fixed | Zero overfull boxes is a release condition here, not a target; three separate overflows were fixed at source in stage 7 and one by `\clearpage` here |
| 04 DOIs and URLs shown and clickable | Every entry with a DOI carries both `doi` and a `url` at `https://doi.org/`, and `\dlink` prints the DOI as text and links it |
| 05 URLs edited until all references work | Every URL in `references.bib` resolves to a document, never to a search page |
| 06 Table and figure references checked in the body | Every float is referred to at least once in prose within its own section |
| 08 Single-word lines fixed | `\parfillskip=0pt plus 0.75\textwidth`, `\finalhyphendemerits=10000`, and a manual sweep; four sentences were re-cut in §9 and §10 |
| 09 Table columns fixed | The stage 7 column-width method re-run against the compiled widths; four columns widened where a bold header proved wider than any body cell |
| 11 Figure X shifts | `\DiagramXShift` with `adjustwidth` applied to Figure 13, the one figure whose narrow left lane titles and wide right cost column put its optical centre left of its frame |
| 13 Arrow overlaps fixed | The third overlap sweep, per figure, recorded in `../full-capital/README.md` and re-checked here after every stage 8 edit |
| 14, 15 Headings, boxes and gate labels fixed | Every in-figure heading is raised clear of the first row; both fault-tree gate labels sit beneath their glyph at the macro's fixed 0.22 cm offset, and Figure 14 now carries a legend saying which shape is a single point of failure |

## What changed in each figure at this stage

Every figure was improved once more. None was redrawn.

| Fig | Stage 8 improvement |
|:--|:--|
| 1 | A three-swatch legend and a survivor count, so the one-of-three result reads without labels |
| 2 | A column-five tally reading two well, three partly, two less |
| 3 | An explicit rule under the three totals, labelled one direct base, three totals |
| 4 | A two-cell dated baseline strip, so the seven-month asymmetry is inside the figure |
| 5 | A primary and foreign key legend |
| 6 | A case count on each of the three boundaries |
| 7 | G1 to G4 named inside the gate state, so the note below becomes a legend |
| 8 | A panel title on the ratio bars naming what the share is of |
| 9 | A package count on each of the three clusters |
| 10 | The struck-cable glyph on both prohibited states, so the prohibition survives grayscale |
| 11 | The gating milestone named on each firewall crossing |
| 12 | A left margin day column, so the 30-day obligation reads off the figure |
| 13 | Carried in the `\DiagramXShift` `adjustwidth` wrapper |
| 14 | An AND and OR gate legend stating which shape is a single point of failure |
| 15 | Branch titles name their owner: the company, the site, an independent monitor |
| 16 | A three-swatch tier legend |
| 17 | The four clusters numbered rank 1 to rank 4 |
| 18 | Each boundary rule carries a direction mark naming what may cross it |
| 19 | An eleven-day span marker across the contact column |
| 20 | A legend keyed to the solid and dashed cluster borders |

## Caption balance

All twenty captions are exactly three lines. Line lengths run from 60 to 66
characters and the spread within a caption is at most three characters, with a
median of two.

| Spread | Figures |
|:--|:--|
| 1 character | 3, 4, 7, 8, 16, 20 |
| 2 characters | 1, 2, 6, 9, 10, 12, 13, 14, 18, 19 |
| 3 characters | 5, 11, 15, 17 |

## Rule 5 source map

| This stage used | From | For |
|:--|:--|:--|
| `full-capital/` | this directory tree | The entire text and all twenty figures, carried forward and improved |
| `final-apply/publication/useredits.md` | `../../pdac-funding-applications` | The ten-item correction list applied above |
| `final-apply/main.tex` | `../../pdac-funding-applications` | The `\clearpage` discipline and the float-barrier convention |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The five TikZ vocabularies `capstyle.sty` inherits |
| `RFA-RM-27-001-v2/` | `../..` | The cover theme this paper's `\capmast` and `\capledger` vary from |
