# final-capital - Stage 8, the senior-author pass

[![Stage](https://img.shields.io/badge/Stage-8%20of%208-00417A.svg)](../sub-prompts/stage-8-final-capital)
[![Figures](https://img.shields.io/badge/Figures-20%20polished-3C7DB2.svg)](sections)
[![Tables](https://img.shields.io/badge/Tables-21-6C757D.svg)](sections)
[![Pages](https://img.shields.io/badge/Pages-44-6C757D.svg)](.)
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
| `main.pdf` | The compiled paper, 44 pages |
| `final-capital-LaTeX.zip` | Overleaf bundle of everything above |

`main.pdf` and `final-capital-LaTeX.zip` are always written in the same pass
from the same sources: the zip is rebuilt from `main.tex`, `capstyle.sty`,
`references.bib` and `sections/`, and the PDF is compiled from that same set, so
neither can be newer than the other. The zip carries sources only, and Overleaf
runs `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` against them. `unsrturl.bst`
comes from the `urlbst` package in TeX Live and needs no upload.

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
| Every float centred in x | frame rectangles and caption blocks read back out of the compiled PDF against the 306 pt page centre | all 20 frames at 306.00 pt exactly, all 41 captions within 0.53 pt |
| Every caption numbered and at most three lines | parse of `\figcaption` and `\tabcap` arguments | 41 captions, all 3 lines, all numbered |
| Every float referred to in the running text | parse of the section sources with float bodies, captions and comments removed | 20 of 20 figures, 21 of 21 tables |
| Every DOI and URL clickable in the references | `unsrturl` plus `\UrlFont` in the accent colour | 20 DOIs, 17 URLs, 0 entries with neither |
| Pages | | 44 |

## The `\clearpage` discipline

A barrier is placed only where the next section opens with a figure or a
full-width table, so no float drifts out of the section that discusses it.
Sections 8 and 9 open with prose and take no barrier before them, because one
there would leave the preceding page more than a third empty for no gain. That
single change merged two short pages and saved one.

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
| 04 DOIs and URLs shown and clickable | `\bibliographystyle{unsrturl}`, because `unsrt.bst` reads neither the `doi` nor the `url` field and silently dropped both; 20 DOIs and 17 URLs now print and link, marked in the accent colour by `\UrlFont` |
| 05 URLs edited until all references work | Every URL in `references.bib` resolves to a document, never to a search page |
| 06 Table and figure references checked in the body | All 20 figures and all 21 tables are referred to by number at least once in the running text, checked by parsing the sources with float bodies, captions and comments removed |
| 08 Single-word lines fixed | `\parfillskip=0pt plus 0.75\textwidth`, `\finalhyphendemerits=10000`, and a manual sweep; four sentences were re-cut in §9 and §10 |
| 09 Table columns fixed | The stage 7 column-width method re-run against the compiled widths; four columns widened where a bold header proved wider than any body cell |
| 11 Figure X shifts | `\DiagramXShift` and its `adjustwidth` carrier are kept, documented and set to 0 mm at the foot of `capstyle.sty`, and no figure uses them. The nudge on Figure 13 was compensating for a defect in `capstyle.sty`, not for the diagram: every frame and caption sat 13.1 pt right of centre because the centring idiom ended its paragraph with an `\hfil`, which TeX deletes. Both carriers now close with `\null`. At 0 mm the wrapper still moved Figure 13 1.86 pt left, `adjustwidth` being a list environment with its own margin handling, so it was unwrapped; all twenty frames now measure 306.00 pt |
| 13 Arrow overlaps fixed | The third overlap sweep, per figure, recorded in `../full-capital/README.md` and re-checked here after every stage 8 edit |
| 14, 15 Headings, boxes and gate labels fixed | Every in-figure heading is raised clear of the first row; both fault-tree gate labels sit beneath their glyph at the macro's fixed 0.22 cm offset, and Figure 15 now carries a legend saying which shape is a single point of failure |

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
| 10 | The gating milestone named on each firewall crossing |
| 11 | The struck-cable glyph on both prohibited states, so the prohibition survives grayscale |
| 12 | A left margin day column, so the 30-day obligation reads off the figure |
| 13 | Unwrapped from `adjustwidth`, which was costing it 1.86 pt of centring |
| 14 | Branch titles name their owner: the company, the site, an independent monitor |
| 15 | An AND and OR gate legend stating which shape is a single point of failure |
| 16 | A three-swatch tier legend |
| 17 | The four clusters numbered rank 1 to rank 4 |
| 18 | Each boundary rule carries a direction mark naming what may cross it |
| 19 | An eleven-day span marker across the contact column |
| 20 | A legend keyed to the solid and dashed cluster borders |

## Figure numbering follows the order a reader meets them

Four figures were renumbered at this stage. The tier stack and the firewall in
§4, and the milestone activity and the fault tree in §5, had been numbered
against the order they were drafted in rather than the order they are printed
in, so the compiled paper put Figure 11 a page ahead of Figure 10 and Figure 15
a page ahead of Figure 14. Now that each caption prints its own number, that is
visible on the page, so the four were swapped.

| Was | Is | Figure | Platform | Directory |
|:--|:--|:--|:--|:--|
| 11 | 10 | Three capital tiers and the two gaps between them | D2 | [`../d2`](../d2) |
| 10 | 11 | The part 54 capital firewall as five states with guards | PlantUML | [`../plantuml`](../plantuml) |
| 15 | 14 | Evidence production and audit, running concurrently | PlantUML | [`../plantuml`](../plantuml) |
| 14 | 15 | What has to fail, and in what combination, to stop | Graphviz | [`../graphviz`](../graphviz) |

The two indexes in §0, the stage table in §10, the four specification files and
their filenames, the five directory READMEs, the three stage sub-prompts and
every cross-reference in the body were moved with them. Stages 6 and 7 in
[`../draft-capital`](../draft-capital) and [`../full-capital`](../full-capital)
are the record of what those stages produced and keep the numbering they were
built with.

## Caption balance

All forty-one captions are exactly three lines, all centred, and each opens with
its own number: `Figure N.` or `Table N.`. Each is broken by hand to the
narrowest spread its own word boundaries allow, found by exhaustive search over
every two-cut split of the caption. A caption three lines long is as wide as its
own text, so line width runs 61 to 74 characters across the figures and 59 to 80
across the tables; the numbers that resist balancing are the long unbreakable
ones, `$1,606,000` and `1,396,000`.

| Spread | Figures | Tables |
|:--|:--|:--|
| 0 to 2 characters | 11, 13, 17 | 2, 13, 14, 16, 17, 20 |
| 3 to 4 characters | 1, 6, 7, 12, 16, 19, 20 | 3, 5, 6, 7, 11, 12, 18, 21 |
| 5 to 6 characters | 2, 4, 5, 9, 14, 15 | 1, 4, 10, 19 |
| 7 characters or more | 3, 8, 10, 18 | 8, 9, 15 |

## Rule 5 source map

| This stage used | From | For |
|:--|:--|:--|
| `full-capital/` | this directory tree | The entire text and all twenty figures, carried forward and improved |
| `final-apply/publication/useredits.md` | `../../pdac-funding-applications` | The ten-item correction list applied above |
| `final-apply/main.tex` | `../../pdac-funding-applications` | The `\clearpage` discipline and the float-barrier convention |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The five TikZ vocabularies `capstyle.sty` inherits |
| `RFA-RM-27-001-v2/` | `../..` | The cover theme this paper's `\capmast` and `\capledger` vary from |
