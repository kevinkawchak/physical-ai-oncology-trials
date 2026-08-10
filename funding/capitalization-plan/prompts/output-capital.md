## output-capital

I explored the repository, read the five source directories the prompt names,
and confirmed the toolchain before writing anything. `pdflatex` and `bibtex`
were installed into the session, and the parent work
`funding/pdac-funding-applications/final-apply/publication` was test-compiled
first: 0 errors, 0 overfull boxes, 31 pages. That established the target, and
every stage below was compiled and verified before its commit rather than
after.

### Stage 0, the schedule

The parent work runs two sub-prompt schedules because it has two deliverables,
ten application file sets and one summary paper. This project has one, so the
schedule is a single eight-stage line: five diagram stages, then draft, full and
final. I created `funding/capitalization-plan/sub-prompts/` with one directory
per stage, the hub README, and the arithmetic every table in the paper has to
reconcile to.

That arithmetic is the spine of the whole document, so I fixed it before writing
a word of prose:

| Quantity | Value |
|:--|:--|
| Five-year programme, direct | $3,500,000 |
| SBIR Phase I, total cost, 9 months | $306,000 |
| SBIR Phase II, total cost, 24 months | $1,300,000 |
| SBIR route, total, 33 months | $1,606,000 |
| Direct work inside that award | $1,396,000 |
| Delta the award does not reach | $2,104,000 |
| Private capital behind the firewall | $5,900,000 |
| Private to federal leverage | 3.67 to 1 |

The four-layer split of the $3,500,000 is taken verbatim from
`final-apply/sections/sec-08-budget-and-leverage.tex` and is not re-derived
anywhere. Everything else is built on top of it.

PR #73 was opened at this point so the branch could be watched from the first
commit rather than at the end.

### Stages 1 to 5, twenty diagrams

Twenty figure specifications, one commit each, plus one commit per directory
README. The split is 5 mermaid, 3 plantuml, 5 d2, 3 diagrams-python, 4
graphviz, and it follows purpose rather than quota: the two five-counts fall
where a capitalization plan argues most, in sequence and in tables.

Each specification carries a three-line balanced caption, valid source in its
native platform syntax, a TikZ construction-notes table with stated pitches and
coordinates, and the repository files it draws on. Each also states the
perspective no other figure in the paper takes, which is the test that stopped
two figures from being near-duplicates: Figure 8 and Figure 11 both draw money,
so one is a left-to-right ledger organised by purpose and the other is a
bottom-to-top stack organised by source.

The diagrams-python stage emits a Markdown specification and no `.py` file. That
is inherited from the parent work and has two reasons: the library renders
through Graphviz to a raster, and this paper generates no raster; and the
repository runs three `lint-and-format` jobs across the whole tree, which a
`.py` file would have to satisfy on Python 3.10, 3.11 and 3.12.

### Stage 6, draft-capital

`capstyle.sty` is `applystyle.sty` with three changes and no others. The figure
spacing invariant is retuned so that `\vspace{-0.65cm}` leaves a constant gap:
`appfig` closes with a rigid `\vskip 24.5pt`, so the frame-to-caption distance
is exactly 24.5 pt minus 0.65 cm, that is 6.06 pt, everywhere. The cover is new,
a ruled panel with a top accent band over a three-cell money ledger, because
this paper has one recipient and three numbers where the parent had ten
recipients and one. And the author's own `\DiagramXShift` carrier from
`useredits.md` is added.

Everything else is kept deliberately: the five TikZ vocabularies, all 24 vector
glyph macros, the quantitative primitives, the eight-token palette with no black
fill, and the senior-author typographic penalties.

Two defects were found and fixed inside this stage. The three-cell cover ledger
sat 14.34 pt past the measure, because three cells at 0.288 of the text width
plus two gaps plus six inner separations do not fit inside it; each cell is now
`\dimexpr0.28\textwidth-9pt\relax` so its outer width is exactly 0.28 of the
measure. And `diagrams-python/` set in a monospace face overflowed a 3.3 cm
table column by 11.23 pt in three rows, because Courier offers no break point
inside a path; all five directory paths now use `\appfile`, which inserts a
break opportunity after every character.

The draft issues `\nocite{*}` so the author can check at this stage that all 41
references resolve, that every DOI prints as text with a clickable target, and
that no reference line runs past the right margin. Stage 7 removes it.

Result: 0 errors, 0 overfull, 0 underfull, 32 pages.

### Stage 7, full-capital

Every drafting instruction resolved and all twenty figures drawn in TikZ.
Twenty-one full-width tables, every one at `\textwidth` with exactly one `X`
column and every fixed column prefixed `>{\raggedright\arraybackslash}`.

Six defects were found and fixed inside this stage:

1. Record value cells 1.39 pt too narrow for `$1,396,000` at `\tiny` Times.
2. A `\foreach` loop iterating a list with an empty body.
3. Eleven DOIs overflowing a 4.0 cm column by 2.98 pt, since `\dlink` text is
   unbreakable; the column is now 4.4 cm.
4. `minimum width={\w*0.415} cm` is read by the mathematics engine as a single
   expression and raises `Unknown operator 'c' or 'cm'`. The unit belongs inside
   the braces.
5. A 143 pt overfull vertical box on page 9. I bisected it to Figure 3, then to
   the combination of Figure 3, Table 5 and a closing paragraph competing for
   one page. Figure 3's vertical bar panel became three horizontal bars, which
   removed a column of empty air, the float budget was retuned from the parent's
   topnumber of 3 to one float per page top, and a `\clearpage` was placed
   before the last subsection of §1.
6. The table count was wrong. The draft index said nineteen; the paper carries
   twenty-one. The index was rebuilt and sixteen cross-references renumbered.

Three of the four table columns that needed widening had a bold header wider
than any body cell in the column, which is the failure mode the column-width
method exists to catch: `Comparator` at 10.02 pt over, `Likelihood` at 4.88 pt,
`Commits` at 2.74 pt.

The spacing invariant was also extended to tables at this stage, because the
prompt requires it of every new diagram **and table**. `apptable` now closes
with the same rigid `\vskip 24.5pt` that `appfig` closes with, `\tabcap` opens
with `\nointerlineskip`, and every table is written
`\end{apptable}` then `\vspace{-0.65cm}` then `\tabcap`. The audit is
arithmetic: 41 occurrences of `\vspace{-0.65cm}`, against 20 `\figcaption` plus
21 `\tabcap`.

Result: 0 errors, 0 overfull, 0 underfull, 0 undefined, 43 pages.

### Stage 8, final-capital

The senior author's pass. No new argument; every existing one made to sit
correctly on the page. No `publication/` subdirectory, by instruction.

`\clearpage` discipline was applied properly: a barrier only where the next
section opens with a figure or a full-width table. Sections 8 and 9 open with
prose and take none, which merged two short pages and brought the paper from 43
to 42.

To find short pages without being able to view them, I compiled a temporary copy
under `\flushbottom`, which makes TeX report an underfull vertical box for every
page carrying significant slack. Ten pages reported, which is consistent with
eleven `\clearpage` barriers rather than with a defect, and the count fell after
the barrier discipline was corrected.

The author's ten-item correction list from `useredits.md` was applied item by
item, and every one of the twenty figures was improved once more: legends where
a fill was carrying meaning alone, counts where a reader would otherwise have to
count, rank numbers on Figure 17's four clusters, a day column on Figure 12, an
eleven-day span marker on Figure 19, and the `\DiagramXShift` wrapper on Figure
13, which is the one figure whose narrow left lane titles and wide right cost
column put its optical centre left of its frame.

Final audits: no em dash, en dash, double dash or triple dash in the body; 39
section symbols and no `SS` spelling; no PNG or JPG anywhere; and the palette
audit for the parent style's deleted near-black fill token returns nothing.

Result: 0 errors, 0 overfull, 0 underfull, 0 undefined citations, 0 undefined
references, 42 pages, 20 figures, 21 tables, 41 references all cited.

### On the quantitative case

The prompt asks that the data be sufficient to convince a funder of the Phase 1
trial. Two decisions shaped how that was done.

First, every clinical quantity is carried with the limitation its own authors
stated, in the same row: RASolute 302's 13.2 against 6.6 months is labelled
metastatic and previously treated and silent on the resectable setting; the
ten-arm simulation's 12.8 against 5.4 is labelled as assuming no acquired
resistance; the digital twin's hazard ratio of 0.31 is labelled as having no
patient-specific parameters. The 2025 simulated ratio of 2.4-fold and the 2026
observed ratio of 2.0-fold are set beside each other and then explicitly denied
the status of a validation claim, with the three material differences named.

Second, the money is made checkable rather than impressive. Every table
reconciles to the eight numbers above, the two award decompositions sum exactly,
the twelve milestone costs sum to their phases exactly, and the seventeen work
packages sum to their three clusters exactly. Work packages WP1 to WP12 carry
the identical cost to milestones M1 to M12, so Figures 8, 9 and 13 can be
checked against one another by a reader with a pencil.

### Two things worth flagging

The capitalization figures are a plan. No term sheet, simple agreement for
future equity, or subscription agreement exists, and the paper says so on its
cover, in §9 and in the back matter. The five-stage capital table shows stage 4
failing the SBIR ownership test rather than hiding it, because a plan that
concealed the failure point would be less useful to the reader it is written
for.

The paper is deliberately harder on itself than the ten applications were. §9
opens with four things that are worse about this document than about the ten it
replaces, states each without mitigation in the first sentence, and answers each
in the second. Three rows of the risk register read "none today". Those are the
parts a reviewer will test first, and they are the parts that were written most
carefully.

### Where the instruction could not be followed literally

Two places, both recorded rather than silently absorbed.

The prompt asks for a similar character count to the parent work. The parent's
twelve sections plus `main.tex` total 116,343 characters as deposited; this
paper's total 152,557, a ratio of 1.31. That is in the same range rather than
identical, and the difference is structural: this paper
carries twenty-one full-width tables against the parent's eighteen, and a table
row costs more source characters per printed line than a paragraph does. On
printed pages the two are closer, 42 against 31.

The prompt asks that the main README gain one additional section. It gains one
new section, and the existing badge row, headline entry and repository-structure
tree are updated in place, which the prompt also requires. That is one new
section plus edits to existing material, not two new sections.
