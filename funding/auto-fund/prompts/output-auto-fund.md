## output-auto-fund

I explored the repository before writing anything: the `funding/` tree, the
capitalization plan that the prompt names as the general template, the ten
application file sets and their `.txt` email format, the La Jolla move-in
package, and the partner research that holds every named address this build
uses. I also checked the CI workflow first, because the prompt asks explicitly
that the pull request pass its checks: `ci.yml` runs `ruff check`, `ruff format
--check` and `yamllint` across three Python versions, so the binding constraint
was that this build must add no Python file and no YAML file. It adds neither.

**Toolchain.** No LaTeX was installed in the container and CTAN was blocked by
the network policy, but the Debian archives were reachable, so I installed TeX
Live from `apt` (`texlive-latex-base`, `-recommended`, `-extra`,
`-fonts-recommended`, `-pictures`, `-bibtex-extra`, `-binaries`) and verified
`unsrturl.bst`, `titlesec`, `adjustbox`, `xltabular`, `changepage`, `ragged2e`,
`needspace`, `colortbl` and `enumitem` were all present before writing a line of
`.tex`. Every packet in this directory was compiled and measured before its
commit, not after.

### Decisions taken, and why

**Five days, five themes, no repetition.** The prompt asks for five business days
of separate and comprehensive instructions. The failure mode is five variations
on one letter. So each day has a distinct addressee class and a distinct
decision: federal re-contacts, private capital instruments, clinical sites and
foundations, staging, and execution. Day 5's cadence brief closes the loop by
showing that the five themes map onto a repeatable weekday frame, which is the
argument that the sequence was structural rather than arbitrary.

**September 7, 2026 is Labor Day.** The prompt lists `07Sep26` as one of the five
business days. Federal offices are closed and the New York Stock Exchange and
Nasdaq are shut. Substituting a different date would have quietly changed the
schedule the prompt sets; sending letters into a closed day would have been
professionally wrong. So the day was built as a staging day whose entire output
carries a `HOLD FOR RELEASE` line naming the next open session, and day 5 is the
release. That is both the correct handling and the reason the block has a
preparation day at all, which the weekly cadence then generalizes.

**Addresses are filled where the repository records them and bracketed where it
does not.** Thirteen addresses across the block are carried from
`funding/potential-partners`: the UC San Diego clinical, escalation, contracting
and clinical trial support services routes, the five Scripps research contacts,
and the four developer external-research addresses. Every other recipient is left
in brackets with the exact page to obtain the address from. Inventing an address
would be worse than a bracket: a wrong address either bounces or reaches a
stranger, and each letter's `BEFORE SENDING` block names the repository file its
addresses came from so the author can verify them on the day of sending.

**Document counts per day were chosen against the day's work, not held constant.**
Day 1 carries five letters and three briefs because it re-contacts five
mechanisms and needs a plain-text evidence page. Day 3 carries two briefs rather
than three because its technical content is a question list and a positioning
statement, and a third would restate one of them. Day 5 carries one form pack
rather than two because it submits rather than registers. Totals: 24 letters, 13
briefs, 9 form packs, 5 capital instruction sets, 15 figure specifications.

**The `-0.60cm` invariant.** The prompt fixes caption spacing at `-0.60cm`. The
parent `capstyle.sty` uses `-0.65cm`, and `new-trial-system/trialstyle.sty`
already carried the `-0.60cm` variant with two-line captions, which is exactly
what this prompt asks for. `fundstyle.sty` therefore takes the parent's
mechanics, the sibling's invariant, and adds three things: a cover that leads
with the decision, a palette confined to one block so five daily copies differ
only in six hex values, and the deletion of the parent's `\umlactor` stick-figure
macro so that no basic human stick figure can be drawn even by accident.

**Diagram platform balance.** Fifteen figures, three per day, three per platform,
and no platform twice in one day. The split follows purpose rather than quota: a
flowchart where a state changes, a record table where fields repeat, a grid where
alternatives are compared, a state machine where transitions carry conditions, a
fault tree where the question is answered by combination, and a glyph topology
where functions have locations. Where two platforms could have served, the one
whose native construct needed no invention was chosen.

### Defects found, with their measured sizes

Every one was found by compiling and measuring rather than by reading.

1. **Two fatal TikZ node-name failures.** A `\foreach` over negative decimal
   coordinates generated names such as `st-2.76`, which TikZ reads as node
   `st-2` at anchor `76`, and the compile entered an unrecoverable error loop
   that had to be killed. The same class appeared twice, in Figure 1 and Figure
   7. Fixed by naming every node with plain letters and digits, and recorded as
   an invariant in every `diagrams/README.md`. The compile script was hardened
   with `-halt-on-error` so this class fails fast rather than hanging.
2. **An unbraced `fit` value.** `fit=(ct)(-1.68,0.30)(8.60,-4.30)` was split into
   separate keys by the TikZ key parser, because the value carries commas. Fixed
   by bracing every `fit` value in the build.
3. **`\pnote` used as a style.** It is a macro taking three arguments, not a
   TikZ style, and using it inside a node's option list halted the day 5 compile.
4. **Markdown emphasis in three `.tex` files.** `**day**` renders as literal
   asterisks in LaTeX. Converted to `\textbf` in three files.
5. **One overfull hbox of 1.48 pt**, a bold `Comparator` header wider than its
   2.0 cm column in day 1's evidence table. Column widths recut.
6. **Three bibliography entries with no resolvable target at all**: the White
   House report, its annexed budget memorandum, and one repository source set.
   All three now carry a `url`. All 56 entries in every packet resolve.
7. **One table caption 23 characters out of balance** between its two lines, 69
   against 46. Rebalanced to 66 and 69. The mean spread across all 40 captions is
   2.9 characters and none exceeds 8.
8. **Fourteen short pages** across the five packets, ranging from three to
   fourteen body lines. Each was fixed at its cause: a nine-line contents page
   removed by setting `tocdepth` to sections only, and thirteen section-closing
   pages brought above twenty lines by adding substantive text or by tightening
   prose. **No `\vspace` was added to the body of any packet.**
9. **One overfull vbox of 15.72 pt** in day 1's reference list, appearing only
   after the three new bibliography urls lengthened it. A page barrier before the
   list fixed it in that packet; the same barrier was measured in the other four
   and made each of them worse, so it is used in one packet only and the reason
   is recorded in the source.
10. **Nine letters and one packet README quoted page counts** that no longer
    matched the compiled documents after the page-shape work. All aligned to the
    measured values.

### Audits run to zero

Across every text source in the directory, excluding the verbatim prompt file:
em dashes, en dashes, smart quotes, ellipses, non-breaking spaces, `SS` in place
of the section symbol, `tabularx` not set to `\textwidth`, fixed columns without
a prepended `\raggedright\arraybackslash`, float spacing other than `-0.60cm`,
and captions that are not exactly two lines. 790 relative links were resolved
against the filesystem; the only two that did not resolve pointed at this file,
which is written in the final commit.

### Instructions that needed interpretation

**British spellings.** Five survive and all five are inside file paths: the
capitalization plan contains `sec-03-gate-and-programme.tex`, and a Rule 5 source
map that cites a real path cannot rename it. Every word of authored prose is
American English.

**"Avoid excessive use of dates."** The date appears in each day's directory
name, once on each packet cover, and in the two or three places where a specific
dated fact is the point, principally the August 26, 2026 approval. Nothing in the
body of any letter or brief is keyed to the day it was written, so the author can
act on any day outside its own date.

**The 425-character summary.** Written to exactly 425 characters with spaces and
verified by count.

### What is not claimed

Nothing in the directory is a submission of record, an agreement, an offering, or
an award. No agreement of any kind exists with any institution, with the agent's
developer, or with any robotic surgery vendor. No offering exists and no
instrument has been selected. No investigational new drug application has been
submitted, no institutional review board has reviewed anything, no patient has
been treated, and no robotic configuration has been specified. Daraxonrasib is
approved in the metastatic setting and is nowhere described as first in human;
the perioperative use this program proposes remains investigational. The $36,330
virtual trial figure is described as projected wherever it appears, never as
estimated. The three presidential recognition letters are stated once, precisely,
as facts about correspondence, and never as awards, agreements, or reviews.

### Final measured state

Five packets, seven sections each, 35 section files. 15 TikZ figures, 25 tables,
40 two-line captions. 24 letters, 13 briefs, 9 form packs, 5 capital instruction
sets, 15 figure specifications, 49 READMEs. Five compiled PDFs at 13, 13, 12, 13
and 14 pages, each at zero errors, zero overfull boxes, zero underfull boxes,
zero undefined citations and zero undefined references, with no page under twenty
lines. Five Overleaf zips, eleven files each. Zero PNG or JPG files, and zero
Python files, so the repository's three lint-and-format checks are unaffected.
