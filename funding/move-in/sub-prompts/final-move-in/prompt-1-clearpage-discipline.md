# Stage 3, sub-prompt 1 - `\clearpage` discipline and page self-standing

## Goal

No section may open on a stranded line, no page may end with a heading, and no
page may carry a single line orphaned from the paragraph that owns it. This is
the first of the senior-author passes and it is done by reading the compiled
page, not the source.

## The author's method, taken from `final-apply/main.tex`

1. **A barrier only where it earns its place.** In the parent build,
   `\clearpage` is placed between sections whose successor opens with a
   full-width table, so no float drifts out of the section that discusses it.
   Where the next section opens with prose, no barrier is issued, because one
   there would leave the preceding page more than a third empty.
2. **In this paper the rule is stronger for parts and unchanged for sections.**
   Clause J requires that each of the fifteen documents starts on its own page,
   so `\clearpage` precedes every `\docpart` without exception. Within a
   document, a barrier is issued only where the measured page would otherwise
   strand material.
3. **`\FloatBarrier` at every `\section`.** Inherited from the parent style, so
   a table can never migrate past the heading of the section that introduces it.
4. **`\needspace` on every heading.** A `\section` reserves 3.4 baselines, a
   `\subsection` 3.0, a `\subsubsection` 2.6. A heading plus three lines of its
   own text move together or move together to the next page.

## Detection without a page viewer

The compiled PDF is measured, not eyeballed:

- `pdftotext -layout main.pdf -` then count the lines on each page. A page whose
  body carries fewer than eight lines and is not the last page of a document is
  a short page and is investigated.
- A page whose final element is a heading is found by checking whether the last
  non-empty line of page *n* matches a heading pattern in the source.
- A page opening with one line that completes a paragraph begun on the previous
  page, followed by white space and a heading, is an orphan and is fixed by
  adding or removing a sentence in the paragraph above, not by inserting a
  `\vspace`.

## The fix hierarchy

Apply in this order, and stop at the first that works:

1. Add or remove a sentence in the paragraph that spills. This is the senior
   author's normal instrument and it is always preferred, because it leaves the
   typography untouched.
2. Rebalance a table caption onto a different line break.
3. Move a table to the head of the following page with `\clearpage`.
4. Only then adjust vertical space, and only through the named macros in
   `movestyle.sty`, never with a bare `\vspace` in the body.

## Acceptance

- No page ends with a heading.
- No document opens anywhere but at the top of a page.
- No page carries a single stranded line.
- Page count is recorded in `final-move-in/README.md`.

## Commit

Folded into the seventeen section commits of stage 3.
