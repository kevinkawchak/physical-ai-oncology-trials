## prompt-2-set-a-surgical

**Stage.** PART I, Stage 2 of 5. **Output.**
`applications/app-01-nih-pioneer-award/` through
`applications/app-05-nih-sbir-seed/`.

### Objective

Write the five surgical-perspective application file sets. Each is an
independent-scientist application in Kevin Kawchak's name, dated **August 3,
2026**, stating the intent to partner at **UC San Diego Moores Cancer Center**,
and describing the hybrid operation that carries both surgical and medical
oncology arms.

### What "surgical perspective" means here

The lead argument is the operation: an eight-arm robotic pancreaticoduodenectomy
with an on-premises LLM confined to advisory output, a human surgeon approving
every motion, and bounded stopping. Daraxonrasib is present in every one of the
five, but it enters as the perioperative arm that the operation is timed
around, not as the headline. Conversion, anastomotic integrity, operative time,
blood loss, R0 margin, and 90-day morbidity carry the endpoint table.

### Required content per application (each at most five compiled pages)

1. **Cover page** in the variant assigned in Stage 1, carrying the title, the
   applicant block, the August 3, 2026 date, San Diego, and the independence
   disclaimer. No DOI for the application itself (Part I carries no DOIs).
2. **Section 1, the independent-scientist case.** Open on the report's own
   position: the U.S. research system should prioritize the individual scientist
   over legacy institutions, and the roughly **$200 billion** annual federal R&D
   portfolio should be realigned toward that end. State plainly that the
   applicant is one person with a repository, and that the work already produced
   is what a legacy institution's program office would have produced with a
   larger team.
3. **Section 2, the recipient-specific fit.** Quote the recipient's own
   mechanism as the report describes it, and answer it directly.
4. **Section 3, evidence.** At least two tables of author-source quantitative
   data: the daraxonrasib chronology and simulation numbers, and the prior-work
   ledger with DOIs.
5. **Section 4, the operation and its governance.** The surgical description,
   the advisory boundary, and the stop authority.
6. **Section 5, budget, milestones, and the partner site.** A costed table and
   the Moores Cancer Center partnership sequence.
7. **Back matter.** References with clickable DOIs, and a short statement of
   what is and is not being claimed.
8. **Figures.** At most five per application, chosen by purpose, not by equal
   quota. Use the type that fits: mermaid-type for a decision or a schedule,
   plantuml-type for actors and states, d2-type for nested structure or a grid,
   diagrams-python-type for system architecture, graphviz-type for dependency
   or fault structure.

### Email `.txt` per application

From `kevink@chemicalqdevice.com`. Four blocks, in order: recipient address or
addresses; subject; body; closing on four separate lines, `Sincerely,` then
`CEO Kevin Kawchak` then `ChemicalQDevice` then `July 10th, 2026`, with no
labels. The body names the compiled attachment, and a numbered list tells the
author exactly which prior PDFs or external works to attach by hand.

### Verification before commit

- No black fill in any figure.
- Every table exactly `\textwidth`; every fixed-width column ragged-right.
- Every figure followed by `\vspace{-0.7cm}` then its caption.
- Every DOI in the references rendered as text plus a clickable URL.
- No em dashes, no double or triple dashes.
- Page estimate at or under five pages.

### Commits

One commit per application directory group, pushed immediately (Rule 8). The
second-to-last commit of the stage fixes all errors found across the five sets.
