## output-full-bill

# Stage 6 narrative: the full H. R. 9510 Bill v5.0

*Independent research draft. Not an enacted law and not legal advice. All
dollar figures are illustrative unless tied to a cited statute or notice.*

Stage 6 executed every bracketed [DRAFTING INSTRUCTION] in
`auto-bill-02/draft-bill` and produced the **full** Financial Data Amendment in
`auto-bill-02/full-bill`. The scaffold's slot markers are gone: every figure
is rendered in its planned medium, every table carries its financial data, and
the bill stands on its own.

## What was rendered

1. **The operative core.** New section 515D is carried verbatim from Bill v3.0
   (subsections (a)-(j)) and the new subsection **(k) Financial Data Record**
   is appended: the per-run direct-cost record in the part 11 audit trail, its
   attestation, the fiscal-year report to the Secretary, and the two defined
   terms. The ten conforming amendments, clerical amendment, rule of
   construction, and effective date are carried, with the effective-date
   subsection extended by the first-report timing.
2. **The comparative print, widened to twelve sections.** The eleven carried
   Title 21 device sections plus the new reproduction of § 379j (FD&C § 738),
   showing the SEC. 5(c) insertion: a verification record filed alone is not a
   fee-bearing submission.
3. **SEC. 5, the financial core.** Transparency (annual de-identified
   publication in the Open Payments manner; the implementation cost
   statement), part 54 disclosure alignment for verification-service
   arrangements, the user-fee treatment with the MDUFA VI sense of Congress,
   the authorization of appropriations (Table 6; 18.0/12.0/10.0/9.0/9.0 =
   58.0 illustrative millions, every column summing), and the boxed Statutory
   PAYGO budgetary-effects clause.
4. **The five financial appendices.** A: the CBO-genre estimate (BA versus
   outlays at the 60/30/10 spend-out; the statutory user-fee percentage
   table; the scorecard and mandates summary). B: verification economics (the
   Table 2 unit-cost column composed to 4,420 dollars per cleared interaction
   class; the nineteen-fold program-year comparison). C: the financial-data
   transparency standard (completeness, traceability, periodicity,
   comparability), demonstrated by the bill's mechanism and by this build's
   own record. D: the financial research-influence matrix (MDUFA VI timing,
   fee notices, appropriations climate, and the verified 119th Congress
   bills). E: development transparency (the v1.0-v5.0 lineage, the
   nine-milestone schedule, the implementation narrative).

## The three media, as delivered

| Medium | Count | Where |
|:--|:--|:--|
| Tables (full-width; `R` columns for dollars) | 15 | Tbl. 1-6 in SEC. 2-5; Tbl. 7-15 in App. A-E |
| ASCII figures (centered monospace frames) | 6 | Cover; Fig. 2 (SEC. 2), 4 (SEC. 3), 5 (SEC. 4), 7 (SEC. 5), 8 (App. A) |
| Gray-scale Mermaid as TikZ | 6 | Fig. 1 (SEC. 2), 3 (SEC. 3), 6 (SEC. 5), 9 (App. B), 10 (App. C), 11 (App. E) |

## Financial-data consistency checks (the Appendix C tests, applied here)

- Table 1 footprints (12,600 + 21,400 + 6,100 + 8,300) sum to the **48,400**
  carried in Figure 2 and Table 11.
- Table 2 unit costs sum to **3,710**; with records (240) and ESCALATE review
  (470) they compose to the **4,420** of Table 10.
- Table 6 sums by row and column to **58.0**; Table 7 outlays (10.8 + 12.6 +
  11.4 + 9.6 + 9.1 = **53.5**) plus the after-window 4.5 return the 58.0; the
  Figure 8 bars are the same series rounded to whole millions (11/13/11/10/9),
  a correction from the Stage 3 catalog sketch.
- Table 11 columns sum (3,000,000 conventional; 155,000 automated; the
  nineteen-fold claim is 3,000,000/155,000).

## Corrections applied to the draft instructions

- The draft's SEC. 3 renumbering note (which misplaced the carried rule of
  construction) was simplified: subsections (a)-(j) are carried untouched and
  the financial-data record is **appended as (k)** with its own definitions in
  (k)(4), keeping every "515D(k)" cross-reference in the build consistent.
- The Figure 8 outlay bars were re-rounded to match the exact Table 7 series.

## Commit record (milestone M7)

One commit per file, pushed in real time: `main.tex`, `usctitle.sty`
(v5.1.0), `references.bib`, the nine rendered `sections/*.tex`, this
narrative, the re-filed `prompt-full-bill.md`, the directory `README.md`, and
the error-fix-and-bundle commit carrying `full-bill-LaTeX.zip`.
