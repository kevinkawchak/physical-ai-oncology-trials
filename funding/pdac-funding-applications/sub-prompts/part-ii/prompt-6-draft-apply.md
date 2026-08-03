## prompt-6-draft-apply

**Stage.** PART II, Stage 6 of 8. **Output.** `funding/pdac-funding-applications/draft-apply/`.

### Objective

Build the first complete paper skeleton: `main.tex`, `applystyle.sty`,
`references.bib`, twelve `sections/*.tex`, and an Overleaf zip. Every section
carries **bracketed drafting instructions** that name the exact repository files
and directories the later stages must process, so `full-apply` has no ambiguity
about where its content comes from.

### Paper identity

- Title: *10 Funding Applications: A Phase 1, First-in-Human, PDAC Clinical
  Trial Protocol of a LLM-Directed Robotic Whipple with Daraxonrasib
  (RMC-6236)*.
- Draft 1.0, repository v4.4.0, San Diego, August 3, 2026.
- DOI stays in the placeholder form `10.5281/zenodo.xxxxxxxx`, hyperlinked to
  `https://doi.org/10.5281/zenodo.xxxxxxxx` (Rule 12).
- Cover page varies in appearance from the `RFA-RM-27-001-v2` theme: a
  left-accent masthead with a ten-cell application strip, not that template's
  centred form-field block.
- Target length: approximately one quarter of the `patient-robot-advocacy`
  character count, that is roughly 75,000 characters of body text.

### Section list (one `.tex` per section, one commit each, Rule 6)

| File | Section |
|:--|:--|
| `sec-00-front.tex` | Abstract, how to read this paper, figure and table inventory |
| `sec-01-golden-age-mandate.tex` | The $200 billion realignment and the individual scientist |
| `sec-02-ten-applications.tex` | The ten recipients, their mechanisms, and their asks |
| `sec-03-surgical-set.tex` | Set A, applications 01 to 05 |
| `sec-04-medical-oncology-set.tex` | Set B, applications 06 to 10 |
| `sec-05-trial-evidence.tex` | Daraxonrasib chronology, simulations, RASolute 302 |
| `sec-06-physical-ai-governance.tex` | The advisory boundary, stop authority, verification |
| `sec-07-moores-partnership.tex` | UC San Diego Moores Cancer Center |
| `sec-08-budget-and-leverage.tex` | Budget, cost share, and the leverage argument |
| `sec-09-build-method.tex` | The eight-stage build and the five diagram vocabularies |
| `sec-10-risks-and-limits.tex` | What is not claimed, and what would falsify the case |
| `sec-11-references-backmatter.tex` | References, abbreviations, availability, back matter |

### Draft-stage requirements

1. A clickable table of contents, back matter, keywords, and an abbreviation
   table.
2. `\draftinstr{...}` markers naming exact paths, for example
   `funding/science-golden-age/chunk-03`, `funding/RFA-RM-27-001-v2/`,
   `funding/supplementary/source-files/`.
3. All twenty figures placed as sized placeholders carrying their final figure
   number, type tag, and caption, so figure numbering never moves again.
4. The full `applystyle.sty`, complete, with no black fill token.

### Commits

Twelve section commits, plus `main.tex`, `applystyle.sty`, `references.bib`,
README, error-fix, and zip: at least sixteen, comfortably over the ten-commit
floor. Push each immediately.
