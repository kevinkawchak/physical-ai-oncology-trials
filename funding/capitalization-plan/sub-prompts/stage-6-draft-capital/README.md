## Stage 6 sub-prompt - draft-capital

[![Stage](https://img.shields.io/badge/Stage-6%20of%208-00417A.svg)](.)
[![Output](https://img.shields.io/badge/Output-..%2Fdraft--capital-3C7DB2.svg)](../../draft-capital)
[![Sections](https://img.shields.io/badge/Sections-12-6C757D.svg)](../../draft-capital/sections)
[![Figure slots](https://img.shields.io/badge/Figure%20slots-20-6C757D.svg)](../../draft-capital)
[![Commits](https://img.shields.io/badge/Commits-16-9AA1A8.svg)](.)

### Instruction

Build the skeleton of the paper in `funding/capitalization-plan/draft-capital/`.
The stage produces a document that already compiles, already carries its final
figure and table numbering, and already knows the exact repository file that
every later stage must read. It does not yet carry the argument.

### Deliverables and commit order

| Commit | File | Contents |
|:--|:--|:--|
| 1 | `main.tex`, `capstyle.sty`, `references.bib`, `README.md` | Cover, badges, clickable contents, twelve `\input` lines |
| 2 | `sections/sec-00-front.tex` | Abstract, executive summary, reader's guide, Tables 1 and 2 |
| 3 | `sections/sec-01-novel-performer-case.tex` | Slots for Figures 1, 2, 3 |
| 4 | `sections/sec-02-entity-and-asset.tex` | Slots for Figures 4, 5, 6 |
| 5 | `sections/sec-03-gate-and-programme.tex` | Slots for Figures 7, 8, 9 |
| 6 | `sections/sec-04-capital-bridge.tex` | Slots for Figures 10, 11, 12 |
| 7 | `sections/sec-05-twelve-milestones.tex` | Slots for Figures 13, 14, 15 |
| 8 | `sections/sec-06-clinical-evidence.tex` | Slots for Figures 16, 17 |
| 9 | `sections/sec-07-operating-plan.tex` | Slot for Figure 18 |
| 10 | `sections/sec-08-san-diego-traction.tex` | Slot for Figure 19 |
| 11 | `sections/sec-09-risks-and-limits.tex` | Tables only |
| 12 | `sections/sec-10-build-method.tex` | Slot for Figure 20 |
| 13 | `sections/sec-11-references-backmatter.tex` | Abbreviations, availability, citation, references |
| 14 | `draft-capital-LaTeX.zip` | Overleaf bundle |
| 15 | second-to-last | Every defect in this stage's own files fixed |
| 16 | last | Stage README completed with the compile record |

### The bracketed drafting instruction

Every section carries `\draftinstr{...}` markers. Each one **must name an exact
repository path**, using `\appfile{}`, so stage 7 has nothing to search for.
The permitted source set is:

| Path | What stage 7 must take from it |
|:--|:--|
| `funding/science-golden-age/chunk-01-front-matter-and-summary.md` | The SBIR and STTR sentence in the summary of recommendations |
| `funding/science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md` | The NOVEL PERFORMERS section, the $200 billion finding, Table 1 |
| `funding/science-golden-age/chunk-04-chapter-three-securing-dominance-in-critical-and-emerging-technologies.md` | The institution-agnostic grants and SBIR or STTR deployment sentence |
| `funding/science-golden-age/chunk-05-chapter-four-science-and-technology-better-lives-of-all-americans.md` | The technician-founded ventures sentence |
| `funding/science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md` | The 3:1 private-to-federal leverage target and the non-federal cost-share instruction |
| `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` | The four-layer $3,500,000 budget frame, reused verbatim |
| `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed/` | The $306,000 and $1,300,000 split and its term |
| `funding/pdac-funding-applications/final-apply/references.bib` | The bibliography |
| `funding/supplementary/source-files/Physical-AI-Oncology-Trial-Competition-Proposal.zip` | The January 13, 2026 baseline the asset register dates from |
| `funding/RFA-RM-27-001-v2/` | The cover theme to vary from |
| `funding/potential-partners/UC-San-Diego/` | The Moores contact record and the first-in-human positioning correction |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | The protocol, IND and Phase 2 assets in the asset register |

### Constraints fixed at this stage and never changed later

- Twelve sections, one `.tex` each, numbered `sec-00` to `sec-11`.
- Twenty figures, numbered 1 to 20 in reading order, each in an `appfloat`.
- **Figure spacing invariant: `\end{appfig}` then `\vspace{-0.65cm}` then
  `\figcaption{}`, for every figure without exception.** `capstyle.sty` closes
  `appfig` with a rigid `\vskip 24.5pt`, so the frame-to-caption distance is
  exactly 24.5 pt minus 0.65 cm, that is 6.06 pt, everywhere.
- Every table is `\begin{apptable}` + `tabularx` at `\textwidth`, every fixed
  column `>{\raggedright\arraybackslash}p{...}`, every caption via `\tabcap`.
- No em dash, no double dash, no triple dash. Single hyphens only.
- `§` for every codified reference.
- No PNG, no JPG, anywhere.
