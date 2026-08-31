# Stage 3, sub-prompt 4 - double verification of context

## Goal

Stage 3 is where the context is verified twice: once against the repository file
that supplied it, and once against the internal consistency of the paper. The
master prompt asks for maximum quality here, and the instrument is a checklist
that is run to completion rather than sampled.

## Pass 1, against the source

For every number, date, name and quoted phrase:

| Claim class | Verified against |
|:--|:--|
| Budget figures | `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` |
| Simulation results | `funding/pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex` |
| Cost benchmarks | `funding/capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex` |
| Company record and $36,330 | `funding/capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex` |
| Author qualifications and dates | `funding/move-in/inputs/ChemicalQDevice_Accomplishments.docx` |
| Deposited paper DOIs | `funding/move-in/inputs/READMES/README-LLM-Pancreatic-Oncology-Clinical-Trial-System-...md` |
| San Francisco predecessor facts | `funding/move-in/inputs/READMES/README-Physical-AI-Oncology-Clinical-Trial-Site-...md` |
| Partner positioning | `funding/potential-partners/UC-San-Diego/priority-steps.md` |
| Codified law citations | The codified source itself, cited in `references.bib` |

## Pass 2, internal consistency

| Check | Rule |
|:--|:--|
| Arithmetic | Every column of figures that claims a total sums to that total. Personnel at $521,000 plus non-personnel at $179,000 equals $700,000. Five years at $700,000 equals $3,500,000 |
| Full-time equivalents | The eleven role fractions sum to the stated 3.95 |
| Counts | Fifteen `\docpart` headings, fifteen strip cells on the cover, fifteen rows in the front matter document index, and "15 Documents" on the cover subtitle |
| Cross-references | Every table referred to by number exists and carries that number |
| Abbreviations | Defined at first use in the body and listed in the back matter |
| Definitions | A term defined in document 01 and reused in document 09 is either repeated there or explicitly cross-referenced |
| Dates | August 23, 2026 on the cover, in the running header, and in the citation line, and nowhere a different date for the same event |
| Version | v4.7.0 in the cover deposit line, the running header, and every README badge |
| DOI placeholder | `10.5281/zenodo.xxxxxxxx` in the cover, the availability paragraph, and the citation line, and nowhere a fabricated number |

## Recorded, not absorbed

Any defect found in this pass is recorded in `prompts/output-move-in.md` with
its measured size, in the manner of the parent builds. A defect that is fixed
silently teaches the next build nothing.

## Acceptance

Both passes complete, every row checked, every defect either fixed or recorded
with the reason it was not.

## Commit

The stage error pass commit, second to last in the stage.
