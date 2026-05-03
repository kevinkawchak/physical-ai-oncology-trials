# Accelerated Patient Prediction in Physical AI Oncology Clinical Trials: Four Comprehensive LLM Simulations

LaTeX paper template (skeleton + bracketed processing instructions) for the
70+ page manuscript at `new-trial/national-24-7-trial/paper/`.

Released: May 3rd, 2026
Author: Kevin Kawchak, CEO ChemicalQDevice
DOI: [10.5281/zenodo.19994945](https://doi.org/10.5281/zenodo.19994945)

## Purpose

This template prepares the LaTeX skeleton, style, bibliography, and
section-level processing instructions that a future Claude Code Opus 4.7
Max (1M token context) run will execute to produce the final 70+ page paper.
The template itself does NOT contain narrative prose for sections that
require synthesis of repository content - those sections contain bracketed
instructions that name the exact repository directories and files to read,
the exact diagrams and tables to render, and the formatting rules to apply.

The single Methods section is the exception: per the prompt, Methods is
written directly as final prose, not as bracketed instructions.

The main goal of the future generation pass is to demonstrate, across four
author simulations, that Claude Code can produce substantially faster and
more powerful patient predictions for safety and efficacy than the AI
patient-prediction methods in current oncology clinical trial practice
(supervised models trained retrospectively) and the recent FDA RTCT
proof-of-concept program announced on 28 April 2026.

## File Structure

```
new-trial/national-24-7-trial/paper/
|-- main.tex                 # Document entry point; loads sections via \input
|-- new_paper.sty            # Style file (geometry, fonts, headers, abstract)
|-- references.bib           # Bibliography with DOIs and URLs (clickable)
|-- orcid_icon.png           # ORCID hyperlink icon for the title page
|-- README.md                # This file
|-- sections/
|   |-- abstract.tex         # Abstract (in title page)
|   |-- introduction.tex     # Section 1 (continues on title page)
|   |-- methods.tex          # Section 2 (final prose, no instructions)
|   |-- results.tex          # Section 3 (instructions for 4 simulations)
|   |-- discussion.tex       # Section 4 (instructions referencing Results)
|   |-- limitations_future.tex # Section 5 (instructions: limits + future)
|   |-- conclusions.tex      # Section 6 (instructions for summary)
|   |-- back_matter.tex      # Acknowledgments, ethics, rights, citation
```

## How Files Relate

- `main.tex` is the entry point; it loads `new_paper.sty` and every section
  in `sections/` via `\input{sections/<name>}`.
- `new_paper.sty` defines page geometry, font sizes, header rules, abstract
  styling, widow/orphan suppression, and section spacing.
- `references.bib` is referenced by `\bibliography{references}` near the end
  of `main.tex`.
- `orcid_icon.png` is invoked by the `\orcidicon` command on the title page.

## Source Repository Directories Referenced by Each Section

This template explicitly names the repository directories that the next
generation pass must read into context. The directories are:

- A. `new-trial/` (and subdirectories) - root of the on-demand simulation
- B. `new-trial/national-24-7-trial/` - the continuous RTCT simulation,
  including FDA-April-2026, Background-A, Background-B, and hours-00-55
  plus extra-hours/hour-56 through extra-hours/hour-83
- C. `new-trial/site/` - California first-site documentation package
  (Simulation 1 source)
- D. `patient-journey/` (and subdirectories) - 10-stage single-patient
  journey orchestration (Simulation 2 source)
- E. `patient-journey/paper/patient_journey_paper.tex` - prior paper for
  abstract length calibration
- F. `sponsor/final_paper/` (and subdirectories) - final autonomous sponsor
  paper, source of Simulations 3 and 4
- G. `sponsor/final_paper/scripts/` - 24-hour sponsor scripts (Simulation 3)
  and `sponsor/final_paper/168_hours/` (Simulation 4)

## Four Simulation Summary (mapped to Results subsections)

| # | Simulation | Repository Path | Key Outputs |
|---|------------|-----------------|-------------|
| 1 | Continuous RTCT (24+ hours, multi-patient, no local agents) | `new-trial/national-24-7-trial/hour-00` through `hour-55` plus `extra-hours/hour-56` through `hour-83` | 3 ASCII diagrams + 4 markdown files per hour, 168 patients, 4 sites, 116 robots |
| 2 | Single-patient 10-stage journey | `patient-journey/stage_01_prescreening.py` through `stage_10_closeout.py` | 10 Python scripts, FDA cost savings, regulatory guidance, single patient PAT-2026-0042 |
| 3 | 24-hour autonomous sponsor (24 Python + 24 JSON) | `sponsor/final_paper/scripts/hourly/sponsor_hour_00.py` through `sponsor_hour_23.py` | 24 hourly scripts, 24 JSON outputs, ~75 ASCII diagrams, 53+ core agents |
| 4 | 168-hour 7-day extension with local verification | `sponsor/final_paper/168_hours/day_01/` through `day_07/`, `instructions/core_i5_6200u_4gb/` | 168 hourly scripts, 168 JSON outputs, 525 text diagrams, 7 daily summaries, 7 branches, 168 commits, 7 PRs |

## Processing Instructions for the Next Claude Code 4.7 Max Run

1. Read every file listed under "Source Repository Directories" above into
   the 1M context window.
2. Open each section file in `sections/` and execute the bracketed
   instructions in order (top to bottom of `main.tex`).
3. The `methods.tex` section is final prose - do not modify or replace it.
4. For every \cite{} introduced, confirm that `references.bib` already
   contains the matching key. If a new key is needed, add it with a DOI,
   URL, and note triplet (see header of `references.bib`).
5. Run `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex
   main.tex` and visually verify that the bibliography page lists clickable
   DOIs and URLs for every entry, with no right-margin overflow.
6. After the body is populated, perform the senior-author formatting pass:
   eliminate orphans/widows, eliminate large white spaces, ensure no page
   has excessive empty space without text, replace any em or double dashes
   with single dashes, and replace "SS" with the section symbol where it
   means a section reference.
7. Final output target: a 70+ page PDF compiled in Overleaf without errors.

## LaTeX Compilation

```bash
cd new-trial/national-24-7-trial/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The repository ships an Overleaf-ready ZIP (`LaTeX_Source_Files.zip`) at
the same path that contains `main.tex`, `new_paper.sty`, `references.bib`,
`orcid_icon.png`, and the `sections/` directory.

## Citation

```bibtex
@misc{kawchak_2026_19994945,
  author    = {Kawchak, Kevin},
  title     = {Accelerated Patient Prediction in Physical {AI} Oncology
               Clinical Trials: Four Comprehensive {LLM} Simulations},
  month     = may,
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19994945},
  url       = {https://doi.org/10.5281/zenodo.19994945}
}
```
