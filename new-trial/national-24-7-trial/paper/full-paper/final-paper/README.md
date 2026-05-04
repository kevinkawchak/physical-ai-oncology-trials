# Accelerated Patient Prediction in Physical AI Oncology Clinical Trials: Four Comprehensive LLM Simulations - Full Paper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19994945.svg)](https://doi.org/10.5281/zenodo.19994945)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Parent Repository DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18445179.svg)](https://doi.org/10.5281/zenodo.18445179)
[![Sponsor Simulations DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19396256.svg)](https://doi.org/10.5281/zenodo.19396256)
[![National Platform DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19244918.svg)](https://doi.org/10.5281/zenodo.19244918)
[![Site Documentation DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19176370.svg)](https://doi.org/10.5281/zenodo.19176370)
[![Patient Journey DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19119939.svg)](https://doi.org/10.5281/zenodo.19119939)
[![Patient Instructions DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18810541.svg)](https://doi.org/10.5281/zenodo.18810541)

The polished, fully populated 70+ page LaTeX manuscript for the National 24/7
Continuous Real-Time Clinical Trial paper. This directory contains the
final-prose version that compiles to a single PDF in Overleaf. The earlier
template skeleton with bracketed processing instructions remains untouched at
`new-trial/national-24-7-trial/paper/`.

Released: May 3rd, 2026
Author: Kevin Kawchak, CEO ChemicalQDevice
DOI: [10.5281/zenodo.19994945](https://doi.org/10.5281/zenodo.19994945)

## Purpose

This directory holds the polished full paper that demonstrates, across four
author Physical AI oncology trial simulations, that Claude Code Opus 4.7 Max
with a 1M token context window produces substantially faster and more
powerful patient prediction for safety and efficacy than the supervised
models currently in clinical trial practice and the recent FDA Real-Time
Clinical Trial (RTCT) proof-of-concept program announced on 28 April 2026.

The four simulations split between clinical trial sites and clinical trial
sponsors. Simulations 1 and 2 are site-side simulations; Simulations 3 and
4 are sponsor-side simulations. Both perspectives are needed for a
continuous trial, and each simulation type complements the other: sites
generate the patient and robot signal stream, sponsors generate the
governance and regulatory decision stream, and the FDA RTCT pilot ingests
both.

## File Structure

```
new-trial/national-24-7-trial/paper/full-paper/
|-- main.tex                 # Document entry point; loads sections via \input
|-- new_paper.sty            # Style file (geometry, fonts, headers, abstract)
|-- references.bib           # Bibliography with DOIs and URLs (clickable)
|-- orcid_icon.png           # ORCID hyperlink icon for the title page
|-- README.md                # This file
|-- LaTeX_Source_Files.zip   # Overleaf-ready ZIP (added by commit 14)
|-- sections/
|   |-- abstract.tex         # Abstract (in title page)
|   |-- introduction.tex     # Section 1
|   |-- methods.tex          # Section 2
|   |-- results.tex          # Section 3 (4 simulations + cross-synthesis)
|   |-- discussion.tex       # Section 4 (FDA + AI baseline + significance)
|   |-- limitations_future.tex # Section 5 (per-sim limits + 2 future tracks)
|   |-- conclusions.tex      # Section 6
|   |-- back_matter.tex      # Acknowledgments, ethics, rights, citation
```

## Repository Structure (Text Diagram)

```
physical-ai-oncology-trials/
|
|-- new-trial/
|   |-- national-24-7-trial/
|   |   |-- FDA-April-2026/             # FDA RTCT 28 Apr 2026 announcement
|   |   |-- Background-A/               # 17-entry deep research summary (A)
|   |   |-- Background-B/               # 8-entry deep research summary (B)
|   |   |-- hour-00/ ... hour-55/       # 56 hours of minute-resolution data
|   |   |-- extra-hours/hour-56/        # 28 approximated hours (NOT in paper)
|   |   |   ... hour-83/                # excluded from full paper text
|   |   |-- paper/                      # ORIGINAL template (DO NOT EDIT)
|   |   |   |-- main.tex
|   |   |   |-- sections/
|   |   |   |-- new_paper.sty
|   |   |   |-- references.bib
|   |   |   |-- orcid_icon.png
|   |   |   |-- README.md
|   |   |   |-- LaTeX_Source_Files.zip
|   |   |   |
|   |   |   +-- full-paper/             # POLISHED full paper (this dir)
|   |   |       |-- main.tex
|   |   |       |-- sections/
|   |   |       |-- new_paper.sty
|   |   |       |-- references.bib
|   |   |       |-- orcid_icon.png
|   |   |       |-- README.md
|   |   |       +-- LaTeX_Source_Files.zip
|   |   +-- README.md
|   +-- site/                           # First-site documentation
|
|-- patient-journey/                    # Simulation 2 source
|   |-- stage_01_prescreening.py
|   |-- ... (10 stage scripts)
|   +-- stage_10_closeout.py
|
|-- sponsor/
|   +-- final_paper/
|       |-- scripts/                    # Simulation 3 source (24h)
|       |   |-- core_agents/            # 53 agent .py files
|       |   |-- coordination/
|       |   |-- dashboard/
|       |   |-- diagrams/               # 75 ASCII diagrams
|       |   |-- hourly/                 # sponsor_hour_00.py - _23.py
|       |   |-- safety/
|       |   +-- sponsor_server/         # FastAPI server
|       +-- 168_hours/                  # Simulation 4 source (168h)
|           |-- day_01/ ... day_07/     # 7 daily directories
|           +-- instructions/
|               +-- core_i5_6200u_4gb/  # Local verification
|
+-- (other repository sections: federation, frameworks, regulatory, etc.)
```

## Four Simulations (Clinical Trial Site vs. Sponsor)

| # | Simulation | Type | Repository Path | Key Outputs |
|---|------------|------|-----------------|-------------|
| 1 | Continuous National 24/7 RTCT (84 hours, multi-patient, no local agents) | Clinical Trial Site | `new-trial/national-24-7-trial/hour-00` through `hour-55` plus `extra-hours/hour-56` through `hour-83` | 3 ASCII diagrams + 4 markdown files per hour, 168 patients, 4 sites, 116 robots |
| 2 | Single-Patient 10-Stage Journey | Clinical Trial Site | `patient-journey/stage_01_prescreening.py` through `stage_10_closeout.py` | 10 Python scripts, FDA cost savings analysis, regulatory tables, single patient PAT-2026-0042 |
| 3 | 24-Hour Autonomous Sponsor (53 agents, 4 layers) | Clinical Trial Sponsor | `sponsor/final_paper/scripts/hourly/sponsor_hour_00.py` through `sponsor_hour_23.py` | 24 hourly scripts, 24 JSON outputs, 75 ASCII diagrams, 53 core agents |
| 4 | 168-Hour 7-Day Sponsor Extension (with local verification on Core i5-6200U) | Clinical Trial Sponsor | `sponsor/final_paper/168_hours/day_01/` through `day_07/`, `instructions/core_i5_6200u_4gb/` | 168 hourly scripts, 168 JSON outputs, 525 text diagrams, 7 daily summaries, 7 branches, 168 commits, 7 PRs |

Simulations 1 and 2 are site-side, where patient and robot signals
originate. Simulations 3 and 4 are sponsor-side, where governance,
regulatory, and decision-making logic resides. The complete continuous
real-time trial under the FDA RTCT framework requires both: sites generate
the signals, sponsors decide on those signals, and the FDA observes both
streams in real time.

## Cloud-Only versus Cloud-Plus-Local Reproducibility

```
+---------------------------------------------------------------------------+
|     CLOUD-ONLY VS CLOUD-PLUS-LOCAL REPRODUCIBILITY ACROSS THE FOUR SIMS   |
+----+--------------+-----------------+-----------------+-------------------+
| #  | Simulation   | Cloud-Only Adv. | Local-Plus Adv. | Reproducibility   |
+----+--------------+-----------------+-----------------+-------------------+
| 1  | RTCT 84h     | High throughput | n/a (text only) | High (deterministic|
|    |              | minute writeups | no Python yet   | text artifacts)   |
+----+--------------+-----------------+-----------------+-------------------+
| 2  | 10-stage     | Cloud composes  | Local re-run    | Highest (10 .py   |
|    | journey      | regulatory cite | each stage on   | files committed)  |
|    |              | tree            | constrained HW  |                   |
+----+--------------+-----------------+-----------------+-------------------+
| 3  | 24h sponsor  | 53 agents fan   | Local agent re- | High (24 .py +    |
|    |              | out in seconds  | run on i5-6200U | 24 JSON committed)|
+----+--------------+-----------------+-----------------+-------------------+
| 4  | 168h sponsor | 7-day agent     | Verified on i5- | Highest (168 .py +|
|    |              | sweep at speed  | 6200U with 4 GB | 168 JSON + 7 PRs) |
+----+--------------+-----------------+-----------------+-------------------+
```

Cloud-only simulations are reproducible because the cloud generates a
complete commit log; their disadvantage is that re-running requires cloud
compute access. Cloud-plus-local simulations are reproducible because the
local re-run validates the cloud output on independent hardware; their
disadvantage is that local hardware introduces operating-system-specific
quirks (Windows Update, antivirus, thermal throttling).

## Code-Based Versus Text-Only Simulations

```
+--------------------------------------------------------------------------+
|     CODE-BASED VS TEXT-ONLY SIMULATIONS - PRACTICAL TRADE-OFFS           |
+--------------------+-----------------------+-----------------------------+
| Dimension          | Text-Only (Sim 1)     | Code-Based (Sim 2, 3, 4)    |
+--------------------+-----------------------+-----------------------------+
| Cloud compute use  | Light (text only)     | Moderate to heavy           |
| Local compute use  | None                  | Yes (Sim 2 light, Sim 4 4GB)|
| Downstream agents  | Cannot consume Python | Can fan out to local agents |
| Auditability       | Markdown plus ASCII   | Markdown plus ASCII plus .py|
| Re-run determinism | Re-run varies (LLM)   | Identical (.py is fixed)    |
| Verification cost  | Minimal               | i5-6200U / 4 GB tested      |
+--------------------+-----------------------+-----------------------------+
```

The benefit of code-based simulations is practicality: a Python script can
re-execute on local hardware to verify the cloud-generated output. The
benefit of text-only simulations is that no Python execution surface is
required, which lowers the barrier for a clinical reader to audit the
output. A future Claude Code or competing AI local instance that runs both
on cloud (1M token context) and on local hardware (small specialist
agents) would offer both the power of cloud compute and the security and
flexibility of local computing for oncology trial sites.

## Build Instructions

This LaTeX paper compiles in Overleaf and locally with the standard
four-pass sequence:

```bash
cd new-trial/national-24-7-trial/paper/full-paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The bibliography uses the `ieeetr` style. Every reference includes a DOI
where one exists and a clickable URL through the `note` field. The
`LaTeX_Source_Files.zip` artifact bundles every file required by Overleaf
for a one-click upload.

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

## Disclaimer

This work is independent and is not endorsed or sponsored by any trial
sponsor, FDA, CRO, site, IRB, regulator, or medical society; and was
generated using Artificial Intelligence.
