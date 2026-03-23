# v2.8.0 Development Prompt

This file contains the prompt used to generate the 24-hour on-demand Physical
AI oncology clinical trial simulation (v2.8.0).

Released on 23 March 2026
CEO Kevin Kawchak, ChemicalQDevice

## Prompt

The main prompt instructs Claude Code Opus 4.6 to simulate a full 24-hour
on-demand patient-centric Physical AI oncology clinical trial at a single
site. The simulation runs around the clock with 1-minute resolution (1,440
total minutes). Every 60 minutes of simulated time constitutes one commit.
The prompt produces 24 commits (Hour 00 through Hour 23) plus a 25th commit
for error correction and final summaries. All 25 commits are completed in a
single unattended Claude Code conversation with no human intervention.

Key specifications:
- Repository: kevinkawchak/physical-ai-oncology-trials
- All output files under: new-trial/
- 7 files per hourly directory (simulation, robot logs, patient records, PSL
  scores, 3 text diagrams)
- 6 files in final-commit directory
- 178 total files across 25 commits

The prompt defines the Physical AI Standard Level (PSL) framework with three
equally weighted dimensions:
- Dimension A (Omniscient): Based on ICH E6(R3) Adaption
- Dimension B (Omnipresent): Based on 21 CFR Part 50 Adaption
- Dimension C (Omnipotent): Based on 21 CFR Part 312 Adaption

The prompt specifies 10 robot types, 15 cancer types, 150-180 patients, and
detailed patient arrival patterns across 24 hours.

Required repository files to read include robot specifications, three
regulatory frameworks, USL framework, patient journey paper, supporting
Python modules, robot-specific USL evaluations, patient journey stages,
and configuration files - totaling 50+ specific repository file paths.

The full prompt text is maintained in the repository development records.
See the pull request description for the complete specification.
