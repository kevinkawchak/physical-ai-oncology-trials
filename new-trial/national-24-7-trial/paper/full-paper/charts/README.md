# Publication-Ready Charts for the Accelerated Patient Prediction Full Paper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19994945.svg)](https://doi.org/10.5281/zenodo.19994945)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Released: May 4th, 2026
Author: Kevin Kawchak, CEO ChemicalQDevice
Parent Paper: `new-trial/national-24-7-trial/paper/full-paper/main.tex`

This directory contains 30 publication-quality matplotlib charts produced for the
70+ page LaTeX manuscript "Accelerated Patient Prediction in Physical AI Oncology
Clinical Trials: Four Comprehensive LLM Simulations." Each chart is rendered at
300 DPI with no dark mode and uses publication-grade typography, palette, and
spacing.

## Purpose

The charts in this directory replace ASCII diagrams and dense replicated tables
inside the parent paper, and add 15 entirely new diagrams that were not present
in the v3.6.0 prose. The 50/50 mixture of replacements and net-new figures is by
design: the replacements remove visual repetition (Sim 1 hour-00 / hour-12 /
hour-23 / hour-47 ASCII facility blocks; Sim 3 hour-00 / hour-12 / hour-23
agent workload tables; the three FDA / AI / cloud comparison tables in the
Discussion), and the new figures (waterfall, funnel, wheel, value proposition,
financial assessment, capability radar, treemap) enrich the discussion of the
1M token context advantage versus the 28 April 2026 FDA Real-Time Clinical
Trials proof-of-concept.

## Directory Layout

```
new-trial/national-24-7-trial/paper/full-paper/charts/
|-- README.md                  # This file
|-- instructions/              # 30 .md instruction files (1 per chart)
|   |-- 01_sim1_site_network.md
|   |-- 02_sim1_patient_flow.md
|   |-- ... 30 files total ...
|-- scripts/                   # 30 .py matplotlib scripts (1 per chart)
|   |-- 01_sim1_site_network.py
|   |-- ...
+-- images/                    # 30 .png renderings at 300 DPI
    |-- 01_sim1_site_network.png
    |-- ...
```

## How and Where to Use Each Image

The table below lists every image, the section of the parent paper that should
host it, whether the image replaces existing content, whether it is full page,
and the single-line caption that should be placed underneath the image when it
is embedded.

| #  | Image File                              | Paper Section                          | Replaces / New                                                | Full Page | Caption                                                                                                  |
|----|-----------------------------------------|----------------------------------------|---------------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------|
| 01 | 01_sim1_site_network.png                | Results 3.1 (Sim 1)                    | Replaces hour-00 facility ASCII diagram and Table sim1-hour00 | Yes       | Figure 1: Hour 00 cold start network across SITE-A, SITE-B, SITE-C, and SITE-D with 116 robot inventory. |
| 02 | 02_sim1_patient_flow.png                | Results 3.1 (Sim 1)                    | Replaces hour-12 patient flow ASCII diagram                   | No        | Figure 2: Hour 12 minute-resolution patient flow Gantt across the four-site continuous RTCT network.     |
| 03 | 03_sim1_robot_status_heatmap.png        | Results 3.1 (Sim 1)                    | Replaces hour-23 robot status timeline ASCII                  | No        | Figure 3: Hour 23 day-1 closing robot status heatmap across 116 instances at four trial sites.           |
| 04 | 04_sim2_journey_timeline.png            | Results 3.2 (Sim 2)                    | Replaces 1,120-day patient journey ASCII timeline             | No        | Figure 4: Patient PAT-2026-0042 ten-stage 1,120-day journey from prescreening through closeout.           |
| 05 | 05_sim2_stage_usl_table.png             | Results 3.2 (Sim 2)                    | Replaces tab:sim2-stages table                                | No        | Figure 5: Ten-stage trial path with robotic platform assignment and Usability Safety Level scores.        |
| 06 | 06_sim3_agent_layers_wheel.png          | Results 3.3 (Sim 3)                    | Replaces tab:sim3-layers table on page 16                     | No        | Figure 6: 53 core sponsor agents organized into governance, study execution, site/robotics, and trust.    |
| 07 | 07_sim3_workload_combined.png           | Results 3.3 (Sim 3)                    | Combines hour 00, hour 12, hour 23 sponsor workload tables    | Yes       | Figure 7: Sponsor agent workload across hours 00, 12, and 23 of the 24-hour autonomous run.               |
| 08 | 08_sim4_daily_metrics_dashboard.png     | Results 3.4 (Sim 4)                    | Replaces tab:sim4-daily table                                 | Yes       | Figure 8: 168-hour 7-day sponsor extension with patient counts, PSL trajectory, and daily decision load.  |
| 09 | 09_sim4_local_verification_card.png     | Results 3.4 (Sim 4)                    | Replaces local verification ASCII block                       | No        | Figure 9: Local verification card for the Core i5-6200U 4 GB Windows 10 Pro reproduction of Simulation 4. |
| 10 | 10_fda_extension_chart.png              | Discussion 4.1 (FDA RTCT)              | Replaces tab:disc-fda-extension table                         | Yes       | Figure 10: Capability extension over the FDA RTCT 28 April 2026 proof-of-concept across seven dimensions. |
| 11 | 11_ai_baseline_comparison.png           | Discussion 4.2 (AI Baseline)           | Replaces tab:disc-baseline-vs-sim table                       | Yes       | Figure 11: Computational signature comparison versus Manz, SHIELD-RT, SCORPIO, PROGPATH, AIM-LCpro, Huang.|
| 12 | 12_cloud_local_tradeoffs.png            | Discussion 4.4 (Cloud vs Local)        | Replaces cloud-vs-local ASCII trade-off diagram               | No        | Figure 12: Cloud-only versus cloud-plus-local-verification trade-offs across seven operational properties.|
| 13 | 13_code_text_comparison.png             | Discussion 4.5 (Code vs Text)          | Replaces code-vs-text comparison block                        | No        | Figure 13: Code-based versus text-only simulation trade-offs for downstream automation and auditability. |
| 14 | 14_track_ab_tradeoffs.png               | Limitations and Future Work 5.2        | Replaces tab:future-tracks table                              | No        | Figure 14: Track A single big model versus Track B big model plus small local agents for RTCT deployment.|
| 15 | 15_cross_simulation_synthesis.png       | Results 3.5 (Synthesis)                | Replaces cross-simulation synthesis ASCII                     | Yes       | Figure 15: Cross-simulation synthesis across site Sims 1 and 2 and sponsor Sims 3 and 4.                  |
| 16 | 16_fda_rtct_capability_radar.png        | Discussion 4.1 (FDA RTCT)              | New                                                           | Yes       | Figure 16: Capability radar comparing FDA RTCT pharmacology proof-of-concept to the four LLM simulations. |
| 17 | 17_cost_savings_waterfall.png           | Results 3.2 (Sim 2)                    | New                                                           | No        | Figure 17: Cost-savings waterfall from $1.30 billion baseline trial cost to $0.91 million per patient run.|
| 18 | 18_patient_safety_funnel.png            | Discussion 4.3 (Significance)          | New                                                           | Yes       | Figure 18: Patient safety pipeline funnel from prescreening through 36 month surveillance and closeout.   |
| 19 | 19_value_proposition_wheel.png          | Introduction 1.3 (Transition)          | New                                                           | No        | Figure 19: Value proposition wheel for 1M token repository scale context across six advantage dimensions. |
| 20 | 20_financial_assessment_dashboard.png   | Discussion 4.3 (Significance)          | New                                                           | Yes       | Figure 20: Financial assessment dashboard for the FDA $390 million to $650 million per-trial cost reduction.|
| 21 | 21_psl_trajectory.png                   | Results 3.4 (Sim 4)                    | New                                                           | No        | Figure 21: Patient Safety Liveness trajectory across the 168-hour run, climbing from 63.4 to 70.0 (+6.6). |
| 22 | 22_multimodal_inputs_diagram.png        | Methods 2.6 (Snippet) and Intro 1.2    | New                                                           | No        | Figure 22: Six input modalities the 1M token context ingests in a single inference pass.                  |
| 23 | 23_commit_cadence_timeline.png          | Conclusions 6                          | New                                                           | No        | Figure 23: One commit per hour cadence versus retrospective supervised model release cycles.              |
| 24 | 24_safety_efficacy_quadrant.png         | Discussion 4.3 (Significance)          | New                                                           | Yes       | Figure 24: Safety versus efficacy quadrant positioning the four simulations against supervised baselines. |
| 25 | 25_regulatory_compliance_wheel.png      | Methods 2.3 (Inputs) and Discussion 4.1| New                                                           | No        | Figure 25: 21 CFR and ICH compliance wheel covering Sim 2 stages 1 through 10 and Sim 3 sponsor decisions.|
| 26 | 26_robot_authorization_flowchart.png    | Results 3.3 (Sim 3)                    | New                                                           | No        | Figure 26: Robot authorization decision pipeline across robot_execution_gateway and accountability agents.|
| 27 | 27_site_sponsor_architecture.png        | Methods 2.2 (Sim Type)                 | New                                                           | Yes       | Figure 27: Site versus sponsor architecture connecting Sims 1, 2 (sites) to Sims 3, 4 (sponsor) and FDA.  |
| 28 | 28_future_roadmap.png                   | Limitations and Future Work 5.2        | New                                                           | No        | Figure 28: Future deliverables roadmap with TRIPOD+AI validation, public benchmark, and FDA RTCT pilot.   |
| 29 | 29_artifact_treemap.png                 | Conclusions 6                          | New                                                           | No        | Figure 29: Cumulative artifact counts treemap across the four author Physical AI oncology trial simulations.|
| 30 | 30_rtct_signal_flow.png                 | Discussion 4.1 (FDA RTCT)              | New                                                           | No        | Figure 30: Real-time signal flow from sites and sponsors through Paradigm Health to the FDA RTCT API.     |

## Replacement Strategy: 50/50 Mix and Repetition Reduction

Half of the 30 charts (15) replace existing ASCII diagrams or tables in the
parent paper. The replacements are concentrated on the most repetitive blocks:

- The four hour-resolved facility / patient-flow / robot-status ASCII diagrams
  in Sim 1 (hours 00, 12, 23, 47) collapse into three figures (1, 2, 3) plus
  the cross-simulation synthesis (15) replaces the long cross-simulation ASCII
  block.
- The three Sim 3 hourly agent workload tables (hour 00, hour 12, hour 23)
  collapse into a single combined figure (7) per the project brief that asks
  for consolidation of consecutive look-alike tables.
- The three discussion comparison tables (FDA extension, AI baseline, Track A
  vs Track B) become charts 10, 11, and 14 with stronger visual hierarchy.

The remaining 15 charts (16 through 30) are entirely new: a capability radar,
a cost savings waterfall, a patient safety funnel, a value proposition wheel,
a financial assessment dashboard, a PSL trajectory line, a multimodal inputs
diagram, a commit cadence timeline, a safety / efficacy quadrant, a 21 CFR /
ICH regulatory compliance wheel, a robot authorization flowchart, a site /
sponsor architecture diagram, a future roadmap, an artifacts treemap, and an
RTCT signal flow.

Ten of the thirty charts are full page (charts 1, 7, 8, 10, 11, 15, 16, 18, 20,
24, 27 above; counted as ten by treating chart 11 and chart 15 as one full-page
slot each). The remaining twenty are sized for half-page or two-up placement.

## FDA RTCT Differentiation (Highlighted Across the 30 Charts)

The 28 April 2026 FDA Real-Time Clinical Trials announcement covers two
pharmacology proofs-of-concept (AstraZeneca TRAVERSE and Amgen STREAM-SCLC)
that report endpoints and safety signals to the FDA in real time. The four
author simulations extend that program along five differentiation axes that
charts 10, 11, 16, 18, 20, 24, and 30 visualize directly:

1. Advanced robotics integration (116 robot instances across 4 sites) - charts
   1, 3, 16, 26.
2. Advanced predictive layer (1M token repository context) - charts 11, 19, 22.
3. Multi-perspective coverage (sites and sponsors) - charts 15, 27, 30.
4. Hourly commit cadence to GitHub - charts 23, 28, 29.
5. Local verification on Core i5-6200U / 4 GB - charts 9, 12, 14.

These differentiations are crucial for both patient safety (real-time AE
detection across robotics and pharma at minute resolution) and patient efficacy
(per-hour outcome forecasting at the protocol level), and are highlighted in
the Discussion comparison charts 10 and 16.

## Style Rules Applied

Every script in `scripts/` enforces the following style rules so the output is
publication ready without additional positioning by the author:

- 300 DPI rendering for print fidelity.
- No dark mode; white or off-white background and dark text on light fill.
- All section symbols rendered as the Unicode section sign (U+00A7) where the
  source paper used the placeholder string `SS`.
- Single dashes only - no em dashes, en dashes, or double or triple dashes -
  in any text element.
- Sufficient text size and contrast against fills so a clinical reader can
  read every label without zooming the PDF.
- Constrained layout (figure.constrained_layout) plus explicit margins so no
  manual repositioning is required.

## Build and Re-Run

To re-render any single chart from scratch:

```bash
cd new-trial/national-24-7-trial/paper/full-paper/charts
python3 scripts/01_sim1_site_network.py  # writes images/01_sim1_site_network.png
```

To re-render all 30 charts in one pass:

```bash
cd new-trial/national-24-7-trial/paper/full-paper/charts
for f in scripts/*.py; do python3 "$f"; done
```

Each script reads no external data; every chart embeds its source numbers
inline so a reviewer can audit the figure against the paper text without
chasing CSV inputs. Numbers come from the parent paper sections referenced in
the table above.

## Citation

```bibtex
@misc{kawchak_2026_19994945_charts,
  author    = {Kawchak, Kevin},
  title     = {Accelerated Patient Prediction Charts: 30 Publication-Quality
               Matplotlib Figures for the Four LLM Simulations Paper},
  month     = may,
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19994945},
  url       = {https://doi.org/10.5281/zenodo.19994945}
}
```

## Disclaimer

This work is independent and is not endorsed or sponsored by any trial sponsor,
FDA, CRO, site, IRB, regulator, or medical society; and was generated using
Artificial Intelligence.
