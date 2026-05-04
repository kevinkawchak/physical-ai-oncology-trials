# 15 - Cross-Simulation Synthesis (Full Page)

## Purpose

Replace the cross-simulation synthesis ASCII block in Section 3.5
(Results, Cross-Simulation Synthesis) with a full-page tabular comparison
figure that spans both site (Sims 1 and 2) and sponsor (Sims 3 and 4)
sides of the FDA RTCT framework.

## Source Paper Section

`sections/results.tex` lines 670 to 690 (the cross-simulation synthesis
verbatim block).

## Image Properties

- Filename: `images/15_cross_simulation_synthesis.png`
- DPI: 300
- Size: 11 inches wide by 8.5 inches tall (US letter landscape, full page)
- Background: white (#FFFFFF)
- Palette: Sim 1 deep navy (#1F4E79), Sim 2 teal (#2C7A7A), Sim 3 gold
  (#B45424), Sim 4 deep purple (#6A4C8C). Header bar dark slate.

## Layout

- Top header: "Cross-Simulation Synthesis - Site (Sims 1, 2) and Sponsor
  (Sims 3, 4)."
- Header band that splits the four columns into a left pair labeled "Sites"
  and a right pair labeled "Sponsors" with a thin vertical divider.
- Nine-row dimension matrix: Scope, Hours, Patients, Robots, ASCII
  diagrams, Python scripts, JSON outputs, GitHub commits, Local
  verification.
- Bottom band: a single-sentence takeaway "Repository-scale 1M token
  context plus hourly commit cadence plus ASCII plus Markdown plus Python
  plus JSON artifacts is the computational signature shared across all
  four."

## Synthesis Data

| Dimension      | Sim 1 (site)        | Sim 2 (site)         | Sim 3 (sponsor)         | Sim 4 (sponsor)       |
| -------------- | ------------------- | -------------------- | ----------------------- | --------------------- |
| Scope          | Continuous RTCT 56h | Single patient 10st  | 24-hour sponsor          | 168-hour sponsor      |
| Hours          | 56                  | 1,120 days           | 24                      | 168                   |
| Patients       | 168 cumul.          | 1                    | 5 peak                  | 1,336 cum.            |
| Robots         | 116                 | 2 (Da Vinci, Franka) | 0 robots, 53 agents     | 0 robots, 53 agents   |
| ASCII diagrams | 168 (3/hr)          | 30 progress          | 75                      | 525                   |
| Python scripts | 0                   | 12 modules           | 24 hourly + 53 agents   | 168 hourly (+ shared) |
| JSON outputs   | 0                   | per-stage            | 24                      | 168                   |
| GitHub commits | 56 (1/hr)           | per-stage            | 24 (1/hr)               | 168 (1/hr)            |
| Local verif.   | n/a                 | local re-run         | n/a                     | i5-6200U / 4 GB OK    |

## Style Rules

- Single dashes only.
- Section sign U+00A7 where source uses SS.
- Black text on light fill.

## Suggested Caption

Figure 15: Cross-simulation synthesis across site Sims 1 and 2 and sponsor
Sims 3 and 4 over nine artifact dimensions.
