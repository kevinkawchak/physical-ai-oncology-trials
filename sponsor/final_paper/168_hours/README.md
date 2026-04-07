## 168-Hour Autonomous Sponsor Simulation

**4/6: v3.4.0 (168-Hour Autonomous Sponsor Simulation)** *Fully Automated Sponsor: 7-Day Continuous Simulation with 168 Commits* - 168 hourly Python scripts, 168 JSON outputs, 525 text diagrams, 7 daily summaries across 7 branches. Complete 7-day (168-hour) simulation of an autonomous AI-native pharmaceutical sponsor operating system for Physical AI oncology clinical trials. Extends the v3.3.0 24-hour simulation to demonstrate continuous 24/7 sponsor operations. [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18445179-blue)](https://doi.org/10.5281/zenodo.18445179)

## Simulation Overview

```
  168-Hour Simulation Architecture
  +-----------------+     +------------------+     +----------------+
  | Day 1: Init     | --> | Day 2: Enroll    | --> | Day 3: Safety  |
  | H000-H023       |     | H024-H047        |     | H048-H071      |
  | 168 patients     |     | 199 patients     |     | 218 patients   |
  | PSL 63.4-64.8   |     | PSL 64.8-66.0    |     | PSL 66.0-67.0  |
  +-----------------+     +------------------+     +----------------+
          |                        |                        |
          v                        v                        v
  +-----------------+     +------------------+     +----------------+
  | Day 4: Scale    | --> | Day 5: Analysis  | --> | Day 6: Audit   |
  | H072-H095       |     | H096-H119        |     | H120-H143      |
  | 233 patients     |     | 195 patients     |     | 173 patients   |
  | PSL 67.0-68.0   |     | PSL 68.0-68.8    |     | PSL 68.8-69.5  |
  +-----------------+     +------------------+     +----------------+
          |                        |                        |
          +------------------------+------------------------+
                                   |
                                   v
                          +------------------+
                          | Day 7: Closeout  |
                          | H144-H167        |
                          | 150 patients     |
                          | PSL 69.5-70.0    |
                          +------------------+
```

## Cumulative Statistics

| Metric                    | Value  |
|---------------------------|--------|
| Total hours               | 168    |
| Total sponsor decisions   | 2,016  |
| Total patients processed  | 1,336  |
| Total escalations         | 125    |
| Total robot authorizations| 1,336  |
| PSL start                 | 63.4   |
| PSL end                   | 70.0   |
| PSL improvement           | +6.6   |
| Text diagrams generated   | 525    |
| Python scripts            | 168    |
| JSON output files         | 168    |
| Daily summary files       | 7      |

## Directory Structure

```
168_hours/
  README.md                           # This file
  requirements.txt                    # Dependencies (stdlib only for simulation)
  run_168h_simulation.py              # Master 168-hour simulation runner
  _config.py                          # Configuration constants for all 7 days
  _gen_hourly.py                      # Hourly script and diagram generator
  _gen_day_summary.py                 # Daily summary and cumulative diagram generator
  _gen_init.py                        # __init__.py generator for each day
  _commit_hour.sh                     # Automation script for commit workflow
  instructions/                       # Real-time 168-hour execution instructions
    rtx_4090_openclaw/
      README.md                       # RTX 4090 setup (Linux, macOS, Windows)
    mac_mini_m4_pro_openclaw/
      README.md                       # Mac Mini M4 Pro setup (Linux, macOS, Windows)
  day_01/                             # Day 1: Trial Initialization (H000-H023)
    README.md
    hourly/
      __init__.py
      sponsor_hour_000.py - sponsor_hour_023.py
      output/
        sponsor_hour_000_output.json - sponsor_hour_023_output.json
    diagrams/
      sponsor_decision_flow_hour_000.txt - _023.txt
      agent_workload_hour_000.txt - _023.txt
      robot_auth_timeline_hour_000.txt - _023.txt
      cumulative_decision_timeline_day_01.txt
      cumulative_agent_utilization_day_01.txt
      cumulative_safety_summary_day_01.txt
    output/
      day_01_summary.json
  day_02/ - day_07/                   # Same structure for each day
```

## Day Themes

| Day | Theme                                         | Hours     | Patients | Escalations | PSL Range    |
|-----|-----------------------------------------------|-----------|----------|-------------|--------------|
| 1   | Trial Initialization and Baseline Operations  | H000-H023 | 168      | 13          | 63.4 - 64.8 |
| 2   | Enrollment Acceleration and Protocol Optimization | H024-H047 | 199  | 19          | 64.8 - 66.0 |
| 3   | Mid-Trial Safety Review and Adaptive Modifications | H048-H071 | 218 | 24          | 66.0 - 67.0 |
| 4   | Robotic Fleet Scaling and Cross-Site Coordination | H072-H095 | 233  | 27          | 67.0 - 68.0 |
| 5   | Data Analysis and Interim Reporting           | H096-H119 | 195      | 18          | 68.0 - 68.8 |
| 6   | Regulatory Compliance and Audit Preparation   | H120-H143 | 173      | 13          | 68.8 - 69.5 |
| 7   | Trial Closeout and Final Documentation        | H144-H167 | 150      | 11          | 69.5 - 70.0 |

## Execution

```bash
# Run the full 168-hour simulation (standalone mode, no dependencies required)
cd sponsor/final_paper/168_hours
python run_168h_simulation.py

# Generate files for a single hour
python _gen_hourly.py 42

# Generate summary for a specific day
python _gen_day_summary.py 3
```

## Technical Details

- All 168 Python scripts follow the same pattern as the v3.3.0 24-hour simulation
- 12 sponsor decisions per hour at 5-minute intervals across 12 agents
- 10 robot categories with 29 total instances per fleet
- 4-layer agent architecture: governance, execution, site/robotics, trust
- 7-gate decision framework (G1 auto through G7 executive approval)
- PSL (Protocol Safety Level) scoring tracks safety improvements over 168 hours
- All code passes ruff lint and format checks (Python 3.10+, line-length 120)
