## Prompt 2nd.md - Claude Code Opus 4.6 02/14/26

Please use the same approach as the last conversation and pull request to create py, html, and png files to complete your tasks in a timely manner (but avoid your output from stalling during image generations like the last conversation). No changelog or release notes are needed. Update the existing “images” directory keeping both interactive and png directories, and for each set of files to have new directories “1st” (the last PR files), 2nd (the current PR files) under both the interactive and png directories. 

Based on the following 10 sets of instructions, using Python Plotly or similar approaches, create high quality and comprehensive interactive visualization scripts that correspond with easy access to view the scripts in html in both light mode and dark mode for “interactive” that will also output 20 high resolution .png diagrams/charts, etc. in “png”. It is important that visualizations be self standing, as some less effective visualizations will be removed by the user if ineffective. 

The visualizations detail processes, methods, mechanisms, and metrics critical for engineers building physical AI oncology trials. It is important not to use file name number prefixes such as 11 or 20, as some images may be removed by the user. Use the following 10 sets of instructions as a guide only, making sure the accuracy and relevance of recent data and effectiveness of visualizations come through extensive analysis for correct context of the current version of the repository. Be sure to fix and address errors that would cause failed checks for the pull request (the most prior PR made sure the Python environment is correct to avoid the following error during final checks): “3 failing checks 
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... “ When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Do not make changes to changelog or release notes.

“Start Instructions”
## Instruction 11 — Agentic AI End-to-End Latency Breakdown

**Chart type:** Stacked bar chart with 95th percentile scatter overlay (`go.Bar` barmode='stack' + `go.Scatter`)

**Title:** `"Agentic AI Pipeline: End-to-End Latency Breakdown (Command → Robot Action)"`

**Purpose:** Show engineers the latency budget for each stage of the agentic pipeline from voice/text command to physical robot execution, with 95th percentile bounds for worst-case planning.

**Data source file:** `agentic-ai/results.md` — Section: "Latency Distribution"

**X-axis label:** `"Latency Component"`
**Y-axis label:** `"Latency (milliseconds)"`

**Data to plot (stacked bar showing median, with scatter for P95):**
| Stage | Median (ms) | P95 (ms) |
|---|---|---|
| Command → Recognition | 200 | 300 |
| Recognition → Planning | 150 | 280 |
| Planning → Execution | 300 | 500 |
| **Total End-to-End** | **650** | **1000** |

Show as a single stacked bar with 3 colored segments for the first 3 stages. Add a separate bar or annotation for the total. Scatter points on each segment for P95.

Also reference from `generative-ai/results.md` — "Real-Time Latency Benchmarks": GR00T N1.6 edge = 5ms (200 Hz), Diffusion distilled = 25ms (40 Hz), VLM+Policy async = 150ms planning + 5ms execution (200 Hz control).

**Light mode:** `plotly_white`, segments in blue/teal/green, P95 markers in red.
**Dark mode:** `plotly_dark`, segments in cyan/teal/lime, P95 markers in orange.

---

## Instruction 12 — Multi-Modal Input Fusion: Accuracy vs Latency

**Chart type:** Bar chart with secondary axis scatter for latency (`go.Bar` + `go.Scatter` on secondary y-axis)

**Title:** `"Multi-Modal Input Fusion: Accuracy and Latency by Modality Combination"`

**Purpose:** Show engineers how combining input modalities (speech, vision, context) improves command accuracy while increasing latency, enabling informed tradeoff decisions for their clinical setup.

**Data source file:** `agentic-ai/results.md` — Section: "Multi-Modal Input Fusion"

**X-axis label:** `"Input Modality"`
**Y-axis label (left):** `"Accuracy (%)"`
**Y-axis label (right):** `"Latency (ms)"`

**Data to plot:**
| Modality | Accuracy (%) | Latency (ms) |
|---|---|---|
| Speech Only | 92.3 | 200 |
| Vision Only | 89.7 | 150 |
| Speech + Vision | 96.5 | 280 |
| Speech + Vision + Context | 98.1 | 350 |

Add horizontal dashed line at 95% accuracy labeled "Clinical Acceptance Threshold". Annotate the bar for "Speech + Vision + Context" with "Best: 98.1% accuracy".

**Light mode:** `plotly_white`, accuracy bars in steel blue, latency scatter in coral.
**Dark mode:** `plotly_dark`, accuracy bars in dodger blue, latency scatter in gold.

---

## Instruction 13 — Multi-Agent Surgical Cooperation Performance

**Chart type:** Grouped bar chart with 3 metric clusters (`go.Bar` barmode='group')

**Title:** `"Multi-Agent Surgical Cooperation: Procedure Time, Collisions, and Success Rate"`

**Purpose:** Quantify the benefits of adding AI agents to surgical teams, showing the tradeoffs between procedure time reduction, collision avoidance, and overall success.

**Data source files:**
- `agentic-ai/results.md` — Section: "Multi-Agent Surgical Cooperation" (2H baseline, 1H+1A, 2A)
- `generative-ai/results.md` — Same table confirmed: "Human-Robot Team Performance"
- `reinforcement-learning/results.md` — Section: "Two-Agent Camera + Retractor" and "Surgeon Satisfaction"

**X-axis label:** `"Team Configuration"`
**Y-axis label:** `"Metric Value (%)"`

**Data to plot:**
| Configuration | Procedure Time (% of baseline) | Collision Rate (% of baseline) | Success Rate (%) |
|---|---|---|---|
| 2 Humans (Baseline) | 100 | 100 | 94 |
| 1 Human + 1 Agent | 55.6 | 55.3 | 96 |
| 2 Agents | 28.8 | 2.0 | 92 |

Also from `agentic-ai/results.md` — Task-specific time reductions: Tissue retraction -52%, Camera positioning -63%, Instrument exchange -41%, Suture assistance -38%. Add these as text annotations.

From `reinforcement-learning/results.md` — Surgeon satisfaction: Independent 3.2/5, Centralized 4.1/5, Communication-based 4.5/5.

**Light mode:** `plotly_white`, Time=blue, Collisions=red, Success=green.
**Dark mode:** `plotly_dark`, Time=cyan, Collisions=salmon, Success=lime.

---

## Instruction 14 — ORBIT-Surgical RL Benchmark: 14 Tasks (PPO vs SAC)

**Chart type:** Horizontal grouped bar chart (`go.Bar` orientation='h', barmode='group')

**Title:** `"ORBIT-Surgical Benchmark: PPO vs SAC Success Rates Across 14 Surgical Tasks"`

**Purpose:** Provide engineers with the complete benchmark landscape of RL algorithm performance on the dVRK surgical robot platform, ordered from easiest to hardest tasks.

**Data source file:** `reinforcement-learning/results.md` — Section: "ORBIT-Surgical Benchmark Suite (14 tasks, dVRK platform)"

**X-axis label:** `"Success Rate (%)"`
**Y-axis label:** `"Surgical Task"`

**Data to plot (ordered by PPO success rate, descending):**
| Task | PPO (%) | SAC (%) |
|---|---|---|
| Reach | 98.2 | 97.8 |
| Lift | 95.7 | 96.3 |
| Peg Transfer | 94.1 | 93.8 |
| Needle Pickup | 89.3 | 91.2 |
| Needle Handover | 85.7 | 87.4 |
| Gauze Cutting | 85.3 | 86.7 |
| Tissue Retraction | 82.4 | 84.1 |
| Tissue Manipulation | 79.8 | 82.3 |
| Thread Through Rings | 78.4 | 81.2 |
| Suture Throw | 78.2 | 76.9 |
| Tissue Cutting | 76.2 | 78.4 |
| Needle Driving | 72.5 | 75.8 |
| Debridement | 68.4 | 71.2 |
| Resection | 62.1 | 65.8 |

Add vertical dashed line at 80% labeled "Clinical Feasibility Threshold". Add vertical dashed line at 90% labeled "Deployment Ready".

**Light mode:** `plotly_white`, PPO=blue, SAC=orange.
**Dark mode:** `plotly_dark`, PPO=cyan, SAC=amber.

---

## Instruction 15 — GPU Hours to 80% Success: Algorithm Sample Efficiency

**Chart type:** Grouped bar chart with log-scale y-axis (`go.Bar`, barmode='group', yaxis type='log')

**Title:** `"Training Efficiency: GPU Hours to Reach 80% Success Rate by Algorithm"`

**Purpose:** Help engineers budget compute resources by comparing how quickly different RL algorithms reach clinical-feasibility performance thresholds across tasks of varying difficulty.

**Data source file:** `reinforcement-learning/results.md` — Section: "Training Efficiency Comparison (GPU hours to reach 80% success)"

**X-axis label:** `"Surgical Task"`
**Y-axis label:** `"GPU Hours to 80% Success (log scale)"`

**Data to plot:**
| Task | PPO (hours) | SAC (hours) | DreamerV3 (hours) |
|---|---|---|---|
| Reach | 0.5 | 0.3 | 0.2 |
| Needle Pickup | 8 | 4 | 1.5 |
| Tissue Retraction | 24 | 12 | 4 |
| Suture Throw | 48 | 32 | 12 |
| Debridement | 96 | 64 | 24 |

Add annotations showing speedup ratios: DreamerV3 is 4x faster than PPO for Debridement. Reference from `reinforcement-learning/results.md` — "DreamerV3 Ablation": Full model = 50K samples baseline; without imagination = 150K (-67%); without representation = 200K (-75%).

**Light mode:** `plotly_white`, PPO=slate blue, SAC=coral, DreamerV3=forest green.
**Dark mode:** `plotly_dark`, PPO=cornflower blue, SAC=salmon, DreamerV3=spring green.

---

## Instruction 16 — Safety-Constrained RL: Performance vs Violation Rate (Pareto)

**Chart type:** Scatter plot with Pareto frontier annotation (`go.Scatter` mode='markers+text')

**Title:** `"Safety-Constrained RL: Task Success vs Safety Violation Tradeoff"`

**Purpose:** Visualize the Pareto frontier between task performance and safety violations, guiding engineers toward the right constraint approach for clinical deployment where safety violations must be near zero.

**Data source files:**
- `reinforcement-learning/results.md` — Section: "Constrained Policy Optimization" and "Performance vs Safety Tradeoff"
- `configs/training_config.yaml` — Section: `safety` — `max_force: 5.0` N, `force_penalty_weight: 10.0`, `max_velocity: 0.1` m/s, `velocity_penalty_weight: 5.0`, `workspace_bounds: x [-0.3, 0.3], y [-0.3, 0.3], z [0.0, 0.6]`, `critical_structure_margin: 5.0` mm

**X-axis label:** `"Safety Violation Rate (%)"`
**Y-axis label:** `"Task Success Rate (%)"`

**Data to plot (4 points):**
| Algorithm | Success (%) | Violations (%) | Marker Size |
|---|---|---|---|
| PPO (Unconstrained) | 89 | 12.0 | 20 |
| PPO + Penalty | 84 | 5.0 | 20 |
| CPO | 82 | 1.5 | 20 |
| Safe Layer | 78 | 0.2 | 20 |

Draw a dashed Pareto frontier line through all 4 points. Add a vertical dashed line at 1% violations labeled "Clinical Safety Threshold". Add a shaded green region in the bottom-right (high success, low violations) labeled "Deployment Zone". Annotate each point with its algorithm name.

Also from `reinforcement-learning/results.md` — Error recovery rates: Excessive force 98% detection / 92% recovery / 0.8s; Workspace violation 99% / 95% / 0.5s; Collision prediction 94% / 88% / 0.3s. Add as text box annotation.

**Light mode:** `plotly_white`, points in blue with red danger zone, green safe zone.
**Dark mode:** `plotly_dark`, points in cyan with red danger zone, green safe zone.

---

## Instruction 17 — Sim-to-Real Transfer Gap by Surgical Task

**Chart type:** Dumbbell/lollipop chart (pairs of connected dots: `go.Scatter` for sim, `go.Scatter` for real, connected by lines)

**Title:** `"Sim-to-Real Transfer Gap: dVRK Physical Platform Validation (5 Core Tasks)"`

**Purpose:** Visualize the performance gap between simulation and physical robot execution for each surgical task, highlighting which tasks need the most sim-to-real bridging work.

**Data source file:** `reinforcement-learning/results.md` — Section: "Physical Robot Validation — dVRK Platform (5 core tasks)"

**X-axis label:** `"Success Rate (%)"`
**Y-axis label:** `"Surgical Task"`

**Data to plot (horizontal dumbbell: left dot = physical, right dot = sim, connected by line):**
| Task | Sim Success (%) | Physical Success (%) | Gap (%) |
|---|---|---|---|
| Peg Transfer | 94 | 89 | 5 |
| Needle Pickup | 91 | 82 | 9 |
| Tissue Retraction | 84 | 71 | 13 |
| Cutting | 86 | 72 | 14 |
| Suture Throw | 79 | 64 | 15 |

Annotate each connecting line with the gap percentage. Color the gap line by severity: <10% green, 10–13% yellow, >13% red.

Also from `reinforcement-learning/results.md` — Failure Mode Analysis (n=200 trials): Grasp slip 28%, Position error 22%, Timing error 18%, Visual confusion 15%, Unexpected obstacle 12%, Hardware fault 5%. Add as a small pie chart inset or text box.

**Light mode:** `plotly_white`, sim dots=blue, physical dots=dark orange, gap lines colored by severity.
**Dark mode:** `plotly_dark`, sim dots=cyan, physical dots=gold, gap lines colored by severity.

---

## Instruction 18 — Diffusion Policy vs Behavior Cloning on ORBIT-Surgical Tasks

**Chart type:** Grouped bar chart with delta annotations (`go.Bar` barmode='group')

**Title:** `"Diffusion Policy vs Behavior Cloning: ORBIT-Surgical Task Performance (dVRK)"`

**Purpose:** Demonstrate the superiority of diffusion-based policy generation over standard behavior cloning for surgical manipulation tasks, with specific improvement deltas.

**Data source file:** `generative-ai/results.md` — Section: "ORBIT-Surgical Task Suite (dVRK Platform)"

**X-axis label:** `"Surgical Task"`
**Y-axis label:** `"Success Rate (%)"`

**Data to plot:**
| Task | Diffusion Policy (%) | Behavior Cloning (%) | Improvement (%) |
|---|---|---|---|
| Peg Transfer | 94.1 | 87.3 | +6.8 |
| Needle Pickup | 89.3 | 72.1 | +17.2 |
| Needle Handover | 85.7 | 68.4 | +17.3 |
| Tissue Retraction | 82.4 | 69.8 | +12.6 |
| Suture Throw | 78.2 | 61.5 | +16.7 |

Add delta text annotations above each pair (e.g., "+17.2%"). Add horizontal dashed line at 80% labeled "Clinical Feasibility".

Also from `generative-ai/results.md` — Diffusion Policy Config: Denoising steps=15, Action chunk size=16, Observation horizon=2, Visual encoder=ResNet-18, 200 demonstrations per task, 500 training epochs, GPU=RTX 4090. Add as text box annotation.

**Light mode:** `plotly_white`, Diffusion=deep blue, BC=light gray, delta text=dark green.
**Dark mode:** `plotly_dark`, Diffusion=bright blue, BC=dim gray, delta text=lime.

---

## Instruction 19 — Generative AI Foundation Model Training Efficiency

**Chart type:** Bubble chart (`go.Scatter` mode='markers', marker size proportional to efficiency ratio)

**Title:** `"Foundation Model Training Efficiency: Synthetic vs Human Data Collection"`

**Purpose:** Visualize the massive data efficiency gains from generative AI synthetic data pipelines compared to manual human data collection, showing the transformative ROI for surgical training data.

**Data source files:**
- `generative-ai/results.md` — Section: "Training Efficiency": Synthetic = 780,000 trajectories in 11 hours (NVIDIA DGX), Equivalent human time = 6,500 hours, Data efficiency = 35x. Section: "FF-SRL High-Frequency Surgical RL": Training time 12 min vs 8 hours (40x speedup), Simulation throughput 50K vs 2K steps/sec (25x). Section: "Surgical Video Synthesis": Anatomical accuracy 89%, Instrument realism 92%, Tissue deformation plausibility 84%.
- `generative-ai/strengths.md` — Foundation model overview (GR00T N1.6, Cosmos Predict 2.5, Cosmos Reason 2)

**X-axis label:** `"Data Collection Time (hours, log scale)"`
**Y-axis label:** `"Trajectories Generated (thousands)"`

**Data to plot (bubbles):**
| Method | Time (hours) | Trajectories (K) | Efficiency Label | Bubble Size |
|---|---|---|---|---|
| Human Demonstration | 6,500 | 780 | 1x (baseline) | Small |
| Isaac Lab Synthetic (GPU) | 11 | 780 | 35x | Large |
| FF-SRL (GPU, tissue tasks) | 0.2 | 50 (per task) | 40x speedup | Medium |

Also add annotations: "NVIDIA GR00T N1.6: 94.2% novel object grasping", "Cosmos Predict 2.5: FVD 142 (24% better than prior)".

**Light mode:** `plotly_white`, Human=gray bubble, Isaac Lab=blue bubble, FF-SRL=green bubble.
**Dark mode:** `plotly_dark`, Human=dim gray, Isaac Lab=cyan, FF-SRL=lime.

---

## Instruction 20 — Needle Insertion Approach Comparison: From Hand-Coded to VLA

**Chart type:** Horizontal bar chart with adaptability color coding (`go.Bar` orientation='h')

**Title:** `"Needle Insertion: Success Rate by Control Approach (Hand-Coded → VLA)"`

**Purpose:** Trace the evolution of control approaches for the needle insertion task, showing engineers the progression from classical control to modern VLA methods and their relative capabilities.

**Data source file:** `reinforcement-learning/results.md` — Section: "Needle Insertion Approach Comparison"

**X-axis label:** `"Success Rate (%)"`
**Y-axis label:** `"Control Approach"`

**Data to plot (ordered by success rate ascending):**
| Approach | Success (%) | Adaptability |
|---|---|---|
| Hand-Coded | 65 | None |
| Model Predictive Control (MPC) | 72 | Low |
| Behavior Cloning | 78 | Low |
| RL (PPO) | 86 | High |
| Diffusion Policy | 89 | Medium |
| Vision-Language-Action (VLA) | 92 | Very High |

Color each bar by adaptability: None=gray, Low=yellow, Medium=orange, High=blue, Very High=green. Add annotations for key references: "PPO training: 2M samples, 48 GPU hrs" and "VLA: language-conditioned, zero-shot transfer" from `reinforcement-learning/results.md` — "Sample Efficiency Comparison".

**Light mode:** `plotly_white`, bars colored by adaptability, gray→yellow→orange→blue→green.
**Dark mode:** `plotly_dark`, bars in brighter shades of same scheme.
