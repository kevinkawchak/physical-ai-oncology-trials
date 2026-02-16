## Prompt 1st.md - Claude Code Opus 4.6 02/14/26

Based on the following 10 sets of instructions, create a new directory in kevinkawchak/physical-ai-oncology-trials under main labelled “images” with a readme, and subdirectories “png” and “interactive”. Using Python Plotly or similar approaches, create high quality and comprehensive interactive visualization scripts that correspond with easy access to view the scripts in html in both light mode and dark mode for “interactive” that will also output 20 high resolution .png diagrams/charts, etc. in “png”. It is important that visualizations be self standing, as some less effective visualizations will be removed by the user if ineffective. 

The visualizations detail processes, methods, mechanisms, and metrics critical for engineers building physical AI oncology trials. It is important not to use file name number prefixes such as 1 or 10, as some images may be removed. Use the following 10 sets of instructions as a guide only, making sure the accuracy and relevance of recent data and effectiveness of visualizations come through extensive analysis for correct context of the current version of the repository. Be sure to fix and address errors that would cause failed checks for the pull request (the most prior PR made sure the Python environment is correct to avoid the following error during final checks): “3 failing checks 
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... “ When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Do not make changes to changelog or release notes.


## Instruction 1 — Repository Module Architecture Treemap

**Chart type:** Plotly Treemap (`go.Treemap`)

**Title:** `"Physical AI Oncology Trials — Repository Module Architecture (v1.0.0)"`

**Purpose:** Show engineers the full repository module hierarchy with relative code volume so they understand where the bulk of the implementation lives and how modules relate.

**Data source files (read these to verify line counts):**
- `federation/federated_coordinator.py` — 631 lines
- `federation/differential_privacy.py` — ~550 lines
- `federation/secure_aggregation.py` — 556 lines
- `federation/site_enrollment.py` — 630 lines
- `federation/data_harmonization.py` — ~600 lines
- `federation/consortium_reporting.py` — ~600 lines
- `federation/privacy_analytics.py` — 631 lines
- `regulatory-submit/presub_generator.py` — 817 lines
- `regulatory-submit/pccp_engine.py` — 706 lines
- `regulatory-submit/iec62304_generator.py` — 891 lines
- `regulatory-submit/classification_advisor.py` — 752 lines
- `regulatory-submit/clinical_evidence.py` — 729 lines
- `regulatory-submit/audit_trail.py` — 728 lines
- `tools/deployment-readiness/deployment_readiness.py` — ~700 lines
- `tools/dose-calculator/dose_calculator.py` — ~650 lines
- `tools/trial-site-monitor/trial_site_monitor.py` — ~650 lines
- `tools/dicom-inspector/dicom_inspector.py` — ~600 lines
- `tools/sim-job-runner/sim_job_runner.py` — ~500 lines
- `digital-twins/patient-modeling/tumor_twin_pipeline.py` — ~600 lines
- `digital-twins/treatment-simulation/treatment_simulator.py` — ~550 lines
- `digital-twins/clinical-integration/clinical_dt_interface.py` — ~500 lines
- `unification/simulation_physics/isaac_mujoco_bridge.py` — ~500 lines
- `unification/simulation_physics/urdf_sdf_mjcf_converter.py` — ~450 lines
- `unification/agentic_generative_ai/unified_agent_interface.py` — ~500 lines
- `unification/cross_platform_tools/framework_detector.py` — ~400 lines
- `unification/cross_platform_tools/validation_suite.py` — ~450 lines
- `privacy/phi-pii-management/phi_detector.py` — ~600 lines
- `privacy/de-identification/deidentification_pipeline.py` — ~550 lines
- `privacy/access-control/access_control_manager.py` — ~500 lines
- `privacy/breach-response/breach_response_protocol.py` — ~450 lines
- `privacy/dua-templates/dua_generator.py` — ~400 lines
- `regulatory/fda-compliance/fda_submission_tracker.py` — ~550 lines
- `regulatory/irb-management/irb_protocol_manager.py` — ~500 lines
- `regulatory/ich-gcp/gcp_compliance_checker.py` — ~500 lines
- `regulatory/regulatory-intelligence/regulatory_tracker.py` — ~450 lines
- `q1-2026-standards/objective-1-bidirectional-conversion/isaac_to_mujoco_pipeline.py` — ~500 lines
- `q1-2026-standards/objective-1-bidirectional-conversion/mujoco_to_isaac_pipeline.py` — ~450 lines
- `q1-2026-standards/objective-2-robot-model-repository/model_validator.py` — ~400 lines
- `q1-2026-standards/objective-3-validation-benchmark/benchmark_runner.py` — ~500 lines

**Treemap hierarchy:** Root = "physical-ai-oncology-trials"; Level 1 = top-level directories (`federation`, `regulatory-submit`, `tools`, `digital-twins`, `unification`, `privacy`, `regulatory`, `q1-2026-standards`, `examples`, `examples-new`, `agentic-ai/examples-agentic-ai`, `tests`); Level 2 = individual Python files. Size = line count. Color = line count (continuous color scale).

**Labels:** Each tile shows module name and line count.

**Light mode:** `plotly_white` template, blue-to-teal continuous color scale.
**Dark mode:** `plotly_dark` template, cyan-to-magenta continuous color scale.

---

## Instruction 2 — Multi-Framework Simulation Pipeline with Throughput Metrics

**Chart type:** Plotly horizontal bar chart with annotation arrows (`go.Bar` orientation='h' + `go.layout.Annotation` arrows between stages)

**Title:** `"Sim-to-Real Pipeline: Framework Throughput & Hardware Requirements"`

**Subtitle/annotation:** `"Isaac Lab 2.3.1 → MuJoCo 3.4.0 → Gazebo Ionic 10.0.0 → Physical Hardware"`

**Purpose:** Visualize the four-stage simulation-to-deployment pipeline, each stage's throughput in steps/second, and the GPU/hardware requirements so engineers know what infrastructure to provision.

**Data source files:**
- `frameworks/nvidia-isaac/INTEGRATION.md` — Isaac Lab: ~40,000 steps/sec with 4096 parallel envs on RTX 4090; min GPU RTX 3080, recommended RTX 4090/L40; min 32 GB RAM, 100 GB SSD
- `frameworks/mujoco/INTEGRATION.md` — MuJoCo MJX: ~50,000 steps/sec with 4096 envs on GPU; CPU-only: 1,000–2,000 steps/sec; min 8 GB VRAM (MJX), 16 GB RAM
- `frameworks/gazebo/INTEGRATION.md` — Gazebo Ionic: native ROS 2 Jazzy integration; real-time factor ~1x for single env
- `unification/simulation_physics/isaac_mujoco_bridge.py` — Bridge implementation between Isaac Lab and MuJoCo
- `configs/training_config.yaml` — `sim_to_real.validation_stages: [isaac_to_mujoco, mujoco_to_highfidelity, highfidelity_to_real]`, `min_sim_success: 0.90`, `min_transfer_rate: 0.80`

**X-axis label:** `"Throughput (steps/second)"`
**Y-axis label:** `"Pipeline Stage"`

**Data to plot (4 bars):**
| Pipeline Stage | Throughput (steps/sec) | GPU Requirement | Annotation |
|---|---|---|---|
| Stage 1: Isaac Lab 2.3.1 (GPU RL Training) | 40,000 | RTX 4090 / L40 / Blackwell | 4096 parallel envs |
| Stage 2: MuJoCo 3.4.0 (Physics Validation) | 50,000 | CUDA 12.x (MJX) or CPU | <1% dynamics deviation target |
| Stage 3: Gazebo Ionic (ROS 2 Integration) | 1,000 | CPU (real-time) | Native ROS 2 Jazzy |
| Stage 4: Physical Hardware (Deployment) | N/A (real-time) | Jetson AGX / edge | ONNX opset 17, FP16 |

Add annotation arrows between each bar showing the bridge/converter used:
- Isaac→MuJoCo: `isaac_mujoco_bridge.py`
- MuJoCo→Gazebo: `urdf_sdf_mjcf_converter.py`
- Gazebo→Physical: `ros2_surgical_deployment.py` (from `examples-new/03_ros2_surgical_deployment.py`)

**Light mode:** `plotly_white`, bars in graduated blue shades.
**Dark mode:** `plotly_dark`, bars in graduated cyan shades.

---

## Instruction 3 — Clinical Trial Workflow Time Comparison: Traditional vs Rule-Based vs Agentic

**Chart type:** Grouped bar chart (`go.Bar`, barmode='group', 3 groups per task)

**Title:** `"Clinical Trial Workflow Automation: Time per Task (minutes)"`

**Purpose:** Show engineers the dramatic time savings when moving from traditional manual workflows to rule-based automation to full agentic AI, quantifying the ROI of the agentic-ai module.

**Data source file:** `agentic-ai/results.md` — Section: "Clinical Trial Workflow Comparison"

**X-axis label:** `"Workflow Task"`
**Y-axis label:** `"Time (minutes)"`

**Data to plot:**
| Task | Traditional (min) | Rule-Based (min) | Agentic AI (min) | Improvement |
|---|---|---|---|---|
| Patient Screening | 45 | 15 | 5 | 89% |
| Scheduling | 30 | 12 | 3 | 90% |
| Documentation | 60 | 25 | 8 | 87% |
| Adverse Event Reporting | 40 | 20 | 7 | 83% |
| Protocol Compliance Check | 20 | 8 | 2 | 90% |

Add text annotations above each Agentic bar showing the % improvement (e.g., "89%↓").

**Legend labels:** "Traditional (Manual)", "Rule-Based Automation", "Agentic AI (Claude/GPT-4)"

**Light mode:** `plotly_white`, Traditional=light gray, Rule-Based=steel blue, Agentic=deep green.
**Dark mode:** `plotly_dark`, Traditional=dim gray, Rule-Based=dodger blue, Agentic=lime green.

---

## Instruction 4 — Simulation Framework Comparison Radar Chart

**Chart type:** Radar/polar chart (`go.Scatterpolar`, fill='toself')

**Title:** `"Physics Simulation Framework Comparison for Surgical Robotics"`

**Purpose:** Give engineers an at-a-glance comparison of the four simulation frameworks used in the repository across key engineering dimensions so they can choose the right tool for each stage.

**Data source files:**
- `frameworks/nvidia-isaac/INTEGRATION.md` — Isaac Lab 2.3.1: 4096+ parallel envs, GPU-native, domain randomization, Isaac Sim 5.0 rendering
- `frameworks/mujoco/INTEGRATION.md` — MuJoCo 3.4.0: reference-precision physics, MJX JAX GPU backend, <1% dynamics deviation
- `frameworks/gazebo/INTEGRATION.md` — Gazebo Ionic 10.0.0: native ROS 2 Jazzy, sensor simulation, real-time
- `frameworks/pybullet/INTEGRATION.md` — PyBullet 3.2.5: rapid prototyping, easy API, limited parallelism
- `unification/simulation_physics/isaac_mujoco_bridge.py` — Bridge code
- `q1-2026-standards/objective-1-bidirectional-conversion/format_mappings.yaml` — Parameter equivalences

**Radar axes (6 dimensions, each scored 1–10):**
| Dimension | Isaac Lab | MuJoCo | Gazebo | PyBullet |
|---|---|---|---|---|
| GPU Throughput | 9 | 9 (MJX) | 3 | 4 |
| Physics Fidelity | 8 | 10 | 7 | 6 |
| Parallel Environments | 10 | 9 (MJX) | 2 | 3 |
| ROS 2 Integration | 7 | 4 | 10 | 3 |
| Ease of Setup | 5 | 8 | 6 | 9 |
| Domain Randomization | 10 | 7 | 5 | 4 |

**Axis label:** Each spoke labeled with the dimension name.

**Light mode:** `plotly_white`, each framework a distinct solid color (blue, red, green, orange) with 0.2 opacity fill.
**Dark mode:** `plotly_dark`, each framework a distinct bright color (cyan, magenta, lime, gold) with 0.15 opacity fill.

---

## Instruction 5 — Domain Randomization Impact on Sim-to-Real Transfer

**Chart type:** Grouped bar chart (`go.Bar`, barmode='group') with line overlay for transfer ratio

**Title:** `"Domain Randomization Ablation: Needle Insertion Sim-to-Real Transfer"`

**Purpose:** Quantify how each type of domain randomization affects sim-to-real transfer, guiding engineers on which randomization parameters to prioritize for oncology tasks.

**Data source files:**
- `reinforcement-learning/results.md` — Section: "Domain Randomization Ablation Study (Needle Insertion Task)"
- `configs/training_config.yaml` — Section: `domain_randomization` — lists all randomization parameters: `lighting_intensity: [0.7, 1.3]`, `camera_noise: 0.02`, `texture_augmentation: true`, `tissue_stiffness: [0.5, 2.0]`, `friction: [0.3, 0.8]`, `damping: [0.8, 1.2]`, `position_noise: 5.0` mm, `orientation_noise: 5.0` deg, `scale_variation: [0.9, 1.1]`, `action_delay: [0, 20]` ms, `action_noise_std: 0.01`
- `generative-ai/results.md` — Section: "Oncology-Specific Domain Randomization": needle insertion 85% sim-to-real, tissue manipulation 78%

**X-axis label:** `"Randomization Level"`
**Y-axis label (left):** `"Success Rate (%)"`
**Y-axis label (right):** `"Transfer Ratio (%)"`

**Data to plot (bars = sim and real success; line = transfer ratio):**
| Randomization Level | Sim Success (%) | Real Success (%) | Transfer Ratio (%) |
|---|---|---|---|
| None | 95 | 52 | 55 |
| Visual Only | 94 | 68 | 72 |
| Physics Only | 93 | 65 | 70 |
| Visual + Physics | 92 | 78 | 85 |
| Full (+ action delay) | 91 | 82 | 90 |

Add horizontal dashed line at 80% for the `min_transfer_rate` threshold from `configs/training_config.yaml`.

**Light mode:** `plotly_white`, Sim=light blue bars, Real=dark blue bars, Transfer line=red.
**Dark mode:** `plotly_dark`, Sim=light cyan bars, Real=bright blue bars, Transfer line=orange.

---

## Instruction 6 — Physics Parameter Mapping: Isaac Lab ↔ MuJoCo Equivalences

**Chart type:** Heatmap with text annotations (`go.Heatmap` or `ff.create_annotated_heatmap`)

**Title:** `"Isaac Lab ↔ MuJoCo Physics Parameter Mapping (Bidirectional Conversion)"`

**Purpose:** Show engineers the exact parameter equivalences between Isaac Lab and MuJoCo so they understand the bridge conversion accuracy and know which parameters require special handling.

**Data source files:**
- `q1-2026-standards/objective-1-bidirectional-conversion/format_mappings.yaml` — Contains the full parameter mapping table including: contact stiffness, contact damping, friction cone, solver iterations, gravity, timestep, joint damping, joint stiffness, joint limits, actuator gains
- `unification/simulation_physics/physics_parameter_mapping.yaml` — Additional parameter equivalences
- `q1-2026-standards/objective-1-bidirectional-conversion/physics_equivalence_tests.py` — Validation tests with <1% deviation target

**Heatmap structure:** Rows = parameter categories (Contact, Joint, Solver, Environment, Actuator); Columns = sub-parameters. Cell values = conversion complexity score (1=direct mapping, 2=formula transform, 3=approximation required). Cell text annotations = the actual conversion formula or note.

**Data to plot:**
| Parameter Category | Sub-Parameter | Isaac Lab Name | MuJoCo Name | Conversion | Complexity |
|---|---|---|---|---|---|
| Contact | Stiffness | `contact_stiffness` | `solref[0]` | Formula: `solref = -stiffness/damping` | 2 |
| Contact | Damping | `contact_damping` | `solref[1]` | Formula: `solref[1] = -2*sqrt(stiffness*mass)` | 2 |
| Contact | Friction | `static_friction, dynamic_friction` | `friction[0:2]` | Direct + cone model | 2 |
| Joint | Damping | `joint_damping` | `damping` | Direct mapping | 1 |
| Joint | Stiffness | `joint_stiffness` | `stiffness` | Direct mapping | 1 |
| Joint | Limits | `lower_limit, upper_limit` | `range[0:2]` | Direct mapping | 1 |
| Solver | Iterations | `solver_iterations` | `iterations` | Direct mapping | 1 |
| Solver | Timestep | `physics_dt: 0.002` | `timestep: 0.002` | Direct mapping | 1 |
| Environment | Gravity | `gravity: [0,0,-9.81]` | `gravity: [0,0,-9.81]` | Direct mapping | 1 |
| Actuator | Gains | `stiffness, damping` | `kp, kv` | Scale transform | 2 |
| Actuator | Force limits | `effort_limit` | `forcerange` | Direct mapping | 1 |

**Color scale:** 1 (green) = direct mapping, 2 (yellow) = formula transform, 3 (red) = approximation.

**Light mode:** `plotly_white`, green-yellow-red color scale with dark text.
**Dark mode:** `plotly_dark`, teal-amber-crimson color scale with white text.

---

## Instruction 7 — Digital Twin Patient State Vector Dashboard

**Chart type:** Combined indicator cards + line chart (`make_subplots` with `go.Indicator` type="number+delta" for 8 cards in top rows, `go.Scatter` line chart in bottom row)

**Title:** `"Patient-Specific Digital Twin: 8-Dimensional State Vector (FOLFOX Colorectal Cancer)"`

**Purpose:** Visualize the complete patient digital twin state vector that drives treatment simulation, showing engineers the 8 tracked variables, their baseline values, physiological ranges, and measurement uncertainties.

**Data source files:**
- `digital-twins/examples-twins/01_realtime_dt_synchronization.py` — Contains the full `STATE_CONFIG` dict with variable names, baselines, ranges, units, and process noise; also `OBSERVATION_CONFIG` with measurement noise values; also `DYNAMICS_PARAMS` with growth/recovery rates
- `digital-twins/patient-modeling/tumor_twin_pipeline.py` — TumorTwin framework integration, tumor growth models (reaction-diffusion, logistic, Gompertz)
- `digital-twins/treatment-simulation/treatment_simulator.py` — Treatment response simulation

**Data for the 8 indicator cards:**
| Variable | Symbol | Baseline | Range | Unit | Process Noise (σ/day) |
|---|---|---|---|---|---|
| Tumor Volume | V | 18.5 | 0.01–100 | cm³ | 0.5 |
| Growth Rate | g | 0.01 | -0.1 to +0.1 | /day | 0.001 |
| Drug Effect | d | 0.0 | 0–1 | dimensionless | 0.01 |
| Neutrophils (ANC) | ANC | 4.5 | 0.1–10 | ×10⁹/L | 0.2 |
| Creatinine | Cr | 0.85 | 0.3–4.0 | mg/dL | 0.01 |
| Hemoglobin | Hb | 14.1 | 4.0–18.0 | g/dL | 0.05 |
| Weight | W | 78 | 40–150 | kg | 0.1 |
| ECOG Status | E | 1.0 | 0–4 | score | 0.05 |

**For the line chart:** Plot a simulated 21-day trajectory for Tumor Volume (V) showing exponential baseline growth rate of 0.01/day with drug effect (d) applied on Day 1 (FOLFOX cycle). Tumor marker sensitivity = 3.0 ng/mL per cm³. Show the volume declining from 18.5 cm³ to ~17.5 cm³ by day 10, then stabilizing.

**Light mode:** `plotly_white`, indicator cards with blue headers, green/red delta arrows.
**Dark mode:** `plotly_dark`, indicator cards with cyan headers, lime/red delta arrows.

---

## Instruction 8 — FOLFOX Chemotherapy Cycle: Lab Value Trajectories Over 21 Days

**Chart type:** Multi-line chart with markers and shaded nadir region (`go.Scatter` with `fill='tozeroy'` for nadir band)

**Title:** `"FOLFOX Cycle Simulation: Lab Value Trajectories (21-Day Cycle)"`

**Subtitle:** `"85 mg/m² oxaliplatin, BSA 1.85 m², Total dose 157.25 mg"`

**Purpose:** Show engineers the expected temporal dynamics of key lab values during a single FOLFOX chemotherapy cycle as predicted by the digital twin, including the hematologic nadir window and recovery phases.

**Data source file:** `digital-twins/examples-twins/01_realtime_dt_synchronization.py` — Section containing `FOLFOX_CYCLE` simulation parameters and the cycle walkthrough comments describing Day 1, Day 3, Day 7 (nadir), Day 14, Day 21.

**X-axis label:** `"Day of Cycle"`
**Y-axis label (left):** `"Lab Value"`
**Y-axis label (right):** `"Tumor Marker (ng/mL)"`

**Data to plot (4 lines over 21 days):**
| Day | ANC (×10⁹/L) | Creatinine (mg/dL) | Hemoglobin (g/dL) | Tumor Marker (ng/mL) |
|---|---|---|---|---|
| 0 (baseline) | 4.5 | 0.85 | 14.1 | 48 |
| 1 (infusion) | 4.5 | 0.85 | 14.1 | 48 |
| 3 | 3.8 | 0.90 | 13.5 | 46 |
| 7 (nadir) | 1.8 | 0.95 | 12.8 | 44 |
| 10 | 2.2 | 0.92 | 13.0 | 42 |
| 14 | 2.9 | 0.90 | 13.2 | 43 |
| 21 (recovery) | 4.0 | 0.86 | 13.9 | 44 |

Add shaded vertical band from Day 5–Day 12 labeled "Nadir Window". Add horizontal dashed line at ANC = 1.0 labeled "Grade 4 Neutropenia Threshold". Add horizontal dashed line at ANC = 1.5 labeled "Grade 3 Threshold".

**Dynamics parameters from source:** ANC recovery rate = 0.15/day, Creatinine clearance = 0.05/day, Hb recovery = 0.02/day, ANC baseline = 4.5, Creatinine baseline = 1.0, Hb baseline = 13.0, Tumor marker sensitivity = 3.0 ng/mL per cm³.

**Light mode:** `plotly_white`, ANC=blue, Creatinine=red, Hemoglobin=green, Tumor Marker=purple.
**Dark mode:** `plotly_dark`, ANC=cyan, Creatinine=salmon, Hemoglobin=lime, Tumor Marker=violet.

---

## Instruction 9 — Multi-Organ Toxicity Accumulation Over 6 Chemotherapy Cycles

**Chart type:** Multi-panel line chart with CTCAE grade threshold bands (`make_subplots` with 5 rows, shared x-axis)

**Title:** `"Multi-Organ Toxicity Twin: Cumulative Toxicity Over 6 FOLFOX Cycles"`

**Purpose:** Show engineers how the digital twin predicts cumulative organ toxicity across 6 chemotherapy cycles, with CTCAE grading thresholds, so they understand when dose modifications are triggered.

**Data source file:** `digital-twins/examples-twins/02_multi_organ_toxicity_twin.py` — Contains full PBPK compartment model, drug-specific parameters (cisplatin, doxorubicin, oxaliplatin), organ-specific toxicity models with sensitivity coefficients, baseline values, CTCAE grade thresholds, and cycle-by-cycle simulation output.

**X-axis label (shared):** `"Cycle Number (1–6, each 21 days)"`

**5 subplots:**

**Panel 1 — Cardiac (LVEF):**
- Y-axis: `"LVEF (fraction)"`, range 0.20–0.65
- Baseline: 0.60; Sensitivity: 0.0015 decline per µg·hr/mL AUC
- CTCAE Grade 4 threshold: <0.20 or >0.20 decline from baseline
- Simulated values: Cycle 1: 0.58, C2: 0.56, C3: 0.54, C4: 0.52, C5: 0.50, C6: 0.48

**Panel 2 — Renal (GFR):**
- Y-axis: `"GFR (mL/min/1.73m²)"`, range 30–100
- Baseline: 95; Sensitivity: 0.008 decline per AUC; Hydration benefit: 30%; Irreversible fraction: 40%
- Simulated: C1: 90, C2: 85, C3: 80, C4: 76, C5: 72, C6: 69

**Panel 3 — Hepatic (Bilirubin):**
- Y-axis: `"Bilirubin (mg/dL)"`, range 0.5–3.0
- Baseline: 0.7; Liver regeneration: 2%/cycle; CTCAE Grade 3: >3× baseline (>2.1)
- Simulated: C1: 0.85, C2: 1.0, C3: 1.15, C4: 1.3, C5: 1.4, C6: 1.5

**Panel 4 — Neurological (TNSc):**
- Y-axis: `"TNSc Score (0–28)"`, range 0–20
- Baseline: 0; Sensitivity: 0.012 per AUC; Minimal recovery (cumulative irreversible)
- CTCAE Grade 2: >8, Grade 3: >14, Grade 4: >20
- Simulated: C1: 2, C2: 5, C3: 8, C4: 11, C5: 15, C6: 13 (after 25% dose reduction at C5)

**Panel 5 — Hematologic (ANC nadir):**
- Y-axis: `"ANC Nadir (×10⁹/L)"`, range 0–5
- Baseline: 5.2; Grade 4: <0.5, Grade 3: <1.0, Grade 2: <1.5
- Simulated nadir per cycle: C1: 2.1, C2: 1.8, C3: 1.5, C4: 1.2, C5: 1.0, C6: 1.3 (after dose reduction)

Add vertical dashed line between C4 and C5 labeled "Dose Reduction Triggered (25%)".

**Light mode:** `plotly_white`, each panel a distinct muted color.
**Dark mode:** `plotly_dark`, each panel a distinct bright color.

---

## Instruction 10 — LLM Model Comparison for Surgical Robot Control

**Chart type:** Grouped bar chart with average line overlay (`go.Bar` barmode='group')

**Title:** `"LLM Performance Comparison: Natural Language → Surgical Robot Actions"`

**Purpose:** Help engineers select the optimal LLM backbone for their agentic surgical system by comparing accuracy across task complexities.

**Data source file:** `agentic-ai/results.md` — Section: "LLM Model Comparison" (table with Navigation, Manipulation, Multi-Step, Average columns)

**X-axis label:** `"LLM Model"`
**Y-axis label:** `"Task Accuracy (%)"`

**Data to plot:**
| Model | Navigation (%) | Manipulation (%) | Multi-Step (%) | Average (%) |
|---|---|---|---|---|
| Claude Opus 4 | 97 | 89 | 92 | 93 |
| Claude Sonnet 4 | 95 | 86 | 89 | 90 |
| GPT-4o | 94 | 84 | 87 | 88 |
| Claude Haiku 4 | 91 | 78 | 81 | 83 |
| Llama 70B | 88 | 75 | 77 | 80 |

Additional context from `agentic-ai/results.md` — "Latency vs Quality Tradeoff": Claude Opus 4 = 350ms, Claude Sonnet 4 = 200ms (best tradeoff), Claude Haiku 4 = 80ms. Add annotation: "Sonnet 4: Best latency-quality tradeoff at 200ms".

**Legend:** "Navigation", "Manipulation", "Multi-Step", "Average" (as line with markers).

**Light mode:** `plotly_white`, bar groups in blue/green/orange shades, average line in red.
**Dark mode:** `plotly_dark`, bar groups in cyan/lime/amber shades, average line in white.

---

### Follow-up Prompts
Finish

Complete

Finalize

Skip png generations. Make sure html visualizations work. [Claude Finished Quickly and Provided png Files]

