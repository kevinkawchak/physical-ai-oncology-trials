# Visualizations - Physical AI Oncology Trials

Interactive and static visualizations detailing processes, methods, mechanisms,
and metrics critical for engineers building physical AI oncology clinical trials.

Three sets of 10 visualizations (30 total) were produced across three Claude Code
sessions using human-authored prompts combined with AI-recommended data extraction
and chart design. Each visualization ships in both **light** and **dark** theme
variants (60 PNGs, 30 Python scripts).

## Directory Structure

```
images/
├── README.md                  ← this file
├── prompts/
│   ├── plan.md                # Master planning prompt (human-authored)
│   ├── 1st.md                 # 1st set prompt + AI recommendations
│   ├── 2nd.md                 # 2nd set prompt + AI recommendations
│   └── 3rd.md                 # 3rd set prompt + AI recommendations
├── interactive/
│   ├── 1st/                   # 1st set — 10 Python scripts
│   │   ├── README.md
│   │   └── *.py
│   ├── 2nd/                   # 2nd set — 10 Python scripts
│   │   ├── README.md
│   │   └── *.py
│   └── 3rd/                   # 3rd set — 10 Python scripts
│       ├── README.md
│       └── *.py
└── png/
    ├── 1st/                   # 1st set — 20 PNGs (10 light + 10 dark)
    ├── 2nd/                   # 2nd set — 20 PNGs (10 light + 10 dark)
    └── 3rd/                   # 3rd set — 20 PNGs (10 light + 10 dark)
```

---

## Prompt-to-Visualization Workflow

The visualization pipeline follows a human + AI collaborative workflow:

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                    HUMAN-AUTHORED PROMPTS                        │
 │                                                                  │
 │  plan.md ──► 30 instruction outlines (chart types, data goals)   │
 │  1st.md  ──► Instructions 1–10  (top portion = human prompt)     │
 │  2nd.md  ──► Instructions 11–20 (top portion = human prompt)     │
 │  3rd.md  ──► Instructions 21–30 (top portion = human prompt)     │
 └────────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │              AI RECOMMENDATIONS (Claude Code Plan Mode)          │
 │                                                                  │
 │  Claude Code reads the repository in plan mode, then:            │
 │  • Identifies relevant source files and data tables              │
 │  • Recommends chart types, axis labels, color schemes            │
 │  • Extracts metrics from results.md, configs, and Python modules │
 │  • Produces detailed per-chart specifications                    │
 │  (bottom portions of 1st.md, 2nd.md, 3rd.md)                    │
 └────────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                   GENERATION PIPELINE                            │
 │                                                                  │
 │  Python (.py) ──► Plotly figures ──► HTML (interactive)          │
 │                                  └──► PNG  (static, 1920x1080)  │
 │                                                                  │
 │  Each .py script produces:                                       │
 │    • 1 light-mode HTML + 1 dark-mode HTML                        │
 │    • 1 light-mode PNG  + 1 dark-mode PNG  (@2x resolution)      │
 └───────────────────────────────────────────────────────────────────┘
```

### Process Detail: `.py` → `.html` → `.png`

Each Python script uses Plotly to build the figure object, then:

1. **HTML export** — `fig.write_html()` produces a self-contained interactive HTML
   file with embedded Plotly.js (no server required).
2. **PNG export** — `fig.write_image()` uses the Kaleido engine to render the
   Plotly figure to a static 1920x1080 PNG at 2x scale factor.

The HTML files were generated during development and are available on
[Google Drive](https://drive.google.com/drive/folders/1C092zdAyP3_go9fx7rj2yiCW0KhLo7er)
for interactive viewing. The repository retains the Python source scripts and
PNG exports.

---

## Repository Data Used by Claude Code (Plan Mode)

During plan mode, Claude Code analyzed the full repository to extract data for
visualizations. The following source files were referenced across the 30 charts:

| Source Category | Files Read | Data Extracted |
|-----------------|-----------|----------------|
| Federation | `federated_coordinator.py`, `differential_privacy.py`, `secure_aggregation.py`, `privacy_analytics.py` | FedAvg/FedProx/SCAFFOLD algorithms, convergence metrics, DP budgets |
| Regulatory Submit | `classification_advisor.py`, `pccp_engine.py`, `iec62304_generator.py`, `clinical_evidence.py` | FDA device classification, IEC 62304 checklists, PCCP items |
| Digital Twins | `tumor_twin_pipeline.py`, `treatment_simulator.py`, examples `01`–`02` | State vectors, FOLFOX cycle dynamics, PBPK toxicity models |
| Tools | `deployment_readiness.py`, `trial_site_monitor.py` | Readiness radar scores, site monitoring thresholds |
| Privacy | `phi_detector.py`, `deidentification_pipeline.py` | 18 HIPAA identifiers, confidence scores, risk levels |
| RL Results | `reinforcement-learning/results.md` | ORBIT-Surgical benchmarks, GPU training hours, sim-to-real gaps |
| Agentic Results | `agentic-ai/results.md` | LLM comparison, latency breakdown, multi-agent cooperation |
| Generative Results | `generative-ai/results.md` | Diffusion policy vs BC, foundation model efficiency |
| Supervised/SSL Results | `supervised-learning/results.md`, `self-supervised-learning/results.md` | Production readiness scores |
| Configs | `training_config.yaml`, `format_mappings.yaml` | Domain randomization, safety limits, physics mappings |
| Frameworks | `INTEGRATION.md` (Isaac, MuJoCo, Gazebo, PyBullet) | Throughput, GPU requirements, framework scores |

---

## Visualization Significance

The 30 visualizations serve as a comprehensive visual reference for engineers
working across the entire physical AI oncology trial stack:

- **Architecture & Infrastructure** (1st set) — Repository structure, simulation
  pipelines, framework comparisons, and physics parameter mappings that orient
  engineers to the codebase and infrastructure requirements.

- **AI/ML Performance** (2nd set) — Quantitative benchmarks for RL algorithms,
  diffusion policies, agentic latency, and sim-to-real transfer that guide
  algorithm selection and compute budgeting.

- **Regulatory, Privacy & Deployment** (3rd set) — FDA classification pathways,
  compliance scorecards, federated learning convergence, PHI detection matrices,
  and production readiness assessments that map the path from research to
  clinical deployment.

---

## Conversion Efficiency & Output Metrics

### Per-Set Summary

| Metric | 1st Set | 2nd Set | 3rd Set | Total |
|--------|---------|---------|---------|-------|
| Python scripts | 10 | 10 | 10 | 30 |
| Python LOC | 1,829 | 1,786 | 2,040 | 5,655 |
| Avg LOC/script | 183 | 179 | 204 | 189 |
| HTML files generated | 20 | 20 | 20 | 60 |
| PNG files generated | 20 | 20 | 20 | 60 |
| PNG disk size | 7.7 MB | 7.1 MB | 9.2 MB | 24.0 MB |
| Prompt instructions | 10 | 10 | 10 | 30 |
| Conversion success | 10/10 | 10/10 | 10/10 | 30/30 |

### Conversion Pipeline Success Rate

```
Prompt Instructions ──► Python Scripts ──► HTML Files ──► PNG Files
      30/30                30/30             60/60         60/60
      100%                 100%              100%          100%
```

All 30 prompts were successfully converted to Python scripts. All 30 scripts
successfully generated both light and dark HTML files (60 total). All 60 HTML
renders were successfully exported to PNG (60 total). Overall pipeline
success rate: **100%** across all three sets.

### Lines of Code Detail

**1st Set — Repository Architecture & Clinical Infrastructure**

| Script | LOC | Chart Type |
|--------|-----|------------|
| `repo_architecture_treemap.py` | 192 | Treemap |
| `sim_pipeline_throughput.py` | 170 | Horizontal Bar |
| `clinical_trial_workflow.py` | 142 | Grouped Bar |
| `framework_comparison_radar.py` | 146 | Radar / Polar |
| `domain_randomization_transfer.py` | 157 | Bar + Line |
| `physics_parameter_mapping.py` | 175 | Annotated Heatmap |
| `digital_twin_state_vector.py` | 214 | Indicators + Line |
| `folfox_lab_trajectories.py` | 249 | Multi-line |
| `multi_organ_toxicity.py` | 221 | Multi-panel Line |
| `llm_model_comparison.py` | 163 | Grouped Bar + Line |

**2nd Set — AI/ML Benchmarks & Performance**

| Script | LOC | Chart Type |
|--------|-----|------------|
| `agentic_latency_breakdown.py` | 173 | Stacked Bar + Scatter |
| `multimodal_input_fusion.py` | 157 | Bar + Secondary Scatter |
| `multi_agent_surgical_cooperation.py` | 179 | Grouped Bar |
| `orbit_surgical_benchmark.py` | 168 | Horizontal Grouped Bar |
| `gpu_training_efficiency.py` | 171 | Grouped Bar (log) |
| `safety_constrained_rl.py` | 210 | Scatter + Pareto |
| `sim_to_real_transfer_gap.py` | 200 | Dumbbell |
| `diffusion_vs_behavior_cloning.py` | 178 | Grouped Bar |
| `foundation_model_training_efficiency.py` | 191 | Bubble |
| `needle_insertion_approaches.py` | 159 | Horizontal Bar |

**3rd Set — Regulatory, Privacy & Deployment Readiness**

| Script | LOC | Chart Type |
|--------|-----|------------|
| `federated_learning_convergence.py` | 231 | Dual-panel Line |
| `multi_site_trial_dashboard.py` | 200 | Heatmap Table |
| `federated_algorithm_radar.py` | 152 | Radar |
| `fda_device_classification_tree.py` | 284 | Decision Tree |
| `fda_oncology_device_distribution.py` | 172 | Stacked Bar + Pie |
| `regulatory_compliance_scorecard.py` | 158 | Annotated Heatmap |
| `hipaa_phi_detection_matrix.py` | 185 | Annotated Heatmap |
| `privacy_analytics_pipeline.py` | 258 | Process Flow |
| `deployment_readiness_radar.py` | 218 | Radar + Table |
| `production_readiness_tasks.py` | 182 | Horizontal Bar |

---

## Data Inputs by Visualization

| # | Visualization | Primary Data Inputs |
|---|--------------|---------------------|
| 1 | Repository Architecture Treemap | Line counts from 39 Python modules across 8 directories |
| 2 | Sim Pipeline Throughput | Framework throughput: Isaac 40K, MuJoCo 50K, Gazebo 1K steps/sec |
| 3 | Clinical Trial Workflow | Task times: Traditional vs Rule-Based vs Agentic (5 tasks) |
| 4 | Framework Comparison Radar | 6-dimension scores for Isaac, MuJoCo, Gazebo, PyBullet |
| 5 | Domain Randomization Transfer | 5-level randomization ablation: sim/real success + transfer ratio |
| 6 | Physics Parameter Mapping | 11 parameter equivalences: Isaac Lab ↔ MuJoCo |
| 7 | Digital Twin State Vector | 8-variable patient state: V, g, d, ANC, Cr, Hb, W, ECOG |
| 8 | FOLFOX Lab Trajectories | 7-timepoint cycle: ANC, Creatinine, Hemoglobin, Tumor Marker |
| 9 | Multi-Organ Toxicity | 5-organ PBPK model over 6 chemo cycles (CTCAE grading) |
| 10 | LLM Model Comparison | 5 LLMs × 3 task categories + latency tradeoff |
| 11 | Agentic Latency Breakdown | 3-stage pipeline: median + P95 latency (ms) |
| 12 | Multimodal Input Fusion | 4 modality combos: accuracy (%) vs latency (ms) |
| 13 | Multi-Agent Cooperation | 3 team configs: time, collisions, success rate |
| 14 | ORBIT-Surgical Benchmark | 14 surgical tasks: PPO vs SAC success rates |
| 15 | GPU Training Efficiency | 5 tasks × 3 algorithms: GPU hours to 80% success |
| 16 | Safety-Constrained RL | 4 algorithms: success vs violation Pareto frontier |
| 17 | Sim-to-Real Transfer Gap | 5 dVRK tasks: sim vs physical success + gap |
| 18 | Diffusion vs Behavior Cloning | 5 tasks: diffusion policy vs BC improvement |
| 19 | Foundation Model Training | 3 methods: data collection time vs trajectories |
| 20 | Needle Insertion Approaches | 6 approaches: hand-coded → VLA evolution |
| 21 | Federated Learning Convergence | 3 sites × 6 rounds: loss + accuracy convergence |
| 22 | Multi-Site Trial Dashboard | 4 sites × 14 metrics: enrollment + quality |
| 23 | Federated Algorithm Radar | 3 algorithms × 5 dimensions |
| 24 | FDA Device Classification Tree | 9 device types + 4 escalation factors |
| 25 | FDA Oncology Device Distribution | 6 subspecialties, 1,300+ devices |
| 26 | Regulatory Compliance Scorecard | 19 checklist items: IEC 62304 + FDA + ISO 14971 |
| 27 | HIPAA PHI Detection Matrix | 18 identifiers: confidence + risk stratification |
| 28 | Privacy Analytics Pipeline | 6-stage pipeline: raw data → publishable results |
| 29 | Deployment Readiness Radar | 6-axis radar + 7-row benchmark table |
| 30 | Production Readiness Tasks | 15 task categories: readiness scores + thresholds |

---

## 1st Set — Visualizations

| File Stem | Chart Type | Description |
|-----------|-----------|-------------|
| `repo_architecture_treemap` | Treemap | Repository module hierarchy with code volume |
| `sim_pipeline_throughput` | Horizontal Bar | Sim-to-real pipeline throughput and hardware |
| `clinical_trial_workflow` | Grouped Bar | Workflow automation time comparison |
| `framework_comparison_radar` | Radar / Polar | Physics simulation framework comparison |
| `domain_randomization_transfer` | Bar + Line | Domain randomization impact on sim-to-real |
| `physics_parameter_mapping` | Annotated Heatmap | Isaac Lab ↔ MuJoCo parameter equivalences |
| `digital_twin_state_vector` | Indicators + Line | Patient digital twin 8D state vector |
| `folfox_lab_trajectories` | Multi-line | FOLFOX cycle lab value trajectories |
| `multi_organ_toxicity` | Multi-panel Line | Cumulative toxicity over 6 chemo cycles |
| `llm_model_comparison` | Grouped Bar + Line | LLM performance for surgical robot control |

## 2nd Set — Visualizations

| File Stem | Chart Type | Description |
|-----------|-----------|-------------|
| `agentic_latency_breakdown` | Stacked Bar + Scatter | Agentic AI pipeline end-to-end latency breakdown |
| `multimodal_input_fusion` | Bar + Secondary Scatter | Multi-modal input fusion accuracy vs latency |
| `multi_agent_surgical_cooperation` | Grouped Bar | Multi-agent surgical cooperation metrics |
| `orbit_surgical_benchmark` | Horizontal Grouped Bar | ORBIT-Surgical PPO vs SAC across 14 tasks |
| `gpu_training_efficiency` | Grouped Bar (log) | GPU hours to 80% success by algorithm |
| `safety_constrained_rl` | Scatter + Pareto | Safety-constrained RL success vs violation tradeoff |
| `sim_to_real_transfer_gap` | Dumbbell | Sim-to-real transfer gap on dVRK (5 tasks) |
| `diffusion_vs_behavior_cloning` | Grouped Bar | Diffusion policy vs behavior cloning on ORBIT-Surgical |
| `foundation_model_training_efficiency` | Bubble | Foundation model synthetic vs human data efficiency |
| `needle_insertion_approaches` | Horizontal Bar | Needle insertion success by control approach |

## 3rd Set — Visualizations

| File Stem | Chart Type | Description |
|-----------|-----------|-------------|
| `federated_learning_convergence` | Dual-panel Line | Federated learning convergence across 3 hospital sites |
| `multi_site_trial_dashboard` | Heatmap Table | Multi-site trial enrollment and quality dashboard |
| `federated_algorithm_radar` | Radar | FedAvg vs FedProx vs SCAFFOLD comparison |
| `fda_device_classification_tree` | Decision Tree | FDA AI/ML device classification pathway |
| `fda_oncology_device_distribution` | Stacked Bar + Pie | FDA-authorized oncology device distribution |
| `regulatory_compliance_scorecard` | Annotated Heatmap | IEC 62304 + FDA AI/ML + ISO 14971 checklist |
| `hipaa_phi_detection_matrix` | Annotated Heatmap | 18 HIPAA identifiers: confidence and risk |
| `privacy_analytics_pipeline` | Process Flow | Privacy-preserving analytics pipeline stages |
| `deployment_readiness_radar` | Radar + Table | ONNX validation and safety compliance |
| `production_readiness_tasks` | Horizontal Bar | Surgical task production readiness scores |

---

## Regenerating

Install dependencies and run any individual script:

```bash
pip install plotly kaleido
python images/interactive/1st/repo_architecture_treemap.py
python images/interactive/2nd/agentic_latency_breakdown.py
python images/interactive/3rd/federated_learning_convergence.py
```

Or regenerate all visualizations:

```bash
for f in images/interactive/1st/*.py images/interactive/2nd/*.py images/interactive/3rd/*.py; do
  python "$f"
done
```

## Requirements

- Python 3.10+
- plotly >= 6.0
- kaleido >= 1.0

## Interactive HTML Versions

The interactive HTML files (60 total) are available for viewing on Google Drive:

[View Interactive Visualizations on Google Drive](https://drive.google.com/drive/folders/1C092zdAyP3_go9fx7rj2yiCW0KhLo7er)
