# Visualizations - Physical AI Oncology Trials

Interactive and static visualizations detailing processes, methods, mechanisms,
and metrics critical for engineers building physical AI oncology trials.

## Directory Structure

```
images/
├── README.md
├── interactive/
│   ├── 1st/                 # First set of visualizations
│   │   ├── *.py             # Generation scripts (require plotly + kaleido)
│   │   ├── *_light.html     # Light-mode interactive charts
│   │   └── *_dark.html      # Dark-mode interactive charts
│   └── 2nd/                 # Second set of visualizations
│       ├── *.py
│       ├── *_light.html
│       └── *_dark.html
└── png/
    ├── 1st/                 # High-resolution static exports (1st set)
    │   ├── *_light.png      # Light-mode renders (1920×1080 @2×)
    │   └── *_dark.png       # Dark-mode renders (1920×1080 @2×)
    └── 2nd/                 # High-resolution static exports (2nd set)
        ├── *_light.png
        └── *_dark.png
```

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

## Regenerating

Install dependencies and run any individual script:

```bash
pip install plotly kaleido
python images/interactive/1st/repo_architecture_treemap.py
python images/interactive/2nd/agentic_latency_breakdown.py
```

Or regenerate all visualizations:

```bash
for f in images/interactive/1st/*.py images/interactive/2nd/*.py; do python "$f"; done
```

## Requirements

- Python 3.10+
- plotly >= 6.0
- kaleido >= 1.0
