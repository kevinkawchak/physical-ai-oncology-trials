# Visualizations — Physical AI Oncology Trials

Interactive and static visualizations detailing processes, methods, mechanisms,
and metrics critical for engineers building physical AI oncology trials.

## Directory Structure

```
images/
├── README.md
├── interactive/          # Python scripts + self-contained HTML visualizations
│   ├── *.py              # Generation scripts (require plotly + kaleido)
│   ├── *_light.html      # Light-mode interactive charts
│   └── *_dark.html       # Dark-mode interactive charts
└── png/                  # High-resolution static exports
    ├── *_light.png       # Light-mode renders (1920×1080 @2×)
    └── *_dark.png        # Dark-mode renders (1920×1080 @2×)
```

## Visualizations

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

## Regenerating

Install dependencies and run any individual script:

```bash
pip install plotly kaleido
python images/interactive/repo_architecture_treemap.py
```

Or regenerate all visualizations:

```bash
for f in images/interactive/*.py; do python "$f"; done
```

## Requirements

- Python 3.10+
- plotly >= 6.0
- kaleido >= 1.0
