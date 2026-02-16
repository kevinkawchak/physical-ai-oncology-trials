# 1st Set — Interactive Visualization Scripts

Python scripts for the first set of 10 visualizations covering repository
architecture and clinical infrastructure for physical AI oncology trials.

## Scripts

| Script | LOC | Chart Type | Description |
|--------|-----|------------|-------------|
| `repo_architecture_treemap.py` | 192 | Treemap | Repository module hierarchy with code volume |
| `sim_pipeline_throughput.py` | 170 | Horizontal Bar | Sim-to-real pipeline throughput and hardware |
| `clinical_trial_workflow.py` | 142 | Grouped Bar | Workflow automation time comparison |
| `framework_comparison_radar.py` | 146 | Radar / Polar | Physics simulation framework comparison |
| `domain_randomization_transfer.py` | 157 | Bar + Line | Domain randomization impact on sim-to-real |
| `physics_parameter_mapping.py` | 175 | Annotated Heatmap | Isaac Lab ↔ MuJoCo parameter equivalences |
| `digital_twin_state_vector.py` | 214 | Indicators + Line | Patient digital twin 8D state vector |
| `folfox_lab_trajectories.py` | 249 | Multi-line | FOLFOX cycle lab value trajectories |
| `multi_organ_toxicity.py` | 221 | Multi-panel Line | Cumulative toxicity over 6 chemo cycles |
| `llm_model_comparison.py` | 163 | Grouped Bar + Line | LLM performance for surgical robot control |

**Total: 1,829 LOC across 10 scripts**

## Running

```bash
pip install plotly kaleido
python images/interactive/1st/repo_architecture_treemap.py
```

Each script generates light and dark HTML + PNG variants.

## Static PNGs

Static exports are in `images/png/1st/` (20 PNGs: 10 light + 10 dark).

## Interactive HTML Versions

The interactive HTML files for the 1st set are available on Google Drive:

[View 1st Set Interactive HTML Files](https://drive.google.com/drive/folders/1C092zdAyP3_go9fx7rj2yiCW0KhLo7er)

## Prompt Reference

See `images/prompts/1st.md` for the human-authored instructions and AI
recommendations that guided generation of these visualizations.
