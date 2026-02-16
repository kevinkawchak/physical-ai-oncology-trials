# 2nd Set — Interactive Visualization Scripts

Python scripts for the second set of 10 visualizations covering AI/ML
benchmarks and performance metrics for physical AI oncology trials.

## Scripts

| Script | LOC | Chart Type | Description |
|--------|-----|------------|-------------|
| `agentic_latency_breakdown.py` | 173 | Stacked Bar + Scatter | Agentic AI pipeline end-to-end latency breakdown |
| `multimodal_input_fusion.py` | 157 | Bar + Secondary Scatter | Multi-modal input fusion accuracy vs latency |
| `multi_agent_surgical_cooperation.py` | 179 | Grouped Bar | Multi-agent surgical cooperation metrics |
| `orbit_surgical_benchmark.py` | 168 | Horizontal Grouped Bar | ORBIT-Surgical PPO vs SAC across 14 tasks |
| `gpu_training_efficiency.py` | 171 | Grouped Bar (log) | GPU hours to 80% success by algorithm |
| `safety_constrained_rl.py` | 210 | Scatter + Pareto | Safety-constrained RL success vs violation tradeoff |
| `sim_to_real_transfer_gap.py` | 200 | Dumbbell | Sim-to-real transfer gap on dVRK (5 tasks) |
| `diffusion_vs_behavior_cloning.py` | 178 | Grouped Bar | Diffusion policy vs behavior cloning on ORBIT-Surgical |
| `foundation_model_training_efficiency.py` | 191 | Bubble | Foundation model synthetic vs human data efficiency |
| `needle_insertion_approaches.py` | 159 | Horizontal Bar | Needle insertion success by control approach |

**Total: 1,786 LOC across 10 scripts**

## Running

```bash
pip install plotly kaleido
python images/interactive/2nd/agentic_latency_breakdown.py
```

Each script generates light and dark HTML + PNG variants.

## Static PNGs

Static exports are in `images/png/2nd/` (20 PNGs: 10 light + 10 dark).

## Interactive HTML Versions

The interactive HTML files for the 2nd set are available on Google Drive:

[View 2nd Set Interactive HTML Files](https://drive.google.com/drive/folders/1C092zdAyP3_go9fx7rj2yiCW0KhLo7er)

## Prompt Reference

See `images/prompts/2nd.md` for the human-authored instructions and AI
recommendations that guided generation of these visualizations.
