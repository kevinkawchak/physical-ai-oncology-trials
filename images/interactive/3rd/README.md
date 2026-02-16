# 3rd Set — Interactive Visualization Scripts

Python scripts for the third set of 10 visualizations covering regulatory
compliance, privacy frameworks, and deployment readiness for physical AI
oncology trials.

## Scripts

| Script | LOC | Chart Type | Description |
|--------|-----|------------|-------------|
| `federated_learning_convergence.py` | 231 | Dual-panel Line | Federated learning convergence across 3 hospital sites |
| `multi_site_trial_dashboard.py` | 200 | Heatmap Table | Multi-site trial enrollment and quality dashboard |
| `federated_algorithm_radar.py` | 152 | Radar | FedAvg vs FedProx vs SCAFFOLD comparison |
| `fda_device_classification_tree.py` | 284 | Decision Tree | FDA AI/ML device classification pathway |
| `fda_oncology_device_distribution.py` | 172 | Stacked Bar + Pie | FDA-authorized oncology device distribution |
| `regulatory_compliance_scorecard.py` | 158 | Annotated Heatmap | IEC 62304 + FDA AI/ML + ISO 14971 checklist |
| `hipaa_phi_detection_matrix.py` | 185 | Annotated Heatmap | 18 HIPAA identifiers: confidence and risk |
| `privacy_analytics_pipeline.py` | 258 | Process Flow | Privacy-preserving analytics pipeline stages |
| `deployment_readiness_radar.py` | 218 | Radar + Table | ONNX validation and safety compliance |
| `production_readiness_tasks.py` | 182 | Horizontal Bar | Surgical task production readiness scores |

**Total: 2,040 LOC across 10 scripts**

## Running

```bash
pip install plotly kaleido
python images/interactive/3rd/federated_learning_convergence.py
```

Each script generates light and dark HTML + PNG variants.

## Static PNGs

Static exports are in `images/png/3rd/` (20 PNGs: 10 light + 10 dark).

## Interactive HTML Versions

The interactive HTML files for the 3rd set are available on Google Drive:

[View 3rd Set Interactive HTML Files](https://drive.google.com/drive/folders/1C092zdAyP3_go9fx7rj2yiCW0KhLo7er)

## Prompt Reference

See `images/prompts/3rd.md` for the human-authored instructions and AI
recommendations that guided generation of these visualizations.
