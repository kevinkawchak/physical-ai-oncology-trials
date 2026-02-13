# Regulatory Submission Automation Examples

Progressive examples demonstrating FDA regulatory submission document generation for AI/ML-enabled oncology devices, from basic Pre-Sub packages to complete multi-component regulatory strategies.

## Examples

| # | File | Description |
|---|------|-------------|
| 01 | `01_presub_package.py` | FDA Pre-Sub meeting request package with device description, AI models, and questions |
| 02 | `02_pccp_plan.py` | Predetermined Change Control Plan with modification boundaries and V&V protocols |
| 03 | `03_classification.py` | 510(k)/De Novo/PMA pathway decision support for multiple device types |
| 04 | `04_iec62304_docs.py` | IEC 62304 lifecycle documentation (SDP, SRS, SAD, risk analysis) |
| 05 | `05_clinical_evidence.py` | Clinical evidence report with benchmarks, subgroup analysis, and claims |
| 06 | `06_full_submission.py` | Complete regulatory strategy combining all components end-to-end |

## Quick Start

```bash
# Run any example directly
python regulatory-submit/examples-regulatory-submit/01_presub_package.py

# Run the full submission strategy
python regulatory-submit/examples-regulatory-submit/06_full_submission.py
```

## Dependencies

All examples require only core dependencies:
- Python 3.10+

No external packages, APIs, or network connectivity required — all document generation uses Python standard library only.

## Progression

1. **Example 01** introduces Pre-Sub package generation with FDA question templates
2. **Example 02** demonstrates PCCP authoring with modification boundary customization
3. **Example 03** provides pathway classification for multiple device profiles
4. **Example 04** generates IEC 62304 lifecycle documents with risk analysis
5. **Example 05** builds clinical evidence reports with statistical analysis
6. **Example 06** combines everything into a complete De Novo submission strategy

## License

MIT — See repository root LICENSE for details.
