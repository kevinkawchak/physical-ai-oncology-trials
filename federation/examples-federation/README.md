# Federation Examples

Progressive examples demonstrating multi-site federated oncology trial coordination, from basic 2-site federation to complex 8-site multi-cancer-type coordination.

## Examples

| # | File | Description |
|---|------|-------------|
| 01 | `01_basic_two_site.py` | Minimal 2-site federation with FedAvg aggregation on a tumor classifier |
| 02 | `02_differential_privacy.py` | Configurable epsilon/delta budgets, Gaussian/Laplacian noise, gradient clipping |
| 03 | `03_secure_aggregation.py` | Simulated secure multi-party computation with pairwise masking and verification |
| 04 | `04_enrollment_sync.py` | Multi-site enrollment coordination with stratified randomization and conflict resolution |
| 05 | `05_data_harmonization.py` | Cross-site DICOM normalization, ICD-10/SNOMED CT/LOINC vocabulary mapping, FHIR R4 resources |
| 06 | `06_full_consortium.py` | Full 8-site multi-cancer consortium combining all federation capabilities |

## Quick Start

```bash
# Run any example directly
python federation/examples-federation/01_basic_two_site.py

# Run the full consortium demonstration
python federation/examples-federation/06_full_consortium.py
```

## Dependencies

All examples require only core dependencies:
- Python 3.10+
- NumPy 1.24.0+
- SciPy 1.11.0+ (for privacy analytics)

No external services, GPU, or network connectivity required — all multi-site communication is simulated in-process.

## Progression

1. **Example 01** introduces the federated coordinator and FedAvg aggregation
2. **Example 02** adds differential privacy with configurable epsilon budgets
3. **Example 03** demonstrates secure aggregation preventing data reconstruction
4. **Example 04** covers enrollment synchronization with conflict resolution
5. **Example 05** addresses data interoperability (DICOM, FHIR, clinical vocabularies)
6. **Example 06** combines everything into a production-representative consortium scenario

## License

MIT — See repository root LICENSE for details.
