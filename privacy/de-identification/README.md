# De-Identification for AI Oncology Trials

*HIPAA-compliant de-identification using Safe Harbor and Expert Determination methods*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

De-identification transforms PHI into data that cannot reasonably identify an individual, enabling AI model training and multi-site research collaboration without HIPAA constraints on the resulting dataset. This module implements both HIPAA-recognized methods (45 CFR 164.514) with oncology-specific extensions for medical imaging, genomic data, and clinical text.

**Key Capabilities**:
- Safe Harbor method: automated removal of all 18 HIPAA identifiers
- Expert Determination method: statistical re-identification risk assessment
- DICOM image de-identification with burned-in text removal
- Clinical text de-identification via NLP
- Genomic data de-identification with linkage risk analysis
- Re-identification risk scoring and residual risk reporting

---

## Methods Comparison

| Aspect | Safe Harbor | Expert Determination |
|--------|------------|---------------------|
| Regulatory basis | 45 CFR 164.514(b)(2) | 45 CFR 164.514(b)(1) |
| Approach | Remove 18 specified identifiers | Statistical/scientific analysis |
| Expertise required | Low-Medium | High (qualified expert) |
| Data utility preserved | Lower | Higher |
| Re-identification risk | Very low (if properly applied) | "Very small" (documented) |
| Oncology use case | Multi-site data sharing | Longitudinal treatment studies |

---

## Quick Start

```python
from privacy.de_identification.deidentification_pipeline import DeidentificationPipeline

# Safe Harbor de-identification
pipeline = DeidentificationPipeline(
    method="safe_harbor",
    hipaa_identifiers="all_18",
    preserve_clinical_utility=True,
    date_handling="year_only",       # Keep year, remove month/day
    age_handling="cap_at_89",        # Ages >89 grouped as 90+
    geography_handling="state_only"  # Keep state, remove sub-state
)

result = pipeline.deidentify(
    input_path="trial_data/raw/",
    output_path="trial_data/deidentified/",
    data_types=["structured_ehr", "clinical_notes", "dicom_images"]
)

print(f"Records processed: {result.records_processed}")
print(f"Re-identification risk: {result.residual_risk:.4f}")
```

---

## References

- [HHS De-Identification Guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) (reviewed Feb 2025)
- [Medical Image De-Identification](https://github.com/TIO-IKIM/medical_image_deidentification) - DICOM MRI/CT/WSI de-identification
- [NCI MIDI Project](https://datacommons.cancer.gov/news/final-report-medical-imaging-de-identification-midi-project) - Medical Imaging De-Identification benchmark
- [ARX Data Anonymization](https://github.com/arx-deidentifier/arx) - k-anonymity, l-diversity, differential privacy

---

*This module is part of the Privacy Framework for Physical AI Oncology Trials.*
