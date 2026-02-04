# Data Use Agreement Templates for AI Oncology Research

*Standardized DUA generation for multi-site clinical trial data sharing*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

Multi-site AI oncology trials require formal Data Use Agreements before sharing patient data, model training datasets, or AI-generated outputs. DUAs are legally mandated by HIPAA (45 CFR 164.514(e)) for Limited Data Sets and are best practice for all inter-institutional research data exchanges. This module generates standardized DUA documents that incorporate HIPAA, GDPR, and institutional requirements.

**Key Capabilities**:
- Template generation for common AI research data sharing scenarios
- HIPAA Limited Data Set DUA compliance (45 CFR 164.514(e))
- Multi-jurisdiction support (U.S. HIPAA, EU GDPR, cross-border)
- Institutional customization with required and optional provisions
- Automated security requirement specification based on data sensitivity
- Integration with IRB and compliance office workflows

---

## DUA Template Types

| Template | Use Case | Required By | Key Provisions |
|----------|----------|-------------|----------------|
| Limited Data Set | Sharing data with dates, geography | HIPAA 45 CFR 164.514(e) | Permitted uses, re-identification prohibition |
| De-identified Data | Sharing Safe Harbor de-identified data | Best practice | Data handling, publication rights |
| Multi-site AI Training | Federated or pooled model training | Institutional policy | Model ownership, IP rights, data retention |
| Genomic Data | Sequencing and variant data sharing | GA4GH framework | Consent scope, re-contact provisions |
| Imaging Data | DICOM/NIfTI imaging datasets | DICOM/institutional | De-identification verification, storage |

---

## Quick Start

```python
from privacy.dua_templates.dua_generator import DUAGenerator

generator = DUAGenerator(
    template="multi_site_ai_research",
    jurisdiction="us_hipaa"
)

dua = generator.generate(
    data_provider="Memorial Sloan Kettering Cancer Center",
    data_recipient="Physical AI Oncology Consortium",
    data_description="De-identified CT imaging and treatment outcomes",
    permitted_uses=["model_training", "validation", "publication"],
    retention_period_years=7,
    security_requirements=["encryption_at_rest", "encryption_in_transit", "mfa"]
)

dua.export("agreements/msk_consortium_dua_2026.pdf")
```

---

## References

- [HHS Policy for Common DUA Structure](https://www.hhs.gov/web/governance/digital-strategy/it-policy-archive/hhs-policy-common-data-use-agreement-structure-repository.html) (Feb 2025)
- [CDC Core DUA Initiative](https://www.cdc.gov/data-interoperability/php/use-agreement/index.html) (Dec 2025)
- [MIT Model DUA Guide](https://admindatahandbook.mit.edu/book/v1.0-rc4/dua.html)
- [Stanford DUA FAQs](https://privacy.stanford.edu/other-resources/data-use-agreement-dua-faqs)
- [GA4GH Framework for Responsible Sharing](https://www.ga4gh.org/framework/)

---

*This module is part of the Privacy Framework for Physical AI Oncology Trials.*
