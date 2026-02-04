# Regulatory Intelligence for Physical AI Oncology Trials

*Multi-jurisdiction regulatory landscape monitoring and analysis*

**Version**: 1.0.0
**Last Updated**: February 2026

---

## Overview

The regulatory landscape for AI in oncology clinical trials is evolving rapidly across multiple jurisdictions. This module tracks regulatory developments from FDA, EMA, ICH, WHO, and other bodies, providing trial teams with current intelligence on guidance documents, enforcement actions, and emerging requirements.

---

## Jurisdictions Monitored

| Jurisdiction | Agency | Key Regulations | AI-Specific Status |
|-------------|--------|----------------|-------------------|
| United States | FDA (CDRH, CDER, OCE) | 510(k), De Novo, PMA, 21 CFR Part 11 | Draft guidance (Jan 2025) |
| European Union | EMA | MDR 2017/745, EU AI Act | AI Act phased (2025-2027) |
| International | ICH | E6(R3) GCP | Effective (Jan 2025) |
| Global | WHO | Regulatory Considerations on AI | 18 principles (2023) |
| United States | HHS/OCR | HIPAA Security Rule | NPRM (Jan 2025) |
| European Union | EC | GDPR, Clinical Trials Regulation | Biotech Act proposed (Dec 2025) |

---

## Quick Start

```python
from regulatory.regulatory_intelligence.regulatory_tracker import RegulatoryTracker

tracker = RegulatoryTracker(
    jurisdictions=["fda", "ema", "ich"],
    topics=["ai_ml_devices", "oncology", "clinical_trials"]
)

updates = tracker.get_recent_updates(days=30)
for update in updates:
    print(f"  [{update.jurisdiction}] {update.title} ({update.date})")
```

---

## References

- [FDA AI/ML Device List](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [EU AI Act Timeline](https://gardner.law/news/eu-ai-act-compliance-timeline)
- [ICH Guidelines Database](https://database.ich.org/sites/default/files/)
- [WHO AI for Health](https://www.who.int/publications/i/item/9789240078871)

---

*This module is part of the Regulatory Framework for Physical AI Oncology Trials.*
