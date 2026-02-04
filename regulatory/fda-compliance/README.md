# FDA Compliance for Physical AI Oncology Trials

*Submission tracking and documentation for AI/ML-enabled medical devices*

**Version**: 1.0.0
**Last Updated**: February 2026

---

## Overview

As of December 2025, the FDA has authorized over 1,300 AI/ML-enabled medical devices. This module helps trial teams navigate the FDA submission process for AI-enabled oncology devices, from pre-submission meetings through marketing authorization and post-market surveillance.

**Key Capabilities**:
- Submission pathway selection (510(k), De Novo, PMA, Breakthrough Device)
- Pre-submission (Q-Sub) package generation
- Documentation checklist management per Jan 2025 draft guidance
- PCCP (Predetermined Change Control Plan) preparation
- Post-market surveillance planning
- ClinicalTrials.gov registration compliance (SPIRIT-AI/CONSORT-AI)

---

## Regulatory Pathways for AI Oncology Devices

| Pathway | Typical Use | Timeline | Key Requirement |
|---------|------------|----------|-----------------|
| 510(k) | AI device with predicate | 3-6 months | Substantial equivalence demonstration |
| De Novo | Novel AI without predicate | 6-12 months | Risk assessment + special controls |
| PMA | High-risk AI device (Class III) | 12-24 months | Clinical evidence of safety/effectiveness |
| Breakthrough | Irreversibly debilitating condition | Priority review | More effective treatment for serious condition |

---

## Quick Start

```python
from regulatory.fda_compliance.fda_submission_tracker import FDASubmissionTracker

tracker = FDASubmissionTracker(
    sponsor="Physical AI Oncology Consortium",
    device_class="II"
)

submission = tracker.create_submission(
    submission_type="de_novo",
    device_name="AI-Guided Surgical Planning System",
    intended_use="AI-assisted tumor resection planning",
    ai_ml_components=["tumor_segmentation_model", "surgical_path_optimizer"]
)

checklist = tracker.generate_presub_checklist(submission)
```

---

## References

- [FDA AI/ML Device List](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [FDA OCE Oncology AI Program](https://www.fda.gov/about-fda/oncology-center-excellence/oce-oncology-artificial-intelligence-program)
- [FDA Draft Guidance: AI-Enabled Device Software Functions](https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices) (Jan 2025)
- [FDA PCCP Guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence) (Aug 2025)

---

*This module is part of the Regulatory Framework for Physical AI Oncology Trials.*
