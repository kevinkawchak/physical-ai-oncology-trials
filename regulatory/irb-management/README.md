# IRB Management for Physical AI Oncology Trials

*AI-specific IRB protocol submission, review, and compliance tracking*

**Version**: 1.0.0
**Last Updated**: February 2026

---

## Overview

There are currently no federal regulations specific to IRB review of AI in clinical research. However, significant guidance has emerged from HHS SACHRP and the MRCT Center (July 2025) establishing frameworks for IRB review of AI-enabled protocols. This module helps trial teams prepare IRB submissions that address the unique considerations of physical AI oncology trials.

**Key Capabilities**:
- AI-specific protocol preparation with algorithmic bias assessments
- Informed consent template generation with AI disclosure requirements
- Multi-site / single IRB coordination for AI-enabled trials
- Continuing review management for adaptive AI systems
- Amendment tracking for AI model updates during trials

---

## IRB Review Considerations for AI in Clinical Research

Per HHS SACHRP and the MRCT Center Framework (July 2025):

| Area | Review Requirement |
|------|--------------------|
| AI Transparency | How AI involvement is disclosed to participants |
| Algorithmic Bias | Assessment and mitigation of bias across demographics |
| Data Privacy | AI-specific re-identification risks beyond standard PHI |
| Informed Consent | Clear description of AI role in treatment decisions |
| Human Oversight | How clinicians oversee/override AI recommendations |
| Participant Safety | Safety monitoring for AI-influenced outcomes |
| Data Use | Scope of data use for AI training vs. clinical care |

---

## Quick Start

```python
from regulatory.irb_management.irb_protocol_manager import IRBProtocolManager

irb = IRBProtocolManager(
    institution="Memorial Sloan Kettering Cancer Center",
    irb_type="central"
)

protocol = irb.create_protocol(
    title="Physical AI-Guided Surgical Resection for NSCLC",
    pi_name="Dr. Sarah Chen",
    ai_components=["real-time tumor detection", "robotic instrument guidance"],
    participant_count=200
)

ai_review = irb.generate_ai_review_checklist(protocol)
```

---

## References

- [HHS SACHRP: IRB Considerations on AI](https://www.hhs.gov/ohrp/sachrp-committee/recommendations/irb-considerations-use-artificial-intelligence-human-subjects-research/index.html)
- [MRCT Center: Framework for Review of AI in Clinical Research](https://mrctcenter.org/resource/framework-for-review-of-clinical-research-involving-ai/) (Jul 2025)
- [SPIRIT-AI Extension](https://www.spirit-statement.org/) - Protocol reporting for AI trials
- [CONSORT-AI Extension](https://www.consort-statement.org/) - Results reporting for AI trials

---

*This module is part of the Regulatory Framework for Physical AI Oncology Trials.*
