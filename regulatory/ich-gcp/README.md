# ICH E6(R3) GCP Compliance for Physical AI Oncology Trials

*Good Clinical Practice verification and audit tools for AI-enabled trials*

**Version**: 1.0.0
**Last Updated**: February 2026

---

## Overview

ICH E6(R3), adopted January 2025 and effective in the U.S. as of September 2025, represents the first major GCP revision to explicitly address digital technologies and AI in clinical trials. This module provides compliance verification tools to help trial teams meet the updated requirements.

**Key Changes in E6(R3) for AI-Enabled Trials**:
- Risk-Based Quality Management (RBQM) replaces comprehensive SDV
- Digital technology provisions for electronic records, remote monitoring
- AI/ML tool requirements: justify use, document validation, supervise
- Data governance as shared sponsor/investigator responsibility
- Broader "service provider" category (replaces "CRO")
- Annex 2 (expected 2026) will address decentralized and adaptive trials

---

## E6(R3) Structure

| Section | Content | AI Relevance |
|---------|---------|--------------|
| General Principles | Overarching GCP principles | Risk-based approach applies to AI tools |
| Annex 1 | Implementation guidance | Digital technology, data integrity |
| Annex 2 (forthcoming) | Decentralized/pragmatic trials | Telemedicine, wearables, adaptive designs |

---

## Quick Start

```python
from regulatory.ich_gcp.gcp_compliance_checker import GCPComplianceChecker

checker = GCPComplianceChecker(
    guideline_version="E6_R3",
    jurisdiction="us_fda"
)

report = checker.verify_compliance(
    protocol_path="protocols/NSCLC_AI_surgical_v3.0.pdf",
    trial_master_file="tmf/",
    check_categories=["digital_technology_provisions", "data_governance"]
)

print(f"Compliance score: {report.overall_score:.1f}%")
```

---

## References

- [ICH E6(R3) Final Guideline](https://database.ich.org/sites/default/files/ICH_E6(R3)_Step4_FinalGuideline_2025_0106.pdf) (Jan 2025)
- [FDA Adoption of ICH E6(R3)](https://acrpnet.org/2025/09/16/fda-publishes-ich-e6r3-what-it-means-for-u-s-clinical-trials) (Sep 2025)
- [Sidley Austin: Key Takeaways for Sponsors](https://www.sidley.com/en/insights/newsupdates/2025/12/us-fdas-adoption-of-ich-e6r3-good-clinical-practice-key-takeaways-for-sponsors-and-investigators)

---

*This module is part of the Regulatory Framework for Physical AI Oncology Trials.*
