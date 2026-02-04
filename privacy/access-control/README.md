# Access Control for AI Oncology Trials

*Role-based access control with 21 CFR Part 11 audit trails for clinical AI systems*

**Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

## Overview

Clinical trial data requires granular access controls that distinguish between human users (investigators, coordinators, analysts) and AI systems (training pipelines, inference engines, federated nodes). This module implements role-based access control (RBAC) with comprehensive audit logging that satisfies FDA 21 CFR Part 11 electronic records requirements and the proposed HIPAA Security Rule updates (Jan 2025 NPRM).

**Key Capabilities**:
- Role-based access with trial-specific permission profiles
- AI system access controls (model training, inference, export boundaries)
- 21 CFR Part 11 compliant audit trails with tamper detection
- Multi-factor authentication integration
- Automated access reviews and privilege expiration
- Technology asset inventory for ePHI-handling systems

---

## Role Hierarchy for Oncology AI Trials

| Role | PHI Access | De-identified Access | AI Model Access | Audit Access |
|------|-----------|---------------------|----------------|-------------|
| Principal Investigator | Full | Full | Full | Full |
| Sub-Investigator | Site-level | Full | Read | Site-level |
| Clinical Coordinator | Site-level | Full | None | Site-level |
| Data Manager | Full (with justification) | Full | Read | Full |
| Biostatistician | None | Full | Read | None |
| AI/ML Engineer | None | Full | Full | System-level |
| AI System (automated) | None | Read-only | Write outputs | Auto-logged |
| Monitor/CRA | Read (during visit) | Full | Read | Full |
| IRB Member | Summary only | Aggregate only | None | Protocol-level |

---

## Quick Start

```python
from privacy.access_control.access_control_manager import AccessControlManager

acm = AccessControlManager(
    compliance_framework="21_cfr_part_11",
    audit_enabled=True,
    mfa_required=True
)

# Define roles and assign users
acm.define_role("principal_investigator", permissions=[
    "read_phi", "write_clinical_data", "export_deidentified",
    "approve_enrollment", "view_audit_logs"
])

acm.assign_role(user_id="PI-001", role="principal_investigator")

# Check access with full audit logging
access = acm.check_access(
    user_id="PI-001",
    resource="patient_records",
    action="read_phi"
)
print(f"Access granted: {access.granted}")
```

---

## References

- [FDA 21 CFR Part 11 Q&A Guidance](https://www.fda.gov/media/166215/download) (Oct 2024)
- [HIPAA Security Rule NPRM](https://www.federalregister.gov/documents/2025/01/06/2024-30983/hipaa-security-rule-to-strengthen-the-cybersecurity-of-electronic-protected-health-information) (Jan 2025)
- [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final) - Security and Privacy Controls
- [NISTIR 8596](https://nvlpubs.nist.gov/nistpubs/ir/2025/NIST.IR.8596.iprd.pdf) - Cybersecurity Profile for AI (Dec 2025)

---

*This module is part of the Privacy Framework for Physical AI Oncology Trials.*
