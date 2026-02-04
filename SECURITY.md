# Security Policy

## Scope

This repository contains tools and pipelines intended for use in **oncology clinical trial** environments. Security issues may affect patient safety, regulatory compliance, and protected health information (PHI). We take all reports seriously.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, report vulnerabilities privately by emailing the maintainer or using [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-managing-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository.

Include:
1. Description of the vulnerability
2. Steps to reproduce
3. Affected component (e.g., `privacy/`, `regulatory/`, `unification/`, example scripts)
4. Potential impact (data exposure, incorrect clinical output, compliance gap)

You should receive an acknowledgment within 7 days. A fix or mitigation plan will be communicated within 30 days for confirmed issues.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (main branch) | Yes |
| Older commits | Best-effort only |

## Security Considerations for Deployers

This repository is intended for **engineers building physical AI systems for oncology clinical trials**. If you are deploying any of these tools in a clinical or research environment, you are responsible for:

### Protected Health Information (PHI)
- Never commit patient data, imaging, or identifiers to this repository or forks
- Use the `privacy/de-identification/` pipeline before any data enters version control or shared storage
- Follow your institution's HIPAA Security Rule policies (45 CFR 164.302-318)

### Regulatory Compliance
- Tools in `regulatory/` are reference implementations, not certified medical device software
- Any deployment in a clinical setting must undergo your institution's validation, verification, and change-control processes per IEC 62304 and 21 CFR Part 11
- AI/ML models trained using this repo's pipelines require independent clinical validation before patient-facing use

### Infrastructure
- Pin all dependency versions (see `requirements.txt`) to avoid supply-chain changes
- Run simulations and training on isolated compute; do not co-locate with systems that store PHI unless properly segmented
- Audit access to any system that processes clinical trial data (see `privacy/access-control/`)

## Dependencies

This project depends on third-party frameworks (PyTorch, MuJoCo, Isaac Lab, MONAI, etc.). Monitor upstream advisories for those packages independently. The CI workflow runs `ruff` for static analysis but does not perform dependency vulnerability scanning; integrating a tool such as `pip-audit` or Dependabot is recommended for production forks.
