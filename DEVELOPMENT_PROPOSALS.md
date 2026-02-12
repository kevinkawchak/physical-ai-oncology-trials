# Physical AI Oncology Trials — 3 Development Prompt Proposals

## Context

The `physical-ai-oncology-trials` repository is at **v1.0.0** (66 commits, 51 Python modules, 70 Markdown files, 40,526 LOC). It covers surgical robotics, digital twins, agentic AI, radiation therapy, immunotherapy prediction, and regulatory compliance across 7 simulation frameworks. The Q2–Q4 2026 roadmap identifies multi-site coordination, model adapters, and FDA documentation as next priorities.

After thorough analysis, the three proposals below represent high-impact, **error-free** additions that Claude Code can deliver end-to-end. Each follows existing repository patterns (numbered examples, module docstrings, markdown documentation templates, ruff-compliant Python 3.10–3.12 code).

---

## Proposal A: Comprehensive Test Suite & Continuous Validation Infrastructure

**Prompt to submit:**

> "Build a comprehensive pytest-based test suite for the physical-ai-oncology-trials repository. Create `tests/` directories mirroring the source structure with unit tests for every Python module, integration tests for cross-module workflows, shared fixtures and mock data factories, a conftest.py hierarchy, and update CI configuration to run tests across Python 3.10–3.12. All tests must pass `ruff format`, `ruff check`, and `py_compile` validation. Include a `tests/README.md` documenting the testing strategy."

### What This Approach Will Accomplish

- **Unit test coverage for all 51 Python modules** — each module gets a corresponding `test_*.py` file exercising its classes, methods, and edge cases
- **Mock-based isolation** — all external dependencies (NVIDIA Isaac, MuJoCo, ROS 2, DICOM servers) are mocked so tests run without hardware or GPU
- **Shared fixture library** — reusable `conftest.py` fixtures for synthetic patient data, tumor geometries, dose distributions, and trial cohort configurations
- **Integration test workflows** — 6 end-to-end test scenarios connecting digital twins → simulation → agentic decision → regulatory audit trail
- **Regression guards for v0.9.2 bug fixes** — dedicated tests reproducing the 30+ bugs fixed (EKF Jacobian sign error, inverted hazard ratio, division-by-zero, infinite loops) to prevent recurrence
- **CI/CD enhancement** — GitHub Actions updated to run `pytest` with coverage reporting across all supported Python versions
- **Clinical safety validation tests** — boundary-condition tests for dose calculations, safety constraint enforcement, and emergency stop logic
- **Test documentation** — `tests/README.md` with testing philosophy, how to add new tests, and coverage targets

### Why This Is Error-Free

- pytest is a standard library pattern requiring no external services
- All tested modules already exist with clear interfaces and `__main__` demos to reference
- Mocking eliminates hardware/GPU dependencies entirely
- Output is validated by the same CI pipeline already in place (ruff + py_compile)

### Key Files Created/Modified

| File | Action |
|------|--------|
| `tests/conftest.py` | Create — shared fixtures, mock factories |
| `tests/test_digital_twins/*.py` | Create — 6 test modules for digital twin examples |
| `tests/test_agentic_ai/*.py` | Create — 6 test modules for agentic AI examples |
| `tests/test_tools/*.py` | Create — 5 test modules for CLI tools |
| `tests/test_physical_robots/*.py` | Create — 6 test modules for robot examples |
| `tests/test_integration/*.py` | Create — 6 cross-module workflow tests |
| `tests/test_regression/*.py` | Create — regression tests for v0.9.2 fixes |
| `tests/README.md` | Create — testing strategy documentation |
| `.github/workflows/ci.yml` | Modify — add pytest step |
| `pyproject.toml` or `setup.cfg` | Modify — add pytest configuration |

---

## Proposal B: Multi-Site Federated Oncology Trial Coordination Platform

**Prompt to submit:**

> "Build a multi-site federated oncology trial coordination platform for the physical-ai-oncology-trials repository. Create a `federation/` directory with: a federated learning coordinator supporting differential privacy and secure aggregation, site enrollment synchronization with conflict resolution, cross-site data harmonization for DICOM/FHIR interoperability, privacy-preserving analytics with configurable epsilon budgets, consortium reporting dashboards (data models and generators), and 6 numbered example scripts. Include a comprehensive `federation/README.md`. All code must follow existing repository patterns, pass ruff validation, and run without external service dependencies using simulated multi-site scenarios."

### What This Approach Will Accomplish

- **Federated learning coordinator** — orchestrates model training across N simulated clinical sites without sharing raw patient data; supports FedAvg, FedProx, and scaffold aggregation strategies
- **Differential privacy engine** — configurable epsilon/delta privacy budgets with Gaussian and Laplacian noise mechanisms applied to gradient updates and summary statistics
- **Secure aggregation protocol** — simulated secure multi-party computation for model weight aggregation preventing any single site from reconstructing another's contributions
- **Site enrollment synchronizer** — real-time (simulated) enrollment tracking across trial sites with conflict resolution for patient eligibility disputes and stratification rebalancing
- **Cross-site data harmonization** — DICOM metadata normalization, FHIR R4 resource mapping, and vocabulary harmonization (ICD-10, SNOMED CT, LOINC) across heterogeneous site systems
- **Consortium reporting engine** — generates site-level and aggregate trial status reports, enrollment dashboards, adverse event summaries, and Data Safety Monitoring Board (DSMB) packages
- **Privacy-preserving analytics** — enables cross-site survival analysis (Kaplan-Meier, Cox proportional hazards) without exposing individual patient records
- **6 progressive examples** — from basic 2-site federation to complex 8-site multi-cancer-type coordination with adaptive enrollment

### Why This Is Error-Free

- All multi-site communication is simulated in-process (no networking required)
- Differential privacy and secure aggregation use standard numpy/scipy operations
- Follows the exact same directory/example structure as existing `examples-agentic-ai/` and `examples-digital-twins/`
- DICOM/FHIR mappings are dictionary-based (no external FHIR server needed)
- Fills an explicit Q2–Q3 2026 roadmap gap documented in `unification/README.md`

### Key Files Created

| File | Purpose |
|------|---------|
| `federation/README.md` | Platform overview, architecture, quick start |
| `federation/federated_coordinator.py` | Core federated learning orchestration engine |
| `federation/differential_privacy.py` | Privacy budget management, noise mechanisms |
| `federation/secure_aggregation.py` | Simulated secure multi-party computation |
| `federation/site_enrollment.py` | Enrollment sync, conflict resolution, stratification |
| `federation/data_harmonization.py` | DICOM/FHIR normalization, vocabulary mapping |
| `federation/consortium_reporting.py` | DSMB reports, enrollment dashboards, AE summaries |
| `federation/privacy_analytics.py` | Privacy-preserving survival analysis |
| `federation/examples-federation/README.md` | Examples overview |
| `federation/examples-federation/01_basic_two_site.py` | Minimal 2-site federation |
| `federation/examples-federation/02_differential_privacy.py` | Privacy budget demonstration |
| `federation/examples-federation/03_secure_aggregation.py` | Secure weight aggregation |
| `federation/examples-federation/04_enrollment_sync.py` | Multi-site enrollment coordination |
| `federation/examples-federation/05_data_harmonization.py` | Cross-site data normalization |
| `federation/examples-federation/06_full_consortium.py` | 8-site multi-cancer coordination |

---

## Proposal C: Regulatory Submission Automation & FDA Pre-Submission Package Generator

**Prompt to submit:**

> "Build a regulatory submission automation system for the physical-ai-oncology-trials repository. Create a `regulatory/` directory with: FDA pre-submission (Pre-Sub) package generators for AI/ML-enabled medical devices, a Predetermined Change Control Plan (PCCP) template engine, 510(k) and De Novo classification decision support, IEC 62304 software lifecycle documentation generators, clinical evidence report builders (linking simulation results to clinical claims), and 6 numbered example scripts. Include a comprehensive `regulatory/README.md`. All code must follow existing repository patterns, pass ruff validation, and generate structured output documents as Markdown. No external API dependencies."

### What This Approach Will Accomplish

- **FDA Pre-Sub package generator** — produces structured pre-submission meeting request packages including device description, intended use statement, proposed testing protocols, and specific questions for FDA, tailored for AI/ML-enabled surgical robotics and treatment planning devices
- **Predetermined Change Control Plan (PCCP) engine** — generates modification protocols defining what model changes (retraining, architecture updates, data drift adaptation) are pre-authorized vs. requiring new submissions, per FDA's 2023 PCCP guidance
- **510(k)/De Novo classification decision tree** — interactive decision support analyzing device characteristics, predicate devices, and novel features to recommend the appropriate regulatory pathway with justification
- **IEC 62304 documentation generator** — produces software lifecycle documents (Software Development Plan, Software Requirements Specification, Software Architecture Document, risk analysis matrices) from repository metadata and code structure
- **Clinical evidence report builder** — links simulation benchmark results from `unification/benchmark_suite.py` and digital twin validation data to clinical performance claims with statistical analysis and confidence intervals
- **Audit trail documentation** — generates 21 CFR Part 11-compliant audit trail records for all AI model training runs, validation experiments, and configuration changes
- **6 progressive examples** — from single-device Pre-Sub to multi-indication regulatory strategy with PCCP lifecycle management

### Why This Is Error-Free

- All output is generated Markdown (no external FDA systems or APIs needed)
- Decision trees and templates are implemented as Python data structures and string formatting
- Builds directly on existing regulatory content in `privacy/regulatory_compliance.py` and standards referenced throughout the repository
- Follows the exact same module + examples pattern used in all other directories
- Clinical evidence linking reuses existing benchmark data structures from `unification/benchmark_suite.py`

### Key Files Created

| File | Purpose |
|------|---------|
| `regulatory/README.md` | System overview, regulatory context, quick start |
| `regulatory/presub_generator.py` | FDA Pre-Sub meeting package generation |
| `regulatory/pccp_engine.py` | Predetermined Change Control Plan templates |
| `regulatory/classification_advisor.py` | 510(k)/De Novo pathway decision support |
| `regulatory/iec62304_generator.py` | Software lifecycle documentation |
| `regulatory/clinical_evidence.py` | Simulation-to-clinical-claims linking |
| `regulatory/audit_trail.py` | 21 CFR Part 11 audit trail generation |
| `regulatory/examples-regulatory/README.md` | Examples overview |
| `regulatory/examples-regulatory/01_presub_package.py` | Basic Pre-Sub generation |
| `regulatory/examples-regulatory/02_pccp_plan.py` | Change control plan creation |
| `regulatory/examples-regulatory/03_classification.py` | Pathway decision support |
| `regulatory/examples-regulatory/04_iec62304_docs.py` | Lifecycle documentation |
| `regulatory/examples-regulatory/05_clinical_evidence.py` | Evidence report building |
| `regulatory/examples-regulatory/06_full_submission.py` | Complete regulatory strategy |

---

## Comparative Analysis

### Feature-by-Feature Comparison

| Dimension | Proposal A: Test Suite | Proposal B: Federation | Proposal C: Regulatory |
|-----------|----------------------|----------------------|----------------------|
| **Primary Value** | Code reliability & regression prevention | Multi-site trial scalability | Regulatory pathway acceleration |
| **New Python Modules** | ~25 test modules | 8 core + 6 examples | 7 core + 6 examples |
| **New Documentation** | 1 README + inline docstrings | 2 READMEs + module docs | 2 READMEs + module docs |
| **Estimated New LOC** | 8,000–12,000 | 6,000–9,000 | 5,000–8,000 |
| **Reuses Existing Code** | Tests all 51 existing modules | Extends digital twins + privacy | Extends regulatory + benchmarks |
| **CI/CD Impact** | Direct (adds test stage) | Indirect (testable modules) | Indirect (testable modules) |
| **Roadmap Alignment** | Foundation for all Q2–Q4 work | Q2–Q3 2026 explicit priority | Q3 2026 explicit priority |
| **Clinical Impact** | Safety assurance via validation | Multi-center trial enablement | Faster FDA clearance pathway |
| **Regulatory Impact** | Supports IEC 62304 testing evidence | HIPAA/cross-border compliance | Direct FDA submission support |
| **External Dependencies** | pytest (standard) | None (all simulated) | None (all Markdown output) |
| **Risk of Errors** | Very low (mocking, no I/O) | Very low (in-process simulation) | Very low (template generation) |
| **Immediate Usability** | Run `pytest` immediately | Run examples immediately | Generate documents immediately |

### Strategic Impact Matrix

| Strategic Goal | Proposal A | Proposal B | Proposal C |
|---------------|:----------:|:----------:|:----------:|
| Code quality improvement | High | Medium | Medium |
| Production readiness | High | High | Medium |
| Clinical trial scalability | Medium | High | Medium |
| Regulatory compliance | Medium | Medium | High |
| Multi-site deployment | Low | High | Low |
| Developer onboarding | High | Medium | Medium |
| Stakeholder confidence | High | High | High |
| Novel IP / differentiation | Low | High | High |
| Foundation for future work | High | Medium | Medium |
| Fills documented roadmap gap | Medium | High | High |

### Audience Impact

| Stakeholder | Proposal A | Proposal B | Proposal C |
|------------|-----------|-----------|-----------|
| **Software Engineers** | Primary beneficiary | Secondary | Secondary |
| **Clinical Researchers** | Indirect (safety) | Primary beneficiary | Secondary |
| **Regulatory Affairs** | Indirect (evidence) | Secondary | Primary beneficiary |
| **Site Coordinators** | None | Primary beneficiary | None |
| **FDA Reviewers** | Indirect (quality) | Indirect (scale) | Primary beneficiary |
| **IRB Members** | None | Secondary | Primary beneficiary |
| **Data Scientists** | Secondary | Primary beneficiary | None |

---

## Recommendation

All three proposals are designed to be **fully executable by Claude Code without errors**. The choice depends on the user's immediate priority:

- **Choose Proposal A** if the priority is engineering rigor and establishing a safety net before further development
- **Choose Proposal B** if the priority is scaling trials across multiple clinical sites (aligns with Q2 2026 roadmap)
- **Choose Proposal C** if the priority is accelerating regulatory submissions and FDA engagement
