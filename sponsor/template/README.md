# Fully Automated Sponsor: Physical AI Oncology Clinical Trials

LaTeX paper template and processing instructions for a fully autonomous AI-native sponsor
operating system for Physical AI oncology clinical trials.

## Overview

This template defines a comprehensive paper on replacing traditional human pharmaceutical
sponsor functions with autonomous AI agents and robotic subsystems. The template contains
detailed processing instructions (in brackets) for Claude Code Opus 4.6 (1M token context)
to generate the final 40+ page paper.

The autonomous sponsor system comprises 12 functional agents organized into 4 layers:
1. **Governance Layer**: Board agent cluster, portfolio agent, asset lead agent
2. **Study Execution Layer**: Clinical accountability, study orchestrator, ClinOps, safety,
   regulatory, quality, supply, data/biostats agents
3. **Site/Robotics Layer**: Site gateway, robot execution gateway
4. **Trust Layer**: Identity, authorization, audit, provenance, policy enforcement

## File Structure

```
sponsor/template/
|-- main.tex                         # Main document entry point (18 sections + appendices)
|-- sponsor_paper.sty                # Style file (adapted from arxiv.sty, CC BY 4.0)
|-- references.bib                   # Bibliography (48 entries with DOIs and URLs)
|-- orcid_icon.png                   # ORCID icon for author attribution
|-- README.md                        # This file
|-- sections/
|   |-- introduction.tex             # Section 1: Introduction and paper roadmap
|   |-- governance.tex               # Section 2: Board/portfolio/asset agents, decision gates
|   |-- trial_design.tex             # Section 3: Protocol design, biomarkers, biostatistics
|   |-- clinical_operations.tex      # Section 4: Study orchestrator, ClinOps, enrollment
|   |-- safety_pharmacovigilance.tex # Section 5: Safety agent, E2B(R3), robotic safety
|   |-- regulatory_affairs.tex       # Section 6: IND/CTA automation, accelerated pathways
|   |-- quality_compliance.tex       # Section 7: Quality agent, RBQM, electronic systems
|   |-- supply_chain.tex             # Section 8: Supply agent, CMC, robotic pharmacy
|   |-- data_management.tex          # Section 9: Data pipeline, CDISC, federated data
|   |-- robotic_execution.tex        # Section 10: Robot gateway, capability registry
|   |-- site_interface.tex           # Section 11: Site gateway, FHIR/DICOM/MCP integration
|   |-- trust_layer.tex              # Section 12: Identity, audit, provenance, policy engine
|   |-- vendor_management.tex        # Section 13: Vendor mesh, CRO accountability matrix
|   |-- writing_disclosure.tex       # Section 14: Document generation, disclosure, archiving
|   |-- financial_analysis.tex       # Section 15: Cost modeling, ROI, timeline compression
|   |-- implementation_strategy.tex  # Section 16: 3-phase national deployment strategy
|   |-- discussion.tex               # Section 17: Paradigm shift, challenges, limitations
|   |-- conclusion.tex               # Section 18: Summary of contributions and future work
|   |-- appendices.tex               # Appendices A-D: Agent registry, script directory,
|                                    #   source cross-reference, regulatory compliance mapping
```

## How Files Relate to Each Other

- **main.tex** is the entry point that imports the style file and all section files
- **sponsor_paper.sty** defines page layout, fonts, title formatting, and abstract styling
- **references.bib** is referenced by main.tex via `\bibliography{references}`
- **orcid_icon.png** is used by the `\orcidicon` command in the title/author block
- Each **sections/*.tex** file is imported by main.tex via `\input{sections/filename}`
- The **appendices** reference all other sections and provide supplementary detail

## Source Files Used

This template draws from the following repository directories:

### Sponsor Input Files (sponsor/input_files/)
- `sponsor_01` through `sponsor_08`: End-to-End Sponsor Playbook (8 chunks)
- `org_01` through `org_07`: Sponsor Organization (7 chunks)
- `README.md`: Cross-document alignment and processing notes

### National Platform Paper (national-platform/new_paper/final_paper/)
- 21 section .tex files covering regulatory, standards, infrastructure, and implementation
- `references.bib` with 34 sources
- `page_styles.tex` and `main.tex` for structure reference

### Patient Journey Paper (patient-journey/paper/)
- `patient_journey_paper.tex`: Source for abstract length calibration and robotic examples
- `arxiv.sty`: Base style file (adapted into sponsor_paper.sty)
- `orcid_icon.png`: ORCID icon asset

### Regulatory Adaptations (national-platform/)
- `21cfr312_adapt/`: 5 .tex files adapting 21 CFR Part 312 for Physical AI
- `ich_e6r3_adapt/`: 4 .tex files adapting ICH E6(R3)
- `federated_learning/`: 4 .tex files on federated learning framework
- `national_mcp/`: 4 .tex files on MCP server infrastructure
- `new_trial_psl/`: 11 .tex files on trial site documentation
- `usl_standard/`: 2 .tex files on USL standard

## Processing Instructions

### For Claude Code Opus 4.6 (1M token context)

1. **Read all source files** listed above into context
2. **Process each section** following the bracketed instructions in each .tex file
3. **Generate 49 Python scripts** in `sponsor/template/scripts/` as specified in each section
4. **Run Python scripts** to validate functionality and generate output artifacts
5. **Update references.bib** with any additional citations needed
6. **Compile the LaTeX document** and verify formatting (no text overflow, proper spacing)
7. **Verify bibliography** shows DOIs and URLs correctly

### Python Script Requirements
- All 49 scripts must pass `ruff check` with the repository's ruff.toml configuration
- Line length: 120 characters maximum
- Rules: E (pycodestyle), F (Pyflakes), W (pycodestyle warnings)
- Target Python version: 3.10+
- Each script should be self-contained with clear docstrings and type hints

### LaTeX Compilation
```bash
cd sponsor/template
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Expected Output
- Final paper: 40+ pages
- 30+ tables across all sections
- 49 Python scripts in sponsor/template/scripts/
- All bibliography references with clickable DOIs and URLs

## Table Index

| Table | Title | Section |
|-------|-------|---------|
| 1 | Governance Agent Mapping | Section 2 |
| 2 | Decision Gate Framework | Section 2 |
| 3 | Trial Design Pattern Comparison | Section 3 |
| 4 | Estimand Framework Components | Section 3 |
| 5 | Study Startup Automation Workflow | Section 4 |
| 6 | Failure Mode Mitigation Matrix | Section 4 |
| 7 | Safety Reporting Timeline | Section 5 |
| 8 | Robotic Safety Event Classification | Section 5 |
| 9 | Global Regulatory Filing Matrix | Section 6 |
| 10 | Accelerated Pathway Comparison | Section 6 |
| 11 | QTL/KRI Framework | Section 7 |
| 12 | Electronic Systems Validation | Section 7 |
| 13 | IMP Supply Chain Automation | Section 8 |
| 14 | Robotic Pharmacy Operations | Section 8 |
| 15 | Data Supply Chain Pipeline | Section 9 |
| 16 | Federated Data Architecture | Section 9 |
| 17 | Robotic System Registry | Section 10 |
| 18 | Procedure Safety Gate Protocol | Section 10 |
| 19 | Site Systems Integration Matrix | Section 11 |
| 20 | Site Readiness Assessment | Section 11 |
| 21 | Trust Layer Architecture | Section 12 |
| 22 | Safety Gate Classification | Section 12 |
| 23 | Sponsor vs. CRO Accountability | Section 13 |
| 24 | Vendor SLA Framework | Section 13 |
| 25 | Document Generation Pipeline | Section 14 |
| 26 | Disclosure Timeline Requirements | Section 14 |
| 27 | Traditional vs. Autonomous Cost | Section 15 |
| 28 | Timeline Compression Analysis | Section 15 |
| 29 | Implementation Phase Timeline | Section 16 |
| 30 | Implementation Risk Matrix | Section 16 |

## Citation

```bibtex
@misc{sponsor-paper,
  author = {Kawchak, Kevin},
  title = {Fully Automated Sponsor: Physical {AI} Oncology Clinical Trials},
  year = {2026},
  howpublished = {Zenodo},
  doi = {10.5281/zenodo.19396256},
  note = {DOI: 10.5281/zenodo.19396256.
    GitHub: https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/sponsor.
    Zenodo: https://doi.org/10.5281/zenodo.19396256}
}
```
