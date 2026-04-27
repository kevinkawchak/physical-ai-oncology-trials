# National Platform for Physical AI Oncology Trials - LaTeX Template

## Overview

This directory contains the LaTeX source files for the **National Platform for Physical AI Oncology Trials** (Draft 1.0). The template provides a comprehensive structure for a 175-page paper that serves as an end-to-end resource for the pharmaceutical and regulatory industries.

## File Structure

```
new_template/
  main.tex              - Main document (entry point)
  page_styles.tex       - Page style definitions (with template attribution)
  references.bib        - Bibliography (35 sources)
  README.md             - This file
  sections/
    cover_page.tex        - Cover page with title, author, notices
    contents.tex          - Table of contents
    source_documents.tex  - Source documents overview and significance
    executive_summary.tex - Executive summary
    introduction.tex      - Section 1: Introduction to Physical AI Oncology Trials
    gov_framework.tex     - Section 2: U.S. Government Framework
    regulatory_landscape.tex - Section 3: California and Federal Regulatory Landscape
    ich_e6r3_adaptation.tex  - Section 4: Adapted ICH E6(R3)
    cfr50_adaptation.tex     - Section 5: Adapted 21 CFR Part 50
    cfr312_adaptation.tex    - Section 6: Adapted 21 CFR Part 312
    psl_usl_standards.tex    - Section 7: PSL and USL Standards
    site_establishment.tex   - Section 8: Clinical Trial Site Establishment
    patient_journey.tex      - Section 9: A Cancer Patient's Journey
    patient_instructions.tex - Section 10: Patient Instructions
    national_mcp.tex         - Section 11: National MCP Server Infrastructure
    federated_learning.tex   - Section 12: Federated Learning Framework
    financial_analysis.tex   - Section 13: Financial and Economic Impact
    implementation_strategy.tex - Section 14: National Implementation Strategy
    discussion.tex              - Section 15: Discussion
    conclusion.tex              - Section 16: Conclusion
    appendices.tex              - Appendices A-E
```

## Source Documents

Each section references specific files from the `national-platform/` directory. Bracketed instructions within each `.tex` file provide detailed guidance on which source files to use and how to process them.

| Section | Primary Source Directory |
|---------|------------------------|
| Section 2 | `research_a/` |
| Section 3 | `research_b/` |
| Section 4 | `ich_e6r3_adapt/` |
| Section 5 | `21cfr50_adapt/` |
| Section 6 | `21cfr312_adapt/` |
| Section 7 | `usl_standard/` + `new_trial_psl/` |
| Section 8 | `new_trial_psl/` |
| Section 9 | `patient_journey/` |
| Section 10 | `patient_robot/` |
| Section 11 | `national_mcp/` |
| Section 12 | `federated_learning/` |

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Template Attribution

Adapted from University of Groningen MSc AI and CCS Master's Thesis Template (Overleaf, CC BY 4.0). Original template by Manvi Agarwal (2020).

## License

MIT
