**Note:** This directory is a copy transferred to `national-platform/` for centralized Claude Code Opus 4.6 processing. Original files remain in their source location.

# Chunks of main.tex

These files are chunks of the parent file `../main.tex` (the Physical AI Oncology Clinical Trial Unification -- Adaption of ICH E6(R3)). The original file (1300 lines) has been split into four sequential pieces for easier navigation and editing. The original file has not been modified.

## Chunk contents

| File | Lines | Contents |
|------|-------|----------|
| `01_preamble_principles_investigator.tex` | 1--337 | LaTeX preamble, title page, prefatory note, document history, legal notice, Section 1 (Principles of Physical AI Clinical Practice, subsections 1.1--1.5), and Section 2 (Investigator Responsibilities, subsections 2.1--2.12). |
| `02_sponsor_responsibilities.tex` | 338--683 | Section 3 (Sponsor Responsibilities in Physical AI Trials, subsections 3.1--3.15), covering quality management, regulatory submission, IRB/IEC review, trial design, monitoring, safety reporting, data handling, and clinical trial reports. |
| `03_data_governance.tex` | 684--911 | Section 4 (Data Governance for Physical AI Trials -- Investigator and Sponsor, subsections 4.1--4.3), covering blinding safeguards, data lifecycle elements, and computerised systems requirements. |
| `04_appendices_glossary.tex` | 912--1300 | Appendix A (Physical AI System Documentation), Appendix B (Clinical Trial Protocol), Appendix C (Essential Records), Glossary, bibliography (`\printbibliography`), and `\end{document}`. |

## How to reconstruct the original file

Concatenate the four chunks in order:

```bash
cat 01_preamble_principles_investigator.tex \
    02_sponsor_responsibilities.tex \
    03_data_governance.tex \
    04_appendices_glossary.tex > main.tex
```
