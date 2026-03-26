# Physical AI 21 CFR Part 312 - Chunked Source Files

**Note:** This is a copy transferred to national-platform for centralized processing.

Original location: `regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312_chunk/`

This directory contains the file `Physical_AI_21_CFR_Part_312.tex` split into 5 smaller `.tex` files for easier reading and editing. The original file (2275 lines) has not been modified.

## Chunk Files

| File | Lines | Contents |
|------|-------|----------|
| `01_preamble_scope_definitions.tex` | 1--427 (427 lines) | Title page, prefatory note, document history, public domain notice, change summary table, **Subpart A -- General Provisions** (SS 312.1 Scope, SS 312.2 Applicability, SS 312.3 Definitions including 22 Physical AI definitions, SS 312.6 Labeling, SS 312.7 Promotion, SS 312.8 Charging, SS 312.10 Waivers) |
| `02_ind_content_phases.tex` | 428--701 (274 lines) | **Subpart B -- IND Application** (SS 312.20 IND Requirement, SS 312.21 Phases of Investigation including Phase 0 simulation validation, SS 312.22 General IND Submission Principles, SS 312.23 IND Content and Format including Physical AI System Description, SS 312.30 Protocol Amendments including Physical AI amendment triggers) |
| `03_protocol_amendments_reporting.tex` | 702--1029 (328 lines) | Information Amendments (SS 312.31), IND Safety Reporting including Physical AI adverse events and cybersecurity incidents (SS 312.32), Annual Reports with Physical AI performance summary (SS 312.33), IND Withdrawal with Physical AI decommissioning (SS 312.38), **Subpart C -- Administrative Actions** start: General Requirements (SS 312.40), Termination including Physical AI grounds (SS 312.44) |
| `04_annual_reports_withdrawal.tex` | 1030--1492 (463 lines) | Inactive Status and Physical AI dormancy/reactivation (SS 312.45), Meetings including Physical AI system review (SS 312.47), Dispute Resolution with Physical AI technical disputes (SS 312.48), **Subpart D -- Sponsor and Investigator Responsibilities** (SS 312.50--312.70) all with Physical AI adaptations |
| `05_clinical_holds_appendices_closing.tex` | 1493--2275 (783 lines) | **Subpart E -- Life-Threatening Illnesses** (SS 312.80--312.88), **Subpart F -- Miscellaneous** (SS 312.110--312.160), **Subpart G -- Laboratory Use**, **Subpart H [Reserved]**, **Subpart I -- Expanded Access** (SS 312.300--312.320), **Subpart J -- Physical AI Systems (New)** (SS 312.400--312.405), References and Bibliography, `\end{document}` |

## How to Reconstruct the Original File

Concatenate the five chunk files in order:

```bash
cat 01_preamble_scope_definitions.tex \
    02_ind_content_phases.tex \
    03_protocol_amendments_reporting.tex \
    04_annual_reports_withdrawal.tex \
    05_clinical_holds_appendices_closing.tex \
    > Physical_AI_21_CFR_Part_312.tex
```

The result is byte-identical to the original source file.

## Notes

- These chunks are **not** standalone compilable LaTeX documents. Only the first chunk contains `\documentclass` and `\begin{document}`, and only the last chunk contains `\end{document}`.
- The original file at `regulatory/Adaption-21-CFR-Part-312/source/Physical_AI_21_CFR_Part_312.tex` has not been modified.
- Blank separator lines between sections are preserved at chunk boundaries to ensure exact reconstruction.
