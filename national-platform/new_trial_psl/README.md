# all_documents.tex Chunks

These are chunks of `all_documents.tex` split into smaller `.tex` files for processing within token limits. The original file (3376 lines, 11 documents) has been divided into 11 chunk files, one per document.

## File Ordering and Contents

| File | Lines | Content |
|------|-------|---------|
| `01_preamble_and_sb1042.tex` | 1-468 | Preamble (`\documentclass`, packages, macros, `\title`, `\begin{document}`, `\maketitle`, `\tableofcontents`) and Document 1: SB 1042 - California Physical AI Oncology Clinical Trial Authorization and Site Establishment Act |
| `02_ab2847_patient_rights.tex` | 469-794 | Document 2: AB 2847 - California Physical AI Patient Rights and Robotic Safety Act |
| `03_sb892_data_protection.tex` | 795-1101 | Document 3: SB 892 - California Physical AI Clinical Data Protection and Transparency Act |
| `04_sf_municipal_code.tex` | 1102-1363 | Document 4: San Francisco Municipal Code Update - Physical AI Oncology Clinical Trial Site Requirements |
| `05_title22_regulations.tex` | 1364-1685 | Document 5: California Code of Regulations, Title 22 - Physical AI Oncology Trial Site Regulations |
| `06_fda_compliance_guide.tex` | 1686-2002 | Document 6: FDA Physical AI Oncology Trial Site National Compliance Guide |
| `07_building_code.tex` | 2003-2289 | Document 7: Physical AI Oncology Clinical Trial Facility Building Code Standards |
| `08_premises_code.tex` | 2290-2565 | Document 8: Physical AI Oncology Clinical Trial Site Premises Code |
| `09_parking_transportation.tex` | 2566-2815 | Document 9: Physical AI Oncology Clinical Trial Site Parking and Patient Transportation Standards |
| `10_activation_sops.tex` | 2816-3087 | Document 10: Physical AI Oncology Clinical Trial Site Activation and Standard Operating Procedures |
| `11_emergency_preparedness.tex` | 3088-3377 | Document 11: Physical AI Oncology Clinical Trial Site Emergency Preparedness Plan, plus `\printbibliography` and `\end{document}` |

## Reconstructing the Full Document

To reconstruct the full `all_documents.tex`, concatenate all `.tex` files in numerical order:

```bash
cat 01_preamble_and_sb1042.tex \
    02_ab2847_patient_rights.tex \
    03_sb892_data_protection.tex \
    04_sf_municipal_code.tex \
    05_title22_regulations.tex \
    06_fda_compliance_guide.tex \
    07_building_code.tex \
    08_premises_code.tex \
    09_parking_transportation.tex \
    10_activation_sops.tex \
    11_emergency_preparedness.tex > all_documents_reconstructed.tex
```

## Important Notes

- The preamble (`\documentclass`, package imports, macro definitions, `\title`, `\begin{document}`, `\maketitle`, `\tableofcontents`) is in chunk 01.
- The bibliography (`\printbibliography`) and `\end{document}` are in chunk 11.
- Each chunk contains one or more complete document parts and can be processed independently for content review.
- The original file has not been modified.
