# Physical AI 21 CFR Part 50 -- Chunked Source Files

This directory contains the file `Physical_AI_21_CFR_Part_50.tex` split into three smaller `.tex` chunk files for easier review and navigation. The original file (747 lines) has not been modified.

## Chunk Files

### 01_preamble_scope_definitions_consent.tex (lines 1-438)
Contains the document preamble, title page, prefatory note, document history, public domain notice, **Subpart A -- General Provisions** (scope and definitions including all Physical AI definitions), and **Subpart B -- Informed Consent of Human Subjects** (general requirements, exceptions, elements of informed consent, Physical AI consent adaptations, and documentation of informed consent through the end of section 50.27).

### 02_irb_review_pediatric.tex (lines 439-674)
Contains **Subpart C -- Additional Protections for Subjects in Physical AI Clinical Investigations** (Physical AI system safety requirements including pre-procedure safety matrix, runtime safety monitoring, post-procedure requirements, task-order lifecycle, forbidden operations, IRB review of Physical AI investigations, ongoing consent and subject notification, data protection, and Physical AI system classification/regulatory pathways) and **Subpart D -- Additional Safeguards for Children in Clinical Investigations** (IRB duties, minimal risk investigations, greater than minimal risk investigations, wards, and all associated Physical AI adaptations for pediatric populations).

### 03_additional_safeguards_closing.tex (lines 675-747)
Contains the **Glossary** of Physical AI terms supplementing the regulatory definitions, the bibliography (`\printbibliography`), and the `\end{document}` closing.

## How to Reconstruct the Original File

Concatenate the three chunk files in order:

```bash
cat 01_preamble_scope_definitions_consent.tex \
    02_irb_review_pediatric.tex \
    03_additional_safeguards_closing.tex \
    > Physical_AI_21_CFR_Part_50_reconstructed.tex
```

The reconstructed file will be identical to the original `../Physical_AI_21_CFR_Part_50.tex`.
