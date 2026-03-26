**Note:** This directory is a copy transferred to `national-platform/` for centralized Claude Code Opus 4.6 processing. Original files remain in their source location.

# Physical AI 21 CFR Part 50 -- Chunked Source Files

This directory contains the file `Physical_AI_21_CFR_Part_50.tex` split into three smaller `.tex` chunk files for easier review and navigation. The original file (747 lines) has not been modified.

## Chunk Files

### 01_preamble_scope_definitions_consent.tex (lines 1-438)
Contains the document preamble, title page, prefatory note, document history, public domain notice, **Subpart A -- General Provisions** (scope and definitions including all Physical AI definitions), and **Subpart B -- Informed Consent of Human Subjects**.

### 02_irb_review_pediatric.tex (lines 439-674)
Contains **Subpart C -- Additional Protections for Subjects in Physical AI Clinical Investigations** and **Subpart D -- Additional Safeguards for Children in Clinical Investigations**.

### 03_additional_safeguards_closing.tex (lines 675-747)
Contains the **Glossary** of Physical AI terms, the bibliography (`\printbibliography`), and the `\end{document}` closing.

## How to Reconstruct the Original File

```bash
cat 01_preamble_scope_definitions_consent.tex \
    02_irb_review_pediatric.tex \
    03_additional_safeguards_closing.tex \
    > Physical_AI_21_CFR_Part_50_reconstructed.tex
```
