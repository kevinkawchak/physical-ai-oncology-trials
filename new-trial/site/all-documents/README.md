# Physical AI Oncology Trial Site - All Documents Combined

**Version**: 2.9.0
**Last Updated**: March 2026

## Overview

Complete LaTeX source for all 11 Physical AI oncology clinical trial site
documents in a single compilable package. When compiled, this produces a single
PDF containing all legislation drafts, regulatory updates, building code,
premises code, parking standards, site operations, and emergency preparedness
documents with no blank pages between them.

## Files

| File | Description |
|------|-------------|
| `all_documents.tex` | Main LaTeX source with all 11 documents |
| `all_documents.bib` | Combined bibliography (15 deduplicated entries) |
| `physical_ai_legislation.sty` | Shared style package |
| `README.md` | This file |

## Compilation

```bash
pdflatex all_documents.tex
biber all_documents
pdflatex all_documents.tex
pdflatex all_documents.tex
```

## Documents Included

1. SB 1042 - California Physical AI Trial Authorization Act
2. AB 2847 - California Physical AI Patient Rights and Robotic Safety Act
3. SB 892 - California Physical AI Clinical Data Protection Act
4. San Francisco Municipal Code Update
5. California Code of Regulations, Title 22
6. FDA Physical AI National Compliance Guide
7. Building Code Standards
8. Premises Code
9. Parking and Patient Transportation Standards
10. Site Activation and Standard Operating Procedures
11. Emergency Preparedness Plan

## Disclaimer

Unofficial independent draft derived from prior legislation; no third-party
endorsement, sponsorship, affiliation, or authorization is expressed or
implied; and was adapted using Claude Code Opus 4.6.
