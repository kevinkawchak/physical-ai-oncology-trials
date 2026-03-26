**Note:** This directory is a copy transferred to `national-platform/` for centralized Claude Code Opus 4.6 processing. Original files remain in their source location.

# Federated Learning Paper — Chunked Files

Chunked version of `main.tex` from the [pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl) repository.

## Source

- **Original file**: `paper/main.tex` (931 lines)
- **Original repo**: [kevinkawchak/pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl)
- **Bibliography**: `references.bib` (included in this directory)

## Chunk Files

| # | File | Lines | Description |
|---|------|-------|-------------|
| 1 | `01_preamble_introduction_methods.tex` | 1–216 | Document preamble, title, abstract, Section 1 (Introduction), Section 2 (Methods) |
| 2 | `02_results.tex` | 217–777 | Section 3 (Results): platform architecture, privacy, regulatory, unification, standards, cooperation, workflow demos, peer review, testing, analytics, digital twins, tech stack, federated coordinator, CLI tools |
| 3 | `03_discussion_conclusion.tex` | 778–931 | Section 4 (Discussion), Section 5 (Limitations and Future Work), Section 6 (Conclusion), References, Acknowledgments, Ethical Disclosures, Rights and Permissions, Citation |

## Reconstruction

```bash
cat 01_preamble_introduction_methods.tex 02_results.tex 03_discussion_conclusion.tex > main.tex
```
