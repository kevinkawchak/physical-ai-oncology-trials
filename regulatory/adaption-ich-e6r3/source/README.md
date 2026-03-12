# End-to-End Physical AI Oncology Clinical Trial Unification -- LaTeX Source

This directory contains the LaTeX source for the *End-to-End Physical AI Oncology Clinical Trial Unification* guidance, adapted from the prior ICH E6(R3) regulation.

## Files

- `main.tex` -- main LaTeX source (Sections 1-4, Appendices A-C, Glossary)
- `ich_guideline_style.sty` -- custom style package
- `references.bib` -- bibliography (18 references)
- `compiled.pdf` -- compiled PDF output
- `ICH_E6R3_LaTeX_Package.zip` -- archive of all source files

## Version

- **v2.2.0** (March 12, 2026)
- **DOI**: [10.5281/zenodo.18973368](https://doi.org/10.5281/zenodo.18973368)
- **Author**: CEO Kevin Kawchak, ChemicalQDevice
- **Development**: Claude Code Opus 4.6

## Build

```bash
latexmk -pdf main.tex
```

Or manually:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Notes

- Adapted from the prior ICH E6(R3) regulation (adopted 06 January 2025)
- The original ICH E6(R3) document is not endorsed or sponsored by ICH
- This guidance addresses physical AI oncology trial unification using advanced AI and robotics
- References USL scores (v1.4.0 through v1.8.0) for 9 robot platforms across 3 categories
- References the physical-ai-oncology-trials repository (v2.2.0)
