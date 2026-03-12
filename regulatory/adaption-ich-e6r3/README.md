# ICH E6(R3) LaTeX Conversion Package

This archive contains a best-effort LaTeX reconstruction of the source PDF:

- `main.tex` — main LaTeX source
- `ich_guideline_style.sty` — local style package
- `references.bib` — bibliography file
- `compiled.pdf` — compiled PDF output

## Notes

- The package was generated automatically from the PDF text layer.
- Section hierarchy and paragraph structure were reconstructed heuristically.
- The authoritative document remains the original PDF.
- Some spacing, pagination, or list semantics may differ from the source.

## Build

A typical build command is:

```bash
latexmk -pdf main.tex
```
