# LaTeX-Source-Files-01

**One-Page Summary of H. R. 9510** -- a single-column LaTeX template (number 01 of 30) in the *Legislative deliverables* family.

- **Genre / perspective:** Executive one-pager (labeled blocks).
- **Built on INSPIRATIONS:** VVUQ-05 deliverable 01; VVUQ-03 back matter.
- **Body style:** Times.
- **DOI placeholder:** [`10.5281/zenodo.xxxxxxxx`](https://doi.org/10.5281/zenodo.xxxxxxxx) (replace `xxxxxxxx`).
- **Author:** Kevin Kawchak, CEO ChemicalQDevice, ORCID `0009-0007-5457-8667`.

## Files

```
LaTeX-Source-Files-01/
  main.tex
  tmpl01style.sty
  references.bib
  README.md
  orcid_icon.png
  sections/
    section-a.tex
    section-b.tex
    section-c.tex
    section-d.tex
    section-e.tex
    section-f.tex
```

## One section file per document section

This template uses **one `section-x.tex` file per document section** (rule 9); add a section by adding `sections/section-g.tex` and an `\input` line in `main.tex`.

| Section file | Document section |
|:--|:--|
| `sections/section-a.tex` | The problem |
| `sections/section-b.tex` | The solution: a new section 515D |
| `sections/section-c.tex` | What it changes |
| `sections/section-d.tex` | The assurance evidence |
| `sections/section-e.tex` | Cost and the ask |
| `sections/section-f.tex` | Back matter (VVUQ-03) |

## Included per the rules

- One **image placeholder** based on the figure code of `papers/VVUQ-02/final-paper`.
- One **table** modeled on **Table 1** of `papers/VVUQ-05/final-bill`.
- The **back matter** from `papers/VVUQ-03/final-paper` (the parts that fit this genre).
- A **blank Keywords line** on the first page; the **DOI** on the cover and in the citation block.
- The green **ORCID logo** (`orcid_icon.png`) on the cover in place of the green `iD` text.

## Compile (Overleaf, pdfLaTeX)

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## License

CC BY 4.0. Reproduced public-domain U.S. Government text is used under 17 U.S.C. § 105.
