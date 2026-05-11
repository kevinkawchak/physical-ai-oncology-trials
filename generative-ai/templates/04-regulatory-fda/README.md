# Regulatory and FDA Submission Manuscript Template

Single column LaTeX academic paper template focused on the
**Regulatory and FDA Submission** perspective for physical AI oncology trials. This
bundle compiles cleanly with `pdflatex` plus `bibtex` (recommended on
Overleaf) and ships as one of ten distinct templates under
`generative-ai/templates/` in the parent repository.

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Template](https://img.shields.io/badge/Template-04%20regulatory%20fda-blue.svg)](../)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxxx-blue)](https://doi.org/10.5281/zenodo.xxxxxxxx)

## Title page metadata (placeholder)

```
Title:    Regulatory and FDA Submission Manuscript Template
Author:   Kevin Kawchak  (ORCID iD + https://orcid.org/0009-0007-5457-8667)
Affil:    CEO ChemicalQDevice
DOI:      10.5281/zenodo.xxxxxxxx  (https://doi.org/10.5281/zenodo.xxxxxxxx)
Date:     May 11, 2026
```

A blank single-line `Keywords` slot is rendered on page one immediately
under the abstract so authors fill in their own keywords without
disturbing layout. Selected section headings and the title block use a dark blue accent as a stylistic differentiator across the template family.

## File layout

```
04-regulatory-fda/
  README.md          (this file)
  main.tex           (preamble, title page, TOC, \input lines)
  new_paper.sty      (style file specific to this template)
  references.bib     (starter bibliography)
  sections/
    abstract.tex
    introduction.tex
    methods.tex
    results.tex
    discussion.tex
    limitations_future.tex
    conclusions.tex
    back_matter.tex  (Acknowledgments, Ethical Disclosures, Rights,
                      Cite This Article, Data Availability)
```

## Section inventory

| Order | Section | File | Contents |
|:------|:--------|:-----|:---------|
| 1 | Abstract | sections/abstract.tex | One paragraph |
| 2 | Introduction | sections/introduction.tex | Paragraph plus 3-row Table 2 layout |
| 3 | Methods | sections/methods.tex | Paragraph plus 3-row Table 2 layout |
| 4 | Results | sections/results.tex | Paragraph plus 3-row Table 2 layout |
| 5 | Discussion | sections/discussion.tex | Paragraph plus 3-row Table 2 layout |
| 6 | Limitations and Future Work | sections/limitations_future.tex | Paragraph plus 3-row Table 2 layout |
| 7 | Conclusions | sections/conclusions.tex | Paragraph plus 3-row Table 2 layout |
| 8 | References | references.bib (ieeetr style) | 5 starter entries |
| 9 | Back Matter | sections/back_matter.tex | 5 short sections |

## Compile recipe

```
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Tested behavior: single column, no line numbers, no preprint header,
clickable DOI plus ORCID link, raggedright tables to prevent rivers.

## Inspiration

Inspired by leading CC BY templates such as the MDPI single column
article template (Overleaf) and other journal-style CC BY templates,
adapted into ten visually distinct variants without preprint editing
formatting and without line numbers in columns.

## License

This template is distributed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).
