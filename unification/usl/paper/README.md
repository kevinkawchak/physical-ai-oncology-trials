## Paper PDF and LaTeX Source Files

📄 **2/26: New Paper (USL)** *Unification Standard Level for Physical AI Oncology Trials. Standardizing and Evaluating Robot Unification Readiness for Multi-Site Clinical Trials. USL scores range from 1.0 to 10.0* [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18778219-blue)](https://doi.org/10.5281/zenodo.18778219)

Files
-----

  usl_oncology_trials.tex   Main LaTeX document (article class, 11pt)
  usl-oncology.sty          Custom style package (geometry, colors, formatting)
  references.bib            BibTeX bibliography (28 references)
  README                    This file

Compilation
-----------

  # Full compilation with bibliography:
  pdflatex usl_oncology_trials.tex
  bibtex usl_oncology_trials
  pdflatex usl_oncology_trials.tex
  pdflatex usl_oncology_trials.tex

  # Or using latexmk:
  latexmk -pdf usl_oncology_trials.tex

Requirements
------------

  - LaTeX distribution (TeX Live 2024+ recommended)
  - Packages: geometry, fontenc, inputenc, mathptmx, microtype, xcolor,
    graphicx, tikz, booktabs, tabularx, multirow, array, colortbl,
    enumitem, listings, hyperref, fancyhdr, titlesec, caption, amsmath,
    amssymb, float, soul

License
-------

  CC BY 4.0 - Creative Commons Attribution 4.0 International
  https://creativecommons.org/licenses/by/4.0/

Repository
----------

  https://github.com/kevinkawchak/physical-ai-oncology-trials
