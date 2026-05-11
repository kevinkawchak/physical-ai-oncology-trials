# Generative AI LaTeX Paper Templates

Ten distinct single column LaTeX academic paper templates for physical AI
oncology trials, each presented from a different scholarly perspective.
Every template ships as both an unzipped source tree (browsable on
GitHub) and a downloadable zip bundle (Overleaf-ready).

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Templates](https://img.shields.io/badge/Templates-10-blue.svg)](.)
[![Format](https://img.shields.io/badge/Format-Single%20Column-blue.svg)](.)

## Why ten perspectives

Physical AI oncology trials sit at the intersection of clinical
research, engineering, regulation, payer evidence, patient experience,
and translational science. Ten templates with deliberately different
typography give authors a starting point that visually matches the
target audience while preserving an identical section inventory across
every variant.

## Template index

| # | Perspective | Folder | Style note |
|:--|:------------|:-------|:-----------|
| 01 | Clinical Trial Protocol | `01-clinical-trial-protocol/` | Classic Times Roman, neutral black |
| 02 | Robotic Surgery Engineering | `02-robotic-surgery-engineering/` | Modern sans-serif, **dark blue accents** |
| 03 | Bioinformatics and Genomics | `03-bioinformatics-genomics/` | MDPI-inspired clean look |
| 04 | Regulatory and FDA Submission | `04-regulatory-fda/` | Palatino serif, **dark blue navy accents** |
| 05 | Patient-Centered Outcomes | `05-patient-centered-outcomes/` | Slim minimalist (Charter body) |
| 06 | Health Economics and Outcomes | `06-health-economics/` | Bold display, large sans-serif title |
| 07 | AI and Machine Learning Methods | `07-ai-ml-methods/` | Technical monospaced headings |
| 08 | Digital Twin and Simulation | `08-digital-twin-simulation/` | Elegant Palatino serif, italic titles |
| 09 | Multi-Site Federation | `09-multi-site-federation/` | Journal-style uppercase headings |
| 10 | Translational Oncology | `10-translational-oncology/` | Computer Modern (LaTeX default) |

Templates 02 and 04 are the two variants that specify dark blue text
for relevant sections (title, abstract heading, section headings, and
selected rules).

## Shared template family invariants

Every template ships with:

- **Single column layout**, no line numbers, no preprint header.
- **`main.tex`** at the root with the title, author, ORCID iD link,
  DOI link (`10.5281/zenodo.xxxxxxxx`), date, and `\input` lines for
  each section.
- **`new_paper.sty`** template-specific style file.
- **`references.bib`** starter bibliography (5 entries, ieeetr style).
- **`README.md`** with file layout, section inventory, and a compile
  recipe.
- **`sections/`** subdirectory with one `.tex` file per section:
  `abstract.tex`, `introduction.tex`, `methods.tex`, `results.tex`,
  `discussion.tex`, `limitations_future.tex`, `conclusions.tex`,
  `back_matter.tex`.
- A **blank single-line `Keywords` slot** on page one immediately under
  the abstract.
- A **single illustrative paragraph** per section.
- A **three-row Table 2 layout** in every body section using the column
  width pattern `>{\raggedright\arraybackslash}p{3.4cm}` /
  `p{4.6cm}` / `p{5.4cm}` so cells do not develop large interword rivers.
- A back matter block carrying Acknowledgments, Ethical Disclosures,
  Rights and Permissions, Cite This Article, and a generalized Data
  Availability section.

## Compile recipe (every template)

```
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Tested behavior: clickable DOI plus ORCID link, raggedright tables to
prevent rivers, no widows or orphans, no line numbers, no preprint
markers.

## Generate the Overleaf-ready zip locally

The zip bundles live alongside each template directory in this folder
(for example, `01-clinical-trial-protocol.zip`). Regenerate any zip
from the source tree with the recipes below. The minimal Linux / MacOS
recipe is:

```
cd generative-ai/templates
zip -r 01-clinical-trial-protocol.zip 01-clinical-trial-protocol/
```

On Windows PowerShell:

```
cd generative-ai\templates
Compress-Archive -Path 01-clinical-trial-protocol -DestinationPath 01-clinical-trial-protocol.zip
```

The resulting zip uploads directly to Overleaf via **New Project ->
Upload Project**.

## Inspiration

Inspired by leading CC BY templates such as the
[MDPI single column article template](https://www.overleaf.com/latex/templates/mdpi-article-template/fcpwsspfzsph)
and other journal-style CC BY templates, adapted into ten visually
distinct variants without preprint editing formatting and without line
numbers in columns.

## License

All templates in this folder are distributed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).
