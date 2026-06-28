# trial-documents/inputs/llm-adoption - Paper template and adoption guide

[![Template](https://img.shields.io/badge/Role-Paper%20template-2F5D7C.svg)](.)
[![Guide](https://img.shields.io/badge/Role-Adoption%20guide-8B2E3F.svg)](.)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20843290-blue.svg)](https://doi.org/10.5281/zenodo.20843290)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

The *Oncology Trial PI LLM Adoption Guide* (DOI
[10.5281/zenodo.20843290](https://doi.org/10.5281/zenodo.20843290)). It serves two
roles for the paper *Phase 1 Pancreatic Cancer Trial Efficient LLM Document
Generations* (paper v1.0):

1. **Paper template.** The paper adopts this single-column article layout and
   extends it with a Keywords section and a Rights and Permissions (CC) section in
   the back matter. The paper has many more sections and pages than the template.
2. **Practical hands-on guide.** The paper builds on the guide's Repository Setup,
   LLM Limitations, Prompt Proficiency, Project Proficiency, and Figure and Table
   Tips to lead oncology trial PIs into proficient LLM large-document creation.

## Files

| File | Role |
|:--|:--|
| [`main.tex`](main.tex) | The adoption-guide source and the layout template extended by the paper |
| [`sample.bib`](sample.bib) | The guide's bibliography (author works), mirrored selectively into the paper bibliography |
| `LaTeX Source Files.zip` | The Overleaf bundle of the guide |

## What the paper takes from the template

| Template element | Where the paper uses it |
|:--|:--|
| Single-column article, black body text | Kept as the paper template color (figures carry color) |
| Abstract, Table of Contents, sectioned body, back matter | The paper format, extended with Keywords and a CC section |
| Repository Setup and Prompt Proficiency guidance | Methods (repository LLM, sub-prompt schedule, auto-commit) |
| LLM Limitations and Project Proficiency | Limitations and Future Work |
| Figure and Table Tips | The five-color Mermaid-to-TikZ figures and full-width ragged-right tables |

The paper does not use `trial-protocol/final-protocol/publication` as the template;
that directory is consulted only for image, white-space, and formatting code
strategies.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
