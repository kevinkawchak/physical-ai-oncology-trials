# trial-documents/inputs - Paper template and bibliography inputs

[![Template](https://img.shields.io/badge/Template-LLM%20Adoption%20Guide-2F5D7C.svg)](llm-adoption)
[![Bibliography](https://img.shields.io/badge/Bibliography-references.bib-8B2E3F.svg)](references.bib)
[![Paper](https://img.shields.io/badge/Paper-v1.0-D08770.svg)](../draft-paper)
[![Repository](https://img.shields.io/badge/Repository-v4.2.0-2F5D7C.svg)](../../README.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

The curated inputs the repository-based LLM reads to build the paper *Phase 1
Pancreatic Cancer Trial Efficient LLM Document Generations* (paper v1.0, repository
v4.2.0). Keeping a small set of appropriately sized, high-quality inputs in one
directory increases AI processing efficiency, document quality, and length, as
described in the adoption guide.

## Contents

| Item | Role |
|:--|:--|
| [`llm-adoption/`](llm-adoption) | The paper template (the *Oncology Trial PI LLM Adoption Guide*) and the practical hands-on guide this paper builds upon |
| [`references.bib`](references.bib) | The author's ORCID-derived works (Aug 2024 - Jun 2026) used to establish LLM trust and to cite the prior developments |

## How these inputs are used

| Input | Used in | Where |
|:--|:--|:--|
| `llm-adoption/main.tex` | Paper template, method | The paper format (Abstract, Keywords, ToC, body, back matter) plus the repository-LLM, prompt-proficiency, and figure/table guidance in Methods and Limitations |
| `llm-adoption/sample.bib` | Citation keys | Source of the prior-works citation keys mirrored into the paper [`references.bib`](../draft-paper/references.bib) |
| `references.bib` | Evidence trail | The 2024-2026 author works detailed in Results and Discussion as evidence leading up to this single-prompt project |

## Main documents incorporated (per the master prompt)

1. 2030 60 Second PDAC Robotic Whipple and Daraxonrasib Simulation, DOI
   [10.5281/zenodo.20196639](https://doi.org/10.5281/zenodo.20196639).
2. H. R. 9510 Bill v5.0 2026, DOI
   [10.5281/zenodo.20619762](https://doi.org/10.5281/zenodo.20619762).
3. The DARAXONRASIB references (five entries) for the PDAC drug candidate.
4. National Platform for Physical AI Oncology Trials, DOI
   [10.5281/zenodo.19244918](https://doi.org/10.5281/zenodo.19244918).

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
