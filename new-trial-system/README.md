# new-trial-system - Pancreatic Cancer LLM Clinical Trial System (v4.6.0)

[![Paper](https://img.shields.io/badge/Paper-Draft%201.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Figures](https://img.shields.io/badge/Figures-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Stages](https://img.shields.io/badge/Build%20stages-8-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts)
[![Model](https://img.shields.io/badge/Model-Claude%20Code%20Opus%205-800020.svg)](https://claude.ai/code)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-C9C9C9.svg)](https://orcid.org/0009-0007-5457-8667)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-C9C9C9.svg)](https://creativecommons.org/licenses/by/4.0/)

## What this directory holds

The source, specifications and build record for **Pancreatic Cancer LLM Clinical
Trial System: From IND to Protocol to Legislation, Funding, and AI Peer
Review**, Draft 1.0, San Diego, August 14, 2026.

The paper discloses the method by which one author, directing Claude Code Opus 5
through a single master prompt, produced an IND, two clinical trial protocols,
five versions of a Federal bill with four companion documents, and fourteen
funding artifacts, on a 1 to 4 day project time scale rather than the prior
system's month scale, with AI peer review during development rather than after
completion.

## Directory map

```
new-trial-system/
  README.md                    this file
  abstracts/                   the author's deposited abstracts, 2024 to 2026
  inputs/                      the three legislation archives and the AI peer review archive
  prompts/                     the master prompt and the full build output
  references/                  the two source bibliographies
  template-new-system/         the paper template this work adapts
  sub-prompts/                 the eight-stage schedule, one directory per stage
  mermaid/                     6 figure specifications
  plantuml/                    4 figure specifications
  d2/                          6 figure specifications
  diagrams-python/             4 figure specifications
  graphviz/                    5 figure specifications
  draft-new-trial/             stage 6 source, with an Overleaf bundle
  full-new-trial/              stage 7 source, with an Overleaf bundle
  final-new-trial/             stage 8 source, with an Overleaf bundle
```

## The paper

| Section | Title | Figures | Tables |
|:--|:--|:--|:--|
| 0 | Abstract, reader's guide, indexes | none | 1, 2, 3 |
| 1 | Introduction | 1, 2 | 4, 5 |
| 2 | Methods | 3, 4, 5 | 6, 7 |
| 3 | IND | 6, 7, 8, 9 | 8, 9, 10 |
| 4 | Trial Protocol | 10, 11, 12, 13 | 11, 12, 13 |
| 5 | Legislation | 14, 15, 16 | 14, 15, 16 |
| 6 | Funding Proposals | 17, 18, 19, 20 | 17, 18, 19, 20 |
| 7 | AI Peer Review | 21, 22, 23, 24 | 21, 22, 23, 24 |
| 8 | Limitations and Future Work | 25 | 25 |
| 9 | Conclusions | none | none |
| 10 | Back matter and references | none | glossary |

Sections 3, 4, 5, 6 and 7 are the paper's main sections and are written to a
similar character count.

## The five diagram platforms

| Platform | Figures | What it owns |
|:--|:--|:--|
| [Mermaid](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/mermaid) | 1, 4, 7, 11, 17, 21 | Order in time, and decisions taken at a point in time |
| [PlantUML](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml) | 3, 10, 14, 23 | Formal notation: actors, guards, concurrency |
| [D2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/d2) | 2, 8, 12, 16, 18, 22 | Nesting and tabulation |
| [Diagrams (Python)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/diagrams-python) | 6, 13, 20, 25 | Clustered infrastructure carrying glyphs |
| [Graphviz](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/graphviz) | 5, 9, 15, 19, 24 | Records, clusters, fault and decision trees |

No figure in this paper is a raster. Every one is specified in its platform's
own syntax and drawn in TikZ from an absolute-coordinate construction table.

## Palette

Burgundy `#800020`, lighter burgundy 1 `#A32A3C`, lighter burgundy 2 `#E2D6D9`,
Charcoal `#2E2E2E`, Slate Gray `#6B6B6B`, Mist Gray `#C9C9C9`, white. Charcoal is
a stroke and a text color only, so no figure carries a black or near-black fill.
The paper body is black text throughout.

## Files from other directories used here

| Source directory | Used for |
|:--|:--|
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Section 3, Figures 6 to 9, Tables 8 to 10 |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Section 4, Figures 10 to 13, Tables 11 to 13 |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Section 4, Figures 11 and 12, Tables 12 and 13 |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | Section 6, Figures 17 to 20, Tables 17 to 20, and the whole build method and style |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Section 6, Figure 17, Table 17 |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Sections 6 and 7, Figures 19 and 23, Tables 19 and 22 |
| [national-platform](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/national-platform) | Figures 8, 13 and 16, the adapted 21 CFR and ICH text |

## Compile

Each of the three stage directories is a self-contained Overleaf project.
`pdflatex main`, then `bibtex main`, then `pdflatex main` twice.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
