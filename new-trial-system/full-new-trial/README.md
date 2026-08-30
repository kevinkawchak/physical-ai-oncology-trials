# full-new-trial - Stage 7 of 8 (the full paper)

[![Stage](https://img.shields.io/badge/Stage-7%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)
[![Paper](https://img.shields.io/badge/Paper-Draft%201.0-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Figures](https://img.shields.io/badge/Figures%20drawn-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial)
[![Sections](https://img.shields.io/badge/Sections-11-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial/sections)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-C9C9C9.svg)](https://creativecommons.org/licenses/by/4.0/)

## What this stage is

The full paper. Every figure slot placed in stage 6 is replaced by a drawn TikZ
figure, every table is populated from the repository sources stage 6 named, and
every bracketed drafting instruction is discharged and deleted: no `\draftinstr`
survives into this stage.

What changed from `draft-new-trial`:

1. **Twenty-five figures drawn.** Each from its own specification file, using the
   absolute-coordinate construction table and the edge-routing paragraph that
   specification carries. No figure was drawn from memory of another figure.
2. **Twenty-five tables populated.** Column widths cut against the amount of
   text each column actually carries, following the method of
   `funding/capitalization-plan/final-capital`.
3. **Prose written from source.** Direct quotation where the master prompt asks
   for it, from the four author-final `publication/` directories and the four
   input archives.
4. **Every figure checked twice** for text-box and arrow overlap against its own
   edge-routing paragraph. Curved edges carry an explicit bend angle or
   looseness value; box spacing is stated in the construction table.
5. **Depth added to the five main sections** so that IND, Trial Protocol,
   Legislation, Funding Proposals and AI Peer Review sit within a narrow band of
   one another in character count.

## Files

| File | Purpose |
|:--|:--|
| [main.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/main.tex) | Cover page, badges, keywords, clickable contents, `\input` of all eleven sections, `\clearpage` discipline |
| [trialstyle.sty](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/trialstyle.sty) | The burgundy palette, the five TikZ diagram vocabularies, the figure and table carriers, the DOI and prose link machinery, plus the stage 7 `\seqrow` and `\ganttrow` helpers |
| [references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/references.bib) | 122 entries, no duplicate keys, every DOI with a resolver url |
| `sections/sec-00-front.tex` | Abstract at 1347 characters, how to read this document, figure index, table index |
| `sections/sec-01-introduction.tex` | Introduction, the Federal AI and cancer record |
| `sections/sec-02-methods.tex` | Methods, the master prompt and sub-prompt schedule |
| `sections/sec-03-ind.tex` | IND, a main section |
| `sections/sec-04-trial-protocol.tex` | Trial Protocol, a main section |
| `sections/sec-05-legislation.tex` | Legislation, a main section |
| `sections/sec-06-funding-proposals.tex` | Funding Proposals, a main section |
| `sections/sec-07-ai-peer-review.tex` | AI Peer Review, a main section |
| `sections/sec-08-limitations-future-work.tex` | Limitations and Future Work |
| `sections/sec-09-conclusions.tex` | Conclusions |
| `sections/sec-10-references-backmatter.tex` | References and back matter |
| `full-new-trial-LaTeX.zip` | Self-contained Overleaf project for this stage |

## The paper's twenty-five figures, drawn at this stage

| Fig | § | Platform | Specification |
|:--|:--|:--|:--|
| 1 | 1 | Mermaid | [mermaid/fig-01](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-01-policy-chain-to-capability-gap.md) |
| 2 | 1 | D2 | [d2/fig-02](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-02-old-versus-new-system-grid.md) |
| 3 | 2 | PlantUML | [plantuml/fig-03](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-03-master-prompt-fork-join.md) |
| 4 | 2 | Mermaid | [mermaid/fig-04](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-04-one-generation-turn.md) |
| 5 | 2 | Graphviz | [graphviz/fig-05](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-05-figure-storage-record.md) |
| 6 | 3 | Diagrams | [diagrams-python/fig-06](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-06-ind-assembly-clusters.md) |
| 7 | 3 | Mermaid | [mermaid/fig-07](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-07-ind-assembly-clock.md) |
| 8 | 3 | D2 | [d2/fig-08](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-08-ind-1571-crosswalk.md) |
| 9 | 3 | Graphviz | [graphviz/fig-09](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-09-clinical-hold-fault-tree.md) |
| 10 | 4 | PlantUML | [plantuml/fig-10](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-10-participant-state-guards.md) |
| 11 | 4 | Mermaid | [mermaid/fig-11](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-11-escalation-ladder.md) |
| 12 | 4 | D2 | [d2/fig-12](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-12-protocol-inheritance-containers.md) |
| 13 | 4 | Diagrams | [diagrams-python/fig-13](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-13-on-premises-site-stack.md) |
| 14 | 5 | PlantUML | [plantuml/fig-14](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-14-statutory-actor-duties.md) |
| 15 | 5 | Graphviz | [graphviz/fig-15](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-15-bill-lineage-clusters.md) |
| 16 | 5 | D2 | [d2/fig-16](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-16-statute-to-sop-layers.md) |
| 17 | 6 | Mermaid | [mermaid/fig-17](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-17-funding-artifact-calendar.md) |
| 18 | 6 | D2 | [d2/fig-18](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-18-money-grid.md) |
| 19 | 6 | Graphviz | [graphviz/fig-19](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-19-award-work-package-ports.md) |
| 20 | 6 | Diagrams | [diagrams-python/fig-20](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-20-funding-production-pipeline.md) |
| 21 | 7 | Mermaid | [mermaid/fig-21](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/mermaid/fig-21-two-review-clocks.md) |
| 22 | 7 | D2 | [d2/fig-22](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/d2/fig-22-peer-review-economics-grid.md) |
| 23 | 7 | PlantUML | [plantuml/fig-23](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-23-tripartisan-review-concurrency.md) |
| 24 | 7 | Graphviz | [graphviz/fig-24](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/graphviz/fig-24-disagreement-resolution-tree.md) |
| 25 | 8 | Diagrams | [diagrams-python/fig-25](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-25-single-vendor-and-watermark.md) |

Six mermaid, four plantuml, six d2, four diagrams, five graphviz: by purpose,
not by quota.

## Files from other directories used here

| Source | Used for |
|:--|:--|
| [new-trial-system/template-new-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/template-new-system) | The cover page, contents, and back matter shape adapted in `main.tex` and `trialstyle.sty` |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | `capstyle.sty`, adapted into `trialstyle.sty`; the eight-stage build method; the figure, table and caption carriers |
| [new-trial-system/references](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/references) | Both source bib files, concatenated into `references.bib` |
| The five specification directories under [new-trial-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system) | The twenty-five figure slots, their tags, and their captions |
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Named in the section 3 drafting instructions |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) and [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Named in the section 4 drafting instructions |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Named in the section 5 and section 7 drafting instructions |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) and [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Named in the section 6 drafting instructions |
| [new-trial-system/abstracts](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/abstracts) | The trust chronology from late 2025 to August 2026 |

## Compile

Overleaf, pdfLaTeX: `pdflatex main`, then `bibtex main`, then `pdflatex main`
twice. `full-new-trial-LaTeX.zip` is a self-contained project.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
