# final-new-trial - Stage 8 of 8 (the final paper)

[![Stage](https://img.shields.io/badge/Stage-8%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)
[![Paper](https://img.shields.io/badge/Paper-Draft%201.0-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Figures](https://img.shields.io/badge/Figures%20drawn-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial)
[![Sections](https://img.shields.io/badge/Sections-11-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/sections)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-C9C9C9.svg)](https://creativecommons.org/licenses/by/4.0/)

## What this stage is

The senior author's proof-reading pass over `full-new-trial`. No new argument is
introduced; every existing one is made to sit correctly on the page. There is no
`publication/` subdirectory at this stage, by instruction: that space belongs to
the author's own final edits.

What changed from `full-new-trial`:

1. **`\clearpage` discipline.** A barrier is placed only where the next section
   opens with a float or a full-width table. Sections 2, 5 and 9 open with prose
   and take no barrier, because one there would leave the preceding page more
   than a third empty. The float queue is still barriered at every `\section` by
   the style file, so no figure can drift past its own heading.
2. **Float budget retuned.** The inline-figure `\needspace` reserve drops from
   five lines to four, `\topfraction` rises to 0.90 and `\floatpagefraction` to
   0.50, because the drawn figures have a known median height of 8.4 cm that the
   stage 7 reserve over-provisioned.
3. **Table column widths re-cut** against the compiled widths rather than the
   estimated ones. Nine tables changed; each change is recorded in the header
   comment of the section that carries it.
4. **Caption balance** checked for all fifty captions: every one is two lines
   within a four-character spread, opening with `Figure N.` or `Table N.`.
5. **Cross-references made symbolic** so a renumber cannot silently break them.
6. **Depth added** so the five main sections sit within a 2,000-character band
   of one another and the paper reaches its character target.
7. **US clinical terms, dashes and the section symbol** checked throughout: no
   em dash, no en dash, no double or triple hyphen in prose, and `\S` for every
   codified reference.

The spacing invariant is explicitly unchanged: `\end{appfig}` then
`\vspace{-0.6cm}` then `\figcaption` still leaves exactly 7.44 pt from the frame
rule to the first caption line, for every one of the twenty-five figures and
every one of the twenty-five tables.

## Files

| File | Purpose |
|:--|:--|
| [main.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/main.tex) | Cover page, badges, keywords, clickable contents, `\input` of all eleven sections, `\clearpage` discipline |
| [trialstyle.sty](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/trialstyle.sty) | The burgundy palette, the five TikZ diagram vocabularies, the figure and table carriers, the DOI and prose link machinery, plus the stage 7 `\seqrow` and `\ganttrow` helpers |
| [references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/references.bib) | 122 entries, no duplicate keys, every DOI with a resolver url |
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
| `final-new-trial-LaTeX.zip` | Self-contained Overleaf project for this stage |

## The paper's twenty-five figures, final

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

## Character budget

| Stage | Characters across `main.tex` and `sections/` |
|:--|:--|
| draft-new-trial | 83,287 |
| full-new-trial | 182,854 |
| final-new-trial | 212,371 |
| Target, 1.25 times `funding/capitalization-plan/final-capital/publication` | 211,643 |

## Compile

Overleaf, pdfLaTeX: `pdflatex main`, then `bibtex main`, then `pdflatex main`
twice. `final-new-trial-LaTeX.zip` is a self-contained project.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
