# update-final/sections - the eleven section files (second-prompt update)

[![Stage](https://img.shields.io/badge/Stage-Update%20over%208%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final)
[![Sections](https://img.shields.io/badge/Sections-11-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final/sections)
[![Figures](https://img.shields.io/badge/Figures-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

The eleven section files of the update stage, one per `\input` in
[../main.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/main.tex),
in document order. Every file carries a header comment naming what the update
changed in it and why, so the diff against
[../../sections](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/sections)
can be read without the commit messages.

| File | Contents | Figures | Tables | What the update changed here |
|:--|:--|:--|:--|:--|
| [sec-00-front.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-00-front.tex) | Abstract, reader's guide, figure index, table index | none | 1, 2, 3 | Table 3's row 21 re-described; the build note records the three-day window, the second prompt, and the update stage |
| [sec-01-introduction.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-01-introduction.tex) | Introduction, the 2025 to 2026 Federal AI and cancer record | 1, 2 | 4, 5 | Figure 2's emphasis callout moved below the grid rule, clear of the in-figure note it overlapped |
| [sec-02-methods.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-02-methods.tex) | Methods, the master prompt and the storage argument | 3, 4, 5 | 6, 7 | Figure 4's loop frame label moved 9.5 mm inside the west edge, clear of the Author activation bar |
| [sec-03-ind.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-03-ind.tex) | IND, a main section | 6, 7, 8, 9 | 8, 9, 10 | Two `\foreach` headers re-declared so Figure 7 compiles; Figure 6's subject label moved above its halo and all four inbound edges re-terminated on the halo at their own bearing; a closing paragraph ties the dossier to the review clock |
| [sec-04-trial-protocol.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-04-trial-protocol.tex) | Trial Protocol, a main section | 10, 11, 12, 13 | 11, 12, 13 | Table 13's columns re-cut so the `Characters` header stops overflowing; a closing paragraph on protocol review provenance |
| [sec-05-legislation.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-05-legislation.tex) | Legislation, a main section | 14, 15, 16 | 14, 15, 16 | Figure 14 redrawn from a use case diagram to a class diagram; one `\foreach` header re-declared so Figure 16 compiles |
| [sec-06-funding-proposals.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-06-funding-proposals.tex) | Funding Proposals, a main section | 17, 18, 19, 20 | 17, 18, 19, 20 | One `\foreach` header re-declared so Figure 17 compiles; a closing paragraph on the review round's own economics |
| [sec-07-ai-peer-review.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-07-ai-peer-review.tex) | AI Peer Review, a main section | 21, 22, 23, 24 | 21, 22, 23, 24 | Rewritten around four quantified costs and three quantified gains; Table 21 re-cut to nine axes and Figure 22's grid to seven of them; the model roster, the concurrency figure and the disagreement tree carried forward |
| [sec-08-limitations-future-work.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-08-limitations-future-work.tex) | Limitations and Future Work | 25 | 25 | Closing paragraph tightened so it no longer spills a two-line orphan onto its own page |
| [sec-09-conclusions.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-09-conclusions.tex) | Conclusions | none | none | Unchanged |
| [sec-10-references-backmatter.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/sections/sec-10-references-backmatter.tex) | Back matter, glossary, reference list | none | glossary | Unchanged |

## Section 7's length

The five sections marked as main sections in the master prompt are written to a
similar length. After the update, their prose measures, excluding comments,
figure code and table bodies:

| Section | Prose characters |
|:--|:--|
| 3, IND | about 12,500 |
| 4, Trial Protocol | about 12,400 |
| 5, Legislation | about 14,400 |
| 6, Funding Proposals | about 12,000 |
| 7, AI Peer Review | about 18,700 |

Section 7 runs longest because the second prompt directs its expansion, and
three of its older subsections were merged and trimmed to hold the growth down.
Sections 3, 4 and 6 each gained a closing paragraph tying their artifact to the
review clock, which is substantive cross-reference rather than padding.

## Files from other directories used here

| Source | Used for |
|:--|:--|
| [final-new-trial/sections](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/sections) | The stage 8 section set these eleven files are taken from |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | The AI peer review archive quoted throughout sec-07 |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | The model roster of Table 22 |
| [new-trial-system/plantuml](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml) | Figure 14's class diagram specification |
