# full-new-trial/sections - the eleven section files (stage 7 of 8)

[![Stage](https://img.shields.io/badge/Stage-7%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts)
[![Sections](https://img.shields.io/badge/Sections-11-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/full-new-trial/sections)
[![Figures](https://img.shields.io/badge/Figures-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

The eleven section files of the full paper, one per `\input` in
[`../main.tex`](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/main.tex),
in document order. Every figure slot is replaced by a drawn TikZ figure built from its own specification, every table is populated from the repository source the draft named, and no drafting instruction survives.

Each file was written and committed on its own, so the branch carries a usable
state after every section rather than after every stage.

| File | Contents | Figures | Tables |
|:--|:--|:--|:--|
| [sec-00-front.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-00-front.tex) | Abstract, reader's guide, figure index, table index | none | 1, 2, 3 |
| [sec-01-introduction.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-01-introduction.tex) | Introduction, the 2025 to 2026 Federal AI and cancer record | 1, 2 | 4, 5 |
| [sec-02-methods.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-02-methods.tex) | Methods, the master prompt and the storage argument | 3, 4, 5 | 6, 7 |
| [sec-03-ind.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-03-ind.tex) | IND, a main section | 6, 7, 8, 9 | 8, 9, 10 |
| [sec-04-trial-protocol.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-04-trial-protocol.tex) | Trial Protocol, a main section | 10, 11, 12, 13 | 11, 12, 13 |
| [sec-05-legislation.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-05-legislation.tex) | Legislation, a main section | 14, 15, 16 | 14, 15, 16 |
| [sec-06-funding-proposals.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-06-funding-proposals.tex) | Funding Proposals, a main section | 17, 18, 19, 20 | 17, 18, 19, 20 |
| [sec-07-ai-peer-review.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-07-ai-peer-review.tex) | AI Peer Review, a main section | 21, 22, 23, 24 | 21, 22, 23, 24 |
| [sec-08-limitations-future-work.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-08-limitations-future-work.tex) | Limitations and Future Work | 25 | 25 |
| [sec-09-conclusions.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-09-conclusions.tex) | Conclusions | none | none |
| [sec-10-references-backmatter.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/full-new-trial/sections/sec-10-references-backmatter.tex) | Back matter and references | none | glossary |

Sections 3 through 7 are the paper's main sections and are written to a similar
character count.

## Files from other directories used here

| Source | Used by |
|:--|:--|
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | `sec-03-ind.tex` |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | `sec-04-trial-protocol.tex` |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | `sec-04-trial-protocol.tex` |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | `sec-05-legislation.tex`, `sec-07-ai-peer-review.tex` |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | `sec-06-funding-proposals.tex`, and the style throughout |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | `sec-06-funding-proposals.tex` |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | `sec-06-funding-proposals.tex`, `sec-07-ai-peer-review.tex` |
| [new-trial-system/references](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/references) | Every citation in every section |
| The five specification directories under [new-trial-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system) | Every figure in every section |
