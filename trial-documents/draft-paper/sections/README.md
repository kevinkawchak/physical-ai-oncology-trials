# trial-documents/draft-paper/sections - Draft section scaffolds

[![Sections](https://img.shields.io/badge/Sections-8%20.tex-8B2E3F.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-2%20Draft-2F5D7C.svg)](..)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

One `.tex` file per paper section (Rule 6), each `\input` by
[`../main.tex`](../main.tex). At the draft stage each file is a scaffold with
bracketed `[DRAFTING INSTRUCTION]` pointers (`\draftinstr`) that name the exact
repository files the full stage will process.

## Section files (PAPER FORMAT order)

| File | Paper section | Draft contents |
|:--|:--|:--|
| [`sec-01-abstract.tex`](sec-01-abstract.tex) | Abstract, Keywords | Abstract scaffold and keywords pointer |
| [`sec-02-introduction.tex`](sec-02-introduction.tex) | Introduction | OUTLINE themes 1-3 and scope |
| [`sec-03-methods.tex`](sec-03-methods.tex) | Methods | Repository LLM, sub-prompts, gates, before/during/after workflow |
| [`sec-04-results.tex`](sec-04-results.tex) | Results | Stage outputs, acceleration targets, iterations, verifications, evidence |
| [`sec-05-discussion.tex`](sec-05-discussion.tex) | Discussion | Benefit greater than risk, admin/prep compression, prior developments |
| [`sec-06-limitations.tex`](sec-06-limitations.tex) | Limitations and Future Work | LLM and method limits, future work |
| [`sec-07-conclusions.tex`](sec-07-conclusions.tex) | Conclusions | Three closing claims |
| [`sec-08-references-backmatter.tex`](sec-08-references-backmatter.tex) | References, Back Matter | Bibliography and back matter incl. Rights and Permissions (CC) |

The Table of Contents is generated in [`../main.tex`](../main.tex) via
`\tableofcontents`, placed after the Introduction per the PAPER FORMAT.

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
