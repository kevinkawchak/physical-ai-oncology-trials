# prompts - master prompt and full Claude output (10 PDAC Funding Applications, v4.4.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Master prompt](https://img.shields.io/badge/Master%20prompt-verbatim-00417A.svg)](prompt-apply.md)
[![Output](https://img.shields.io/badge/Output-Claude%20Code%20Opus%205-3C7DB2.svg)](output-apply.md)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-blue.svg)](../../../README.md)

This directory preserves the single submitted **master prompt** and the full
**Claude Code output** for the *10 Funding Applications* build (paper Draft 1.0,
repository v4.4.0), following the filing convention of
[`trial-ind/prompts`](../../../trial-ind/prompts).

## Files

| File | Heading | Contents |
|:--|:--|:--|
| [`prompt-apply.md`](prompt-apply.md) | `## prompt-apply` | The entire submitted master prompt, word-for-word and nothing else |
| [`output-apply.md`](output-apply.md) | `## output-apply` | The Claude Code markdown narrative of the build: what was built, what was verified, the six defects found and fixed, the one substantive correction, and the two things that could not be done as literally specified |

Neither file contains code. The LaTeX sources, figure specifications, and email
text live in the stage directories.

## How these files are used downstream (Rule 5)

- `prompt-apply.md` is the authority every one of the thirteen sub-prompts in
  [`../sub-prompts/`](../sub-prompts) answers to, and the source of the
  cover-page specification, the figure budget, the diagram-split rule, and the
  DOI placeholder convention.
- `output-apply.md` records the build narrative, including the verification
  table and the defects list. The per-stage detail lives in each stage's own
  README rather than being repeated here.

## What the output records that the artifacts do not

Three things are in `output-apply.md` and nowhere else:

1. **The six defects found during the build**, with the cause of each. A reader
   auditing `applystyle.sty` will find `\unskip` in `apptable` and the
   `\appfile` character scanner without knowing why either is there.
2. **The first-in-human correction**, which nine of the ten applications carried
   until the audit pass. It is also stated in §7 of the summary paper.
3. **The two instructions that could not be followed literally**, and what was
   done instead: the `robotic-surgeries` README instruction, which Rule 1
   forbids, and the one-quarter length target, which `final-apply` reaches at
   1/3.14 rather than 1/4.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
