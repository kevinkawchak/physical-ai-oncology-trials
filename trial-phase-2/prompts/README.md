# prompts - master prompt and output (Physical AI oncology Phase 2 trial protocol, v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Master prompt](https://img.shields.io/badge/Master%20prompt-verbatim-800020.svg)](prompt-protocol.md)
[![Output](https://img.shields.io/badge/Output-narrative-6B6B6B.svg)](output-protocol.md)
[![Protocol](https://img.shields.io/badge/Protocol-Phase%202%20Randomized-800020.svg)](.)
[![Repo](https://img.shields.io/badge/Repo-v4.1.0-800020.svg)](../../README.md)

This directory holds the single master prompt that drives the entire Phase II
build and the narrative output it produced.

## Files

| File | Purpose |
|:--|:--|
| [`prompt-protocol.md`](prompt-protocol.md) | The master prompt, filed word-for-word under a `## prompt-protocol` heading. **Process A** generates every sub-prompt under [`../sub-prompts/`](../sub-prompts); **Process B** runs those sub-prompts to grow the protocol. |
| [`output-protocol.md`](output-protocol.md) | The narrative output of the build under an `## output-protocol` heading (the Claude Code markdown narrative, not the LaTeX source). |

## How the build runs

```
prompt-protocol.md  (this directory, master)
        |
        v  Process A generates
../sub-prompts/  prompt-1-mermaid .. prompt-4-final-protocol
        |
        v  Process B runs in sequence
../mermaid  ->  ../draft-protocol  ->  ../full-protocol  ->  ../final-protocol  ->  ../final-protocol/publication
```

Every distinguishable file is a separate commit pushed in real time; for each
stage the second-to-last commit fixes all errors and the last commit performs the
remaining repository updates (CHANGELOG, releases, root README, version v4.1.0).

## Files from other directories used here

| Source | Used for |
|:--|:--|
| [`../../trial-protocol/prompts/prompt-protocol.md`](../../trial-protocol/prompts) | the Phase I master prompt whose structure and rules this prompt adapts |
| [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | the Phase I paper this protocol builds the Phase II paper from |
| [`../../trial-protocol/nih-protocol`](../../trial-protocol/nih-protocol) | the NIH-FDA Phase 2/3 IND/IDE template that governs section order |

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
