# prompts - master prompt and full Claude output

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Master prompt](https://img.shields.io/badge/Master%20prompt-verbatim-00417A.svg)](prompt-protocol.md)
[![Output](https://img.shields.io/badge/Output-Claude%20Code%20Opus%204.8-6C757D.svg)](output-protocol.md)

This directory preserves the single submitted **master prompt** and the full
**Claude Code output** for the Physical AI oncology trial protocol build
(version v4.0.0), following the auto-bill-02 filing convention.

## Files

| File | Heading | Contents |
|:--|:--|:--|
| [`prompt-protocol.md`](prompt-protocol.md) | `## prompt-protocol` | The entire submitted master prompt, word-for-word and nothing else. |
| [`output-protocol.md`](output-protocol.md) | `## output-protocol` | The entire Claude Code markdown narrative output of the build (added in the final release commit). |
| `a.md` | n/a | Pre-existing placeholder retained from the repository. |

## How these files are used downstream (Rule 5)

- `prompt-protocol.md` is the master prompt that **Process A** read to generate
  the four sub-prompts in [`../sub-prompts/`](../sub-prompts), and that every
  stage references as its authority.
- `output-protocol.md` records the assistant's narrative for the whole run; the
  per-stage narratives live in each stage's `output-*.md` file
  (`../mermaid/output-mermaid.md`, `../draft-protocol/output-draft-protocol.md`,
  `../full-protocol/output-full-protocol.md`,
  `../final-protocol/output-final-protocol.md`).

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
