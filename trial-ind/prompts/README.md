# prompts - master prompt and full Claude output (Phase 1 PDAC IND, IND v1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Master prompt](https://img.shields.io/badge/Master%20prompt-verbatim-000000.svg)](prompt-ind.md)
[![Output](https://img.shields.io/badge/Output-Claude%20Code%20Opus%204.8-6C757D.svg)](output-ind.md)
[![Repository](https://img.shields.io/badge/Repository-v4.3.0-blue.svg)](../../README.md)

This directory preserves the single submitted **master prompt** and the full
**Claude Code output** for the *Phase 1 PDAC IND: AI Generation* build (IND v1.0,
repository v4.3.0), following the filing convention of
[`trial-protocol/prompts`](../../trial-protocol/prompts) and
[`trial-documents/prompts`](../../trial-documents/prompts).

## Files

| File | Heading | Contents |
|:--|:--|:--|
| [`prompt-ind.md`](prompt-ind.md) | `## prompt-ind` | The entire submitted master prompt, word-for-word and nothing else. |
| [`output-ind.md`](output-ind.md) | `## output-ind` | The entire Claude Code markdown narrative output of the build (added in the final release commit). |
| `a.md` | n/a | Pre-existing placeholder retained from the repository. |

## How these files are used downstream (Rule 5)

- `prompt-ind.md` is the master prompt that **Process A** read to generate the four
  sub-prompts in [`../sub-prompts/`](../sub-prompts), and that every stage
  references as its authority.
- `output-ind.md` records the assistant's narrative for the whole run; the
  per-stage narratives live in each stage's `output-*.md` file
  (`../mermaid/output-mermaid.md`, `../draft-ind/output-draft-ind.md`,
  `../full-ind/output-full-ind.md`, `../final-ind/output-final-ind.md`).

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
