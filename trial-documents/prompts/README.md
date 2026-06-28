# trial-documents/prompts - Master prompt and full output

[![Master prompt](https://img.shields.io/badge/Master%20prompt-prompt--paper.md-2F5D7C.svg)](prompt-paper.md)
[![Output](https://img.shields.io/badge/Output-output--paper.md-8B2E3F.svg)](output-paper.md)
[![Paper](https://img.shields.io/badge/Paper-v1.0-D08770.svg)](../draft-paper)
[![Repository](https://img.shields.io/badge/Repository-v4.2.0-2F5D7C.svg)](../../README.md)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

The single master prompt that drives the whole build, and the full Claude Code
output narrative.

## Contents

| File | Contents |
|:--|:--|
| [`prompt-paper.md`](prompt-paper.md) | The entire master prompt, word-for-word, under a `## prompt-paper` heading (and nothing else) |
| [`output-paper.md`](output-paper.md) | The entire Claude Code markdown output for this prompt, under an `## output-paper` heading (the chat narrative, not the code files) |
| `a.md` | Pre-existing placeholder retained from the directory's prior state |

## How the prompt is executed

**Process A** reads `prompt-paper.md` and writes the four stage sub-prompts in
[`../sub-prompts`](../sub-prompts). **Process B** executes them in order, building
the four stages in [`../mermaid`](../mermaid), [`../draft-paper`](../draft-paper),
[`../full-paper`](../full-paper), and [`../final-paper`](../final-paper). Every
distinguishable file is a separate commit pushed in real time (Rules 6, 7, 8).

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice.
