# prompts - the master prompt and the build output of record (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../README.md)
[![Files](https://img.shields.io/badge/Files-2-00417A.svg)](.)
[![Prompt](https://img.shields.io/badge/prompt--auto--fund.md-verbatim-3C7DB2.svg)](prompt-auto-fund.md)
[![Output](https://img.shields.io/badge/output--auto--fund.md-build%20record-3C7DB2.svg)](output-auto-fund.md)
[![Business days](https://img.shields.io/badge/Business%20days-5-6C757D.svg)](..)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Two files, and no third. This directory is the record of what was asked and what
was produced, so that a reader who has neither the conversation nor the terminal
can reconstruct both.

## The two files

| File | Heading | Contents | Rule |
|:--|:--|:--|:--|
| [`prompt-auto-fund.md`](prompt-auto-fund.md) | `## prompt-auto-fund` | The master prompt, word for word, with nothing added, nothing removed, and nothing reordered | The prompt's own closing paragraph |
| [`output-auto-fund.md`](output-auto-fund.md) | `## output-auto-fund` | The full Claude Code markdown output of the build: the decisions taken, the defects found and measured, the instructions that needed interpretation, and what is not claimed | The same paragraph |

Neither file carries a preface, a summary, a note, or a second heading. The
prompt file opens with its heading and continues with the prompt. The output file
opens with its heading and continues with the output.

## Why the prompt is filed at all

The build it drives is long: five business days, each with its own emails,
briefs, form packs, capital instructions, figure specifications, and compiled
packet. A reader who finds an odd decision in one of those days, such as why day
4 stages work instead of sending it or why one figure uses Graphviz rather than
Mermaid, can settle the question against the instruction that caused it rather
than against a paraphrase of that instruction. The prompt is therefore a
primary source in this directory and is treated as one.

## What the output file records, and what it does not

The output file records:

- The reading pass over `../../capitalization-plan`, `../../pdac-funding-applications`, `../../move-in` and `../../potential-partners`, and what each contributed.
- The document count and type chosen for each business day, with the reason.
- The palette, cover and spacing decisions in `fundstyle.sty`.
- Every compile, with its error count, its overfull box count and its page count.
- Every defect found in the second-to-last pass, with its measured size.
- The instructions that could not be followed literally, with what was done instead.

The output file does not record the contents of the generated `.tex`, `.txt`,
`.bib` or `.sty` files. Those files are the deliverable and are read where they
live.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `prompts/README.md` | [`../../capitalization-plan`](../../capitalization-plan) | The two-file convention, the heading rule, and the separation of prompt from output |
| `prompts/prompt-capital.md` | [`../../capitalization-plan`](../../capitalization-plan) | The verbatim-filing convention followed by `prompt-auto-fund.md` |
| `prompts/output-capital.md` | [`../../capitalization-plan`](../../capitalization-plan) | The build-record structure followed by `output-auto-fund.md` |
| `prompts/README.md` | [`../../move-in`](../../move-in) | The practice of recording defects with their measured size rather than as a list of adjectives |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
