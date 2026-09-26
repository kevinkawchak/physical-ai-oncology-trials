# prompts - the master prompt and the build output of record (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Files](https://img.shields.io/badge/Files-2-00417A.svg)](.)
[![Prompt](https://img.shields.io/badge/prompt--auto--fund--2.md-verbatim-3C7DB2.svg)](prompt-auto-fund-2.md)
[![Output](https://img.shields.io/badge/output--auto--fund--2.md-build%20record-3C7DB2.svg)](output-auto-fund-2.md)
[![Business days](https://img.shields.io/badge/Business%20days-5-6C757D.svg)](..)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Two files, and no third. This directory is the record of what was asked and what
was produced, so that a reader who has neither the conversation nor the terminal
can reconstruct both.

## The two files

| File | Heading | Contents | Rule |
|:--|:--|:--|:--|
| [`prompt-auto-fund-2.md`](prompt-auto-fund-2.md) | `## prompt-auto-fund-2` | The master prompt, word for word, with nothing added, removed, or reordered, including its curly quotation marks and the September 12, 2026 message it quotes | The prompt's own closing instructions |
| [`output-auto-fund-2.md`](output-auto-fund-2.md) | `## output-auto-fund-2` | The full Claude Code markdown output of the build: the decisions taken, the defects found and measured, the instructions that needed interpretation, and what is not claimed | The same instructions |

Neither file carries a preface, a summary, a note, or a second heading. Each
opens with its heading and continues with its content.

## Why the prompt is filed verbatim

The prompt contains two kinds of text that must not be paraphrased. The first is
the September 12, 2026 message to the President, which every day of this block
quotes and which day 3 answers; a paraphrase of it would put words in the chief
executive's mouth. The second is the description of the SEC office's update,
which carries the quoted phrase "very seriously"; the block quotes that phrase
and no other words from the office, and the filed prompt is the evidence of what
was reported and what was not.

A reader who finds an odd decision in one of the five days, such as why the two
applicants are answered on LinkedIn rather than by email, or why the White House
is reached through a form, can settle it against the instruction that caused it:
here, the requirement that every email carry at least three exact and correct
addresses, which rules out email to any counterparty that publishes none.

## What the output file records, and what it does not

The output file records:

- The reading pass over [`../../auto-fund`](../../auto-fund), [`../../capitalization-plan`](../../capitalization-plan), [`../../move-in`](../../move-in) and [`../../potential-partners`](../../potential-partners), and what each contributed.
- How each of the 36 recipient addresses was verified, and which candidate addresses were rejected because no publishing page could be found.
- The document count and type chosen for each business day, with the reason.
- Every compile, with its error count, its overfull box count, and its page count.
- Every defect found in the second-to-last pass, with its measured size.
- The instructions that could not be followed literally, with what was done instead.

The output file does not record the contents of the generated `.tex`, `.txt`,
`.bib`, `.sty` or `.md` files. Those files are the deliverable and are read where
they live.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `prompts/README.md` | [`../../auto-fund`](../../auto-fund) | The two-file convention and the heading rule |
| `prompts/prompt-auto-fund.md` | [`../../auto-fund`](../../auto-fund) | The verbatim-filing convention followed by `prompt-auto-fund-2.md` |
| `prompts/output-auto-fund.md` | [`../../auto-fund`](../../auto-fund) | The build-record structure followed by `output-auto-fund-2.md` |
| `prompts/README.md` | [`../../capitalization-plan`](../../capitalization-plan) | The separation of prompt from output |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
