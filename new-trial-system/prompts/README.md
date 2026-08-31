# prompts - master prompt and build output

[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Model](https://img.shields.io/badge/Model-Claude%20Code%20Opus%205-A32A3C.svg)](https://claude.ai/code)
[![Stages](https://img.shields.io/badge/Stages-8-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts)

## Files

| File | Contents |
|:--|:--|
| [prompt-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/prompt-new-trial.md) | A single `## prompt-new-trial` heading followed by the author's master prompt, word for word and nothing else. |
| [output-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/output-new-trial.md) | A single `## output-new-trial` heading followed by the model's markdown output for the whole build, and nothing else. Code files are not reproduced there; they are in their own directories. |
| [prompt-2-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/prompt-2-new-trial.md) | A single `## prompt-2-new-trial` heading followed by the author's second master prompt, word for word and nothing else. |
| [output-2-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/output-2-new-trial.md) | A single `## output-2-new-trial` heading followed by the model's markdown output for the update stage, and nothing else. |

## The second prompt

The second prompt is an update rather than a new build. It directs six
objectives: adapt the paper to the same template and the first prompt's
formatting rules and revise it at least once; file this prompt and its output
beside the first pair; rewrite the AI Peer Review section around four quantified
disadvantages of the prior regime and three quantified advantages of the new
one, keeping the Gemini and OpenAI reviewer context; replace Figure 14's stick
actors with a comprehensive and professional diagram type while keeping its
context; write thirteen outbound communications; and carry the whole of the
first pull request's progress into a new one. Its output directory is
[final-new-trial/update-final](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final).

## How the master prompt was executed

One turn. The model read the prompt, decomposed it into the eight sub-prompts
recorded in
[sub-prompts](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts),
and executed them in order. Every file produced was committed and pushed the
moment it was finished, so the branch is a live record of the build rather than
a single drop at the end.

The method is adapted from the eight-stage build the author used for
[funding/capitalization-plan](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan),
whose own master prompt is at
[funding/capitalization-plan/prompts/prompt-capital.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/funding/capitalization-plan/prompts/prompt-capital.md).

## Files from other directories used here

| Source | Used for |
|:--|:--|
| [funding/capitalization-plan/prompts](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/prompts) | The prompt and output file convention adapted here |
| [funding/pdac-funding-applications/prompts](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/prompts) | The first use of a single master prompt with model-generated sub-prompts |
