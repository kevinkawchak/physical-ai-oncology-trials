# trial-documents/research/document-types - Long-document decision making

[![AI sources](https://img.shields.io/badge/AI%20sources-2-2F5D7C.svg)](.)
[![Topic](https://img.shields.io/badge/Topic-Decision%20gates-8B2E3F.svg)](.)
[![Paper](https://img.shields.io/badge/Paper-v1.0-D08770.svg)](../../draft-paper)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)

Two AI sources answering the prompt in [`prompt-types.md`](prompt-types.md): which
oncology trial steps rely on long documents that must be made before moving within
a phase or between phases, and whether the trial moves faster if the documents are
created faster.

## Files

| File | Author model | Key contribution |
|:--|:--|:--|
| [`ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md`](ChatGPT-5-5-Thinking-Extended-DocTypes-2026-06-26.md) | ChatGPT 5.5 Thinking Extended | 13 trial steps mapped to document sets and gate types; the ACCELERATION list of the six greatest-acceleration targets; the hard / protocol-defined / decision gate distinction |
| [`Gemini-3-1-Pro-DocTypes-2026-06-26.md`](Gemini-3-1-Pro-DocTypes-2026-06-26.md) | Gemini 3.1 Pro | Phase-by-phase document map; the three timeline buckets and the conclusion that faster authoring compresses only the administrative/prep bucket |
| [`prompt-types.md`](prompt-types.md) | (prompt) | The exact research prompt used to generate the two sources |

## The six ACCELERATION targets (from the ChatGPT source)

1. The initial IND and IRB package.
2. Protocol amendments and synchronized consent/site updates.
3. Cohort-review packages immediately after required safety data mature.
4. A complete clinical-hold response.
5. The Phase 2-to-3 briefing package and Phase 3 protocol.
6. The pivotal CSR and NDA/BLA modules after database lock.

## The three gate types (from the ChatGPT source)

| Gate type | Meaning | Example documents |
|:--|:--|:--|
| Hard gate | The trial legally or ethically cannot proceed | Initial IND, IRB approval, clinical-hold response, material amendments |
| Protocol-defined gate | The trial's own rules block the next cohort, arm, or adaptation | Cohort-review packages, interim-analysis SAP/DSMB charter |
| Decision gate | Legally possible, but the sponsor will not invest without a decision | End-of-Phase 2 briefing, go/no-go after database lock |

## Where used in the paper

These sources feed the paper Methods (gate taxonomy, acceleration targets) and
Results/Discussion (which documents are on the critical path, the time buckets). See
[`../../draft-paper/sections/sec-03-methods.tex`](../../draft-paper/sections/sec-03-methods.tex)
and the figures `fig-04`, `fig-05`, `fig-11`, `fig-23`, `fig-24` in
[`../../mermaid`](../../mermaid).

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice. Sources
attributed to ChatGPT 5.5 Thinking Extended and Gemini 3.1 Pro.
