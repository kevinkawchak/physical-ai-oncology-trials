# diagrams (python)-type figure specifications

[![Platform](https://img.shields.io/badge/Platform-diagrams%20(python)-A32A3C.svg)](https://diagrams.mingrammer.com)
[![Figures](https://img.shields.io/badge/Figures-4%20of%2025-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/diagrams-python)
[![Stage](https://img.shields.io/badge/Produced%20by-stage--4--diagrams--python-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/sub-prompts/stage-4-diagrams-python)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)

Four of the paper's twenty-five figures use the `mingrammer/diagrams` idiom,
because four of its claims are about a system's parts, grouped, where each part
is best named by what kind of thing it is. The idiom renders an icon glyph with
the label beneath it inside a dashed titled cluster, which carries kind and
grouping in one mark.

The Python in each specification is real `diagrams` code and would run, but it
is never executed. Raster output is forbidden by the master prompt, so the paper
redraws each figure in TikZ with the `dg*` vocabulary, where every glyph is a
vector pictogram defined in the paper's style file. The Python source is kept
because it states the cluster membership and the edge set unambiguously, which
is what a later stage needs.

## The four figures

| File | Fig | § | Perspective |
|:--|:--|:--|:--|
| [fig-06-ind-assembly-clusters.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-06-ind-assembly-clusters.md) | 6 | 3 | Four source clusters feeding twelve IND modules, and the three composed rather than copied |
| [fig-13-on-premises-site-stack.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-13-on-premises-site-stack.md) | 13 | 4 | The four-layer site stack and the one boundary no inference request crosses |
| [fig-20-funding-production-pipeline.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-20-funding-production-pipeline.md) | 20 | 6 | One evidence store, one pipeline, four typed mechanism exits |
| [fig-25-single-vendor-and-watermark.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/diagrams-python/fig-25-single-vendor-and-watermark.md) | 25 | 8 | Two structural exposures, six mitigations, one residual |

## Glyph inventory used across the four figures

Every pictogram is a vector construction in the paper's style file; none is an
image file. The four figures draw on nineteen of them: `\glyphserver`,
`\glyphdb`, `\glyphcpu`, `\glyphrobot`, `\glyphshield`, `\glyphdoc`,
`\glyphmon`, `\glyphlock`, `\glyphgear`, `\glyphai`, `\glyphuser`,
`\glyphflask`, `\glyphpill`, `\glyphchart`, `\glyphstop`, `\glyphbank`,
`\glyphscalpel`, `\glyphcloud`, `\glyphsignal`, `\glyphlink`, `\glyphhand`,
and `\glyphteam`.

## Files from other directories used here

| Source directory or archive | Used by | For what |
|:--|:--|:--|
| [trial-ind/final-ind/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-ind/final-ind/publication) | Figures 6 and 13 | The twelve modules, their cited sources, and the filed device description |
| [trial-protocol/final-protocol/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-protocol/final-protocol/publication) | Figures 6 and 13 | The on-premises requirement, the bus rate, the stop budget, and the force caps |
| [trial-phase-2/final-protocol/publication/author](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) | Figures 6 and 13 | The forward path and the multicenter site qualification |
| [national-platform](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/national-platform) | Figures 6 and 13 | Adapted 21 CFR text and the site establishment package |
| [unification](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/unification) and [digital-twins](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/digital-twins) | Figure 6 | The simulation and VVUQ sources cited in the IND |
| [funding/pdac-funding-applications/final-apply/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/pdac-funding-applications/final-apply/publication) | Figure 20 | The ten applications and the mechanism class of each |
| [funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication) | Figures 20 and 25 | The small-business exit, the thirteen deposited assets, the risks and limits structure, and the `dg*` vocabulary |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | Figures 20 and 25 | The NIH exit and the model-disclosure language |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | Figure 25 | The AI peer review study's own recorded limitations |
| [new-trial-system/prompts/prompt-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/prompt-new-trial.md) | Figure 25 | The single-model instruction that creates the first exposure |

## Palette

Burgundy `#800020` for the subject tile with a white pictogram, lighter burgundy
1 `#A32A3C` for mid tiles, lighter burgundy 2 `#E2D6D9` for pale tiles, Mist
Gray `#C9C9C9` and its tints for neutral tiles, Slate Gray `#6B6B6B` for cluster
strokes and neutral edges, Charcoal `#2E2E2E` for text and rules, white for
ground. No tile in this directory carries a black or near-black fill.
