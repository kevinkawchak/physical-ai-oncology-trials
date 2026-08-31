# template-new-system - the paper template this work adapts

[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Template](https://img.shields.io/badge/Template-Groningen%20MSc%20thesis-A32A3C.svg)](https://creativecommons.org/licenses/by/4.0/)

## Contents

`template-new-system.zip` holds the LaTeX skeleton of the author's March 2026
paper, *National Platform for Physical AI Oncology Trials*, with its images and
its surplus sections removed so that only the structure remains: `main.tex`,
`page_styles.tex`, and the cover page, contents, executive summary,
introduction, conclusion and appendices section files.

## What this work took from it

| Template element | Where it appears in this paper |
|:--|:--|
| Centered cover block between two rules | `\trialmast` in [trialstyle.sty](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/trialstyle.sty), used in every stage's `main.tex` |
| Title, draft line, DOI, author with ORCID iD, notice, disclaimer, city, date | The cover page of all three stages |
| `\tableofcontents` with a right-aligned Page label | The contents block of all three stages |
| One `\input` per section from `main.tex` | The eleven-file section set |
| Back matter after the reference list | `sections/sec-10-references-backmatter.tex` |

## What this work did not take

The template's page geometry, its `fancyhdr` page styles, its `ieeetr`
bibliography style, and its black-only link configuration are replaced by
`trialstyle.sty`, which is adapted instead from
[funding/capitalization-plan/final-capital/publication](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/capitalization-plan/final-capital/publication).
The template has no diagram vocabulary; all five in this paper come from that
directory's `capstyle.sty`.

## Template attribution

Adapted from the University of Groningen MSc AI and CCS Master's Thesis Template
(Overleaf, CC BY 4.0). Original template by Manvi Agarwal, 2020.
