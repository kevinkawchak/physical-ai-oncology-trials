# d2 - Stage 3 of the PART II schedule (4 figure specifications)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-3%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-3-d2.md)
[![Figures](https://img.shields.io/badge/Figures-4%20of%2020-3C7DB2.svg)](.)
[![Palette](https://img.shields.io/badge/Palette-patient--robot--advocacy-6C757D.svg)](../../supplementary/source-files)
[![Black fill](https://img.shields.io/badge/Black%20fill-none-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)

Four **d2-type** figure specifications. D2 is used where the subject is
structure rather than motion: a true grid, a nesting, a layering, or a record
with typed fields and keys.

## The four figures

| Fig | File | Construct | Perspective |
|:--|:--|:--|:--|
| 3 | [`fig-03-ten-application-grid.d2.md`](fig-03-ten-application-grid.d2.md) | grid | Ten applications on four attributes; the science column is constant |
| 7 | [`fig-07-evidence-container-stack.d2.md`](fig-07-evidence-container-stack.d2.md) | nested containers | Four evidence tiers as containment, not as a ladder |
| 11 | [`fig-11-budget-layers.d2.md`](fig-11-budget-layers.d2.md) | layers | Cash beside contributed value, and the two layers on both sides |
| 15 | [`fig-15-data-record-schema.d2.md`](fig-15-data-record-schema.d2.md) | sql tables | Six tables and the five keys that make re-analysis possible |

## Why d2 for exactly these four

| Question | Why the notation is required |
|:--|:--|
| How do ten things compare on four attributes | Only a true grid lets a reader scan a column and a row with equal ease |
| What contains what | Nesting states containment. A ladder of boxes states order, which is a different and weaker claim |
| Where does the money sit | Layers state composition without implying a flow |
| What joins to what | A typed record with a key is the only honest way to promise a dataset is re-analysable |

## The three rules every file here follows

1. A grid is a **true** grid: equal cell heights, columns placed by anchor from
   the left neighbour so they cannot drift, and a Corporate Blue header row with
   white bold text.
2. Container titles sit **outside** the child field, at the top-left corner,
   1.2mm above the container edge. No title overlaps a node.
3. Nesting is drawn on the **background layer**, so no container edge crosses a
   leaf box.

## Relationship to the PART I figures

Figures 3, 7, and 11 each re-read a PART I figure from a different angle rather
than repeating it: application 01's flat evidence grid becomes containment,
application 02's 36-month budget layering becomes a five-year cash-and-contributed
split, and the ten application READMEs become one scored matrix. Figure 15 has no
PART I counterpart; it completes the three record types sketched in application
04.

## Files used from other directories (Rule 5)

| Source | Figures that read it |
|:--|:--|
| [`../applications/README.md`](../applications/README.md) and the ten `app-*/README.md` | 3 |
| [`../applications/app-01-nih-pioneer-award/sections/sec-03-evidence.tex`](../applications/app-01-nih-pioneer-award/sections/sec-03-evidence.tex) | 7 |
| [`../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../supplementary/source-files) | 7 |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | 7 |
| [`../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../RFA-RM-27-001-v2) | 11, 15 |
| [`../applications/app-02-arpa-h/sections/sec-05-budget-site.tex`](../applications/app-02-arpa-h/sections/sec-05-budget-site.tex) | 11 |
| [`../applications/app-06-fnih-amp/`](../applications/app-06-fnih-amp) | 11 |
| [`../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md`](../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | 11 |
| [`../applications/app-04-doe-genesis-mission/sections/sec-04-operation-governance.tex`](../applications/app-04-doe-genesis-mission/sections/sec-04-operation-governance.tex) | 15 |
| [`../applications/app-08-nci-ctep/sections/sec-05-budget-site.tex`](../applications/app-08-nci-ctep/sections/sec-05-budget-site.tex) | 15 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
