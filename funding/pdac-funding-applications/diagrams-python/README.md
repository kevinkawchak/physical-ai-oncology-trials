# diagrams-python - Stage 4 of the PART II schedule (3 figure specifications)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-4%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-4-diagrams-python.md)
[![Figures](https://img.shields.io/badge/Figures-3%20of%2020-3C7DB2.svg)](.)
[![Python files](https://img.shields.io/badge/Python%20files-none%20generated-9AA1A8.svg)](.)
[![Raster](https://img.shields.io/badge/Raster%20output-none-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)

Three **diagrams (python)-type** figure specifications. This vocabulary renders
an icon glyph with its label beneath and groups nodes into dashed titled
clusters, which makes it the right choice when the subject is a **system
deployed across boundaries**: what runs where, on whose hardware, behind which
trust boundary.

## No Python is generated

The specifications below are machine-readable `diagrams` declarations, and the
figures are reproduced **natively in TikZ**. No `.py` file is committed and no
raster is produced, for two reasons: the master prompt forbids PNG and JPG
output, and the repository's `lint-and-format` job runs `ruff check` and
`ruff format --check` across the tree on Python 3.10, 3.11, and 3.12. Adding
speculative Python here would put three required checks at risk for no gain.

## The three figures

| Fig | File | Construct | Perspective |
|:--|:--|:--|:--|
| 9 | [`fig-09-on-premises-topology.md`](fig-09-on-premises-topology.md) | clustered infrastructure | Nine components, one trust boundary, one direction of travel |
| 14 | [`fig-14-tripartisan-model-roles.md`](fig-14-tripartisan-model-roles.md) | clustered by vendor | Three vendors, nine roles, and two return edges |
| 18 | [`fig-18-independent-scientist-stack.md`](fig-18-independent-scientist-stack.md) | clustered by layer | Five layers executed twice, with only the bottom one shared |

## Why diagrams-python for exactly these three

| Question | Why the notation is required |
|:--|:--|
| What runs where, and what may cross | Glyph tiles inside dashed clusters read as deployed infrastructure. Boxes and arrows read as a process, which is the wrong claim |
| Who produced which artifact | Clustering by vendor puts the division of labour in the layout rather than in the labels |
| How do two organizations compare, function for function | Two aligned clusters let a reader scan a row for the comparison and a column for the whole stack |

## The three rules every file here follows

1. Every tile carries a **vector pictogram** drawn in TikZ. No raster, no font
   icons, no external asset.
2. Cluster titles are set **above** the cluster, never inside the node field.
3. A path that does not exist is drawn as a **struck link**, not omitted. Figure
   9's model-to-controller path is the case that matters.

## Files used from other directories (Rule 5)

| Source | Figures that read it |
|:--|:--|
| [`../applications/app-04-doe-genesis-mission/sections/sec-02-mechanism-fit.tex`](../applications/app-04-doe-genesis-mission/sections/sec-02-mechanism-fit.tex) | 9 |
| [`../applications/app-01-nih-pioneer-award/sections/sec-04-operation-governance.tex`](../applications/app-01-nih-pioneer-award/sections/sec-04-operation-governance.tex) | 9 |
| [`../applications/app-05-nih-sbir-seed/sections/sec-04-operation-governance.tex`](../applications/app-05-nih-sbir-seed/sections/sec-04-operation-governance.tex) | 9 |
| [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | 14, all nine roles verbatim |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | 14 |
| [`../prompts/prompt-apply.md`](../prompts/prompt-apply.md) | 14 |
| [`../applications/app-03-nsf-tip-x-labs/sections/sec-03-evidence.tex`](../applications/app-03-nsf-tip-x-labs/sections/sec-03-evidence.tex) | 18 |
| [`../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../supplementary/source-files) | 18 |
| [`../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../supplementary) | 18 |
| [`../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | 18 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
