# plantuml - Stage 2 of the PART II schedule (3 figure specifications)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-2%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-2-plantuml.md)
[![Figures](https://img.shields.io/badge/Figures-3%20of%2020-3C7DB2.svg)](.)
[![Palette](https://img.shields.io/badge/Palette-patient--robot--advocacy-6C757D.svg)](../../supplementary/source-files)
[![Black fill](https://img.shields.io/badge/Black%20fill-none-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)

Three **plantuml-type** figure specifications. PlantUML takes the smallest share
of the twenty-figure budget because only three subjects in the paper are formal
enough to need it: who is permitted to act, under what guard a state changes,
and what runs concurrently.

## The three figures

| Fig | File | Construct | Perspective |
|:--|:--|:--|:--|
| 5 | [`fig-05-actor-authority.puml.md`](fig-05-actor-authority.puml.md) | use case | Seven actors, eleven actions, four structurally denied paths |
| 13 | [`fig-13-advisory-state-guards.puml.md`](fig-13-advisory-state-guards.puml.md) | state with guards | Six advisory states; two transitions only a human can fire |
| 19 | [`fig-19-award-lifecycle-activity.puml.md`](fig-19-award-lifecycle-activity.puml.md) | activity with fork and join | Four workstreams in parallel; the join is the site agreement |

## Why plantuml for exactly these three

| Question | Why the notation is required |
|:--|:--|
| Who is permitted to do what | An association is a claim about permission. Drawing the absence of one, as a struck link, is a claim a flowchart cannot make |
| Under what condition does the state change | A guard written on a transition is checkable. A label on an arrow is not |
| What runs at the same time | Fork and join bars state concurrency. Parallel arrows only suggest it |

Everything else in the paper is a sequence, a container, a topology, or a
dependency, and belongs to one of the other four stages.

## The rule every file here follows

Guards are written **on the transition**, never in a floating note that could be
read as attached to a neighbouring edge. Where two guarded transitions leave the
same state, the bends are equal and opposite and the looseness is stated, so
neither curve can re-enter the box it left.

## Files used from other directories (Rule 5)

| Source | Figures that read it |
|:--|:--|
| [`../applications/app-02-arpa-h/sections/sec-04-operation-governance.tex`](../applications/app-02-arpa-h/sections/sec-04-operation-governance.tex) | 5, the three-actor slice this figure completes |
| [`../applications/app-10-ucsd-moores-engine/sections/sec-04-operation-governance.tex`](../applications/app-10-ucsd-moores-engine/sections/sec-04-operation-governance.tex) | 5 |
| [`../applications/app-08-nci-ctep/sections/sec-04-operation-governance.tex`](../applications/app-08-nci-ctep/sections/sec-04-operation-governance.tex) | 5, 13 |
| [`../applications/app-04-doe-genesis-mission/sections/sec-03-evidence.tex`](../applications/app-04-doe-genesis-mission/sections/sec-03-evidence.tex) | 13 |
| [`../applications/app-01-nih-pioneer-award/sections/`](../applications/app-01-nih-pioneer-award/sections) | 13, 19 |
| [`../applications/app-05-nih-sbir-seed/sections/sec-04-operation-governance.tex`](../applications/app-05-nih-sbir-seed/sections/sec-04-operation-governance.tex) | 19 |
| [`../../potential-partners/UC-San-Diego/priority-steps.md`](../../potential-partners/UC-San-Diego/priority-steps.md) | 19 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
