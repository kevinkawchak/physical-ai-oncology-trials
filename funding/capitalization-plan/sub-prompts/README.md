# Sub-Prompt Schedule - Capitalization Plan (v4.5.0)

[![Stages](https://img.shields.io/badge/Stages-8-00417A.svg)](.)
[![Diagram stages](https://img.shields.io/badge/Diagram%20stages-5-3C7DB2.svg)](.)
[![Paper stages](https://img.shields.io/badge/Paper%20stages-3-3C7DB2.svg)](.)
[![Figures](https://img.shields.io/badge/Figures-20-6C757D.svg)](.)
[![Parts](https://img.shields.io/badge/Parts-1-6C757D.svg)](.)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

The eight sub-prompts that build *From Independent Scientist to Novel
Performer: A Small-Business Operating, Milestone, and Capitalization Plan for a
Phase 1 LLM-Advised Robotic Whipple (ChemicalQDevice)*.

## Why one schedule and not two

`funding/pdac-funding-applications/sub-prompts/` runs two schedules, PART I for
ten application file sets and PART II for one summary paper, because that build
had two deliverables. This project has one: a single paper. The schedule is
therefore a single eight-stage line, five diagram stages followed by three paper
stages, and no stage is shared with another schedule.

## The eight stages

| Stage | Sub-prompt | Output directory | Figures | Commits |
|:--|:--|:--|:--|:--|
| 1 | [`stage-1-mermaid/`](stage-1-mermaid) | [`../mermaid/`](../mermaid) | 1, 7, 12, 13, 19 | 6 |
| 2 | [`stage-2-plantuml/`](stage-2-plantuml) | [`../plantuml/`](../plantuml) | 6, 10, 15 | 4 |
| 3 | [`stage-3-d2/`](stage-3-d2) | [`../d2/`](../d2) | 2, 5, 8, 11, 16 | 6 |
| 4 | [`stage-4-diagrams-python/`](stage-4-diagrams-python) | [`../diagrams-python/`](../diagrams-python) | 4, 18, 20 | 4 |
| 5 | [`stage-5-graphviz/`](stage-5-graphviz) | [`../graphviz/`](../graphviz) | 3, 9, 14, 17 | 5 |
| 6 | [`stage-6-draft-capital/`](stage-6-draft-capital) | [`../draft-capital/`](../draft-capital) | 20 sized slots | 10+ |
| 7 | [`stage-7-full-capital/`](stage-7-full-capital) | [`../full-capital/`](../full-capital) | 20 drawn | 10+ |
| 8 | [`stage-8-final-capital/`](stage-8-final-capital) | [`../final-capital/`](../final-capital) | 20 polished | 10+ |

## Figure allocation by purpose, not by quota

The master prompt forbids an equal split. Each vocabulary answers a question the
other four state badly, and the count follows from how often this paper asks
that question.

| Platform | Count | The question only this vocabulary states well | Where the paper asks it |
|:--|:--|:--|:--|
| Mermaid-type | 5 | What happens next, what decides, how long, who spoke in what order | Clause selection, the Phase I to Phase II gate, the financing sequence, the twelve-milestone calendar, the August traction chain |
| PlantUML-type | 3 | Who is permitted to act, under what guard, what runs concurrently | The sponsor and site boundary, the part 54 capital firewall, milestone evidence production |
| D2-type | 5 | What contains what, what tabulates against what, what joins to what | The mechanism-fit grid, the asset register, the two prices, the capital stack, the clinical evidence panel |
| Diagrams (python)-type | 3 | What runs where, across which trust boundary | The asset topology, the operating topology, the artifact custody topology |
| Graphviz-type | 4 | What depends on what, how a failure propagates | The indirect-cost decomposition, the work-package DAG, the stop-condition fault tree, the evidence chain |

Five, three, five, three, four. The two five-counts fall where the paper argues
in sequence (mermaid) and where it argues in tables (d2), which is what a
capitalization plan does most.

## Commit discipline

Every stage commits in real time as each file is produced. No stage batches its
work to the end.

- One commit per figure specification, plus one for the stage README.
- For each paper stage: one commit for `main.tex`, `capstyle.sty`,
  `references.bib` and the stage README, then one commit per section `.tex`.
- The second-to-last commit of each paper stage fixes every defect found in that
  stage's own files.
- The last commit of the build performs the repository updates: root README,
  `CHANGELOG.md`, `releases.md`.

## Rule 5 source map

| This directory uses | From | For |
|:--|:--|:--|
| `sub-prompts/part-i/README.md`, `part-ii/README.md` | `funding/pdac-funding-applications` | The schedule form, the stage table, the commit discipline |
| `final-apply/applystyle.sty` | `funding/pdac-funding-applications` | The five TikZ diagram vocabularies inherited by `capstyle.sty` |
| `final-apply/references.bib` | `funding/pdac-funding-applications` | The bibliography, extended with this paper's regulatory entries |
| `chunk-01`, `chunk-03`, `chunk-04`, `chunk-05`, `chunk-08` | `funding/science-golden-age` | The SBIR and novel-performer clauses the paper turns on |
| `Physical-AI-Oncology-Trial-Competition-Proposal.zip` | `funding/supplementary/source-files` | The January 13, 2026 baseline the asset register dates from |
| `LaTeX Source Files.zip` | `funding/RFA-RM-27-001-v2` | The cover theme this paper deliberately varies from |
