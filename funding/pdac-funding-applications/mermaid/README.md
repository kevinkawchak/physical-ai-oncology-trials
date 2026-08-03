# mermaid - Stage 1 of the PART II schedule (6 figure specifications)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-1%20of%208-00417A.svg)](../sub-prompts/part-ii/prompt-1-mermaid.md)
[![Figures](https://img.shields.io/badge/Figures-6%20of%2020-3C7DB2.svg)](.)
[![Palette](https://img.shields.io/badge/Palette-patient--robot--advocacy-6C757D.svg)](../../supplementary/source-files)
[![Black fill](https://img.shields.io/badge/Black%20fill-none-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)

Six **mermaid-type** figure specifications for the summary paper. Mermaid gets
the largest share of the twenty-figure budget because the paper's spine is
chronological: a policy document, ten applications, a trial, and a build
pipeline are all sequences.

Each file carries the figure number, the balanced three-line caption, valid
Mermaid source, the TikZ construction notes the later stages compile, and the
repository files the figure draws on.

## The six figures

| Fig | File | Construct | Perspective |
|:--|:--|:--|:--|
| 1 | [`fig-01-golden-age-to-application.md`](fig-01-golden-age-to-application.md) | flowchart | One policy sentence becomes ten addressed applications |
| 2 | [`fig-02-independent-scientist-loop.md`](fig-02-independent-scientist-loop.md) | state diagram | The three states at which the enterprise stalls, and what removes each |
| 4 | [`fig-04-daraxonrasib-chronology.md`](fig-04-daraxonrasib-chronology.md) | gantt | Fourteen months of author work, and the two overlaps |
| 8 | [`fig-08-review-decision-gates.md`](fig-08-review-decision-gates.md) | flowchart with decisions | The reviewer's own five gates, and which section answers each |
| 12 | [`fig-12-perioperative-sequence.md`](fig-12-perioperative-sequence.md) | sequence diagram | One operative day, message by message |
| 17 | [`fig-17-submission-schedule.md`](fig-17-submission-schedule.md) | gantt | Ten review clocks against one binding site date |

Figure numbers are not contiguous because they are assigned by position in the
paper, not by stage. The gaps are filled by the other four stages.

## Why mermaid for these six and not the others

| Question the figure answers | Why mermaid |
|:--|:--|
| What happens next | Flowchart is the only vocabulary here whose primitive is a step |
| What decides | A decision node states a branch without a guard grammar |
| How long, and what overlaps | A gantt puts duration and position on the same axis; no other vocabulary does |
| Who said what, in what order | A sequence diagram orders messages between parties |

Where the question is instead *who is permitted*, *what contains what*, *what
runs where*, or *what depends on what*, the figure belongs to one of the other
four stages.

## Two figures deliberately share a subject with a PART I figure

Figure 4 and application 07's Figure 2 both plot the same chronology. They are
not the same diagram: application 07 separates author work from external readout
on a single axis, because a person-based reviewer must not mistake a simulation
for a result; figure 4 plots duration and overlap, because the summary paper's
claim is about throughput. Neither is copied from the other, and neither is
copied from a prior author work.

## Files used from other directories (Rule 5)

| Source | Figures that read it |
|:--|:--|
| [`../../science-golden-age/chunk-01-front-matter-and-summary.md`](../../science-golden-age/chunk-01-front-matter-and-summary.md) | 1, 2 |
| [`../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md`](../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | 1, 2 |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | 4 |
| [`../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../supplementary/source-files) | 4 |
| [`../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../supplementary) | 4 |
| [`../../RFA-RM-27-001/`](../../RFA-RM-27-001), [`../../RFA-RM-27-001-v2/`](../../RFA-RM-27-001-v2) | 4, 8 |
| [`../applications/`](../applications) | 1, 8, 12, 17 |
| [`../../potential-partners/UC-San-Diego/`](../../potential-partners/UC-San-Diego) | 8, 17 |
| [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | 12 |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
