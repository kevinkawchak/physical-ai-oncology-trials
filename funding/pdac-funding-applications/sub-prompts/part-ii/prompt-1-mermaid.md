## prompt-1-mermaid

**Stage.** PART II, Stage 1 of 8. **Output.** `funding/pdac-funding-applications/mermaid/`.

### Objective

Specify every **mermaid-type** figure the summary paper will carry, as a
machine-readable Mermaid source plus the TikZ construction notes the later
stages compile. Mermaid is the right vocabulary when the subject is a
**decision, a sequence, a state change, or a schedule** - anything whose
meaning is carried by order in time.

### Allocation

Six figures, out of a paper budget of twenty. Mermaid gets the largest share
because the paper's spine is a chronology: a policy document, ten applications,
a trial, and a build pipeline, each of which is a sequence.

| File | Construct | Perspective (must be unique) |
|:--|:--|:--|
| `fig-01-golden-age-to-application.md` | flowchart | How one paragraph of federal policy becomes ten addressed applications |
| `fig-02-independent-scientist-loop.md` | state diagram | The states an independent scientist's proposal passes through, and where the incumbency tax used to stop it |
| `fig-04-daraxonrasib-chronology.md` | timeline / gantt | June 2025 to August 2026, identification through RASolute 302 readout |
| `fig-08-review-decision-gates.md` | flowchart with decisions | The go / no-go gates a reviewer applies, and which section answers each |
| `fig-12-perioperative-sequence.md` | sequence diagram | Who says what to whom across the operative day, including the advisory boundary |
| `fig-17-submission-schedule.md` | gantt | The ten submissions, their review clocks, and the partner-site milestones |

### Rules

1. Palette is the `patient-robot-advocacy` palette. **No black fill.** Black is
   allowed for strokes and text only.
2. No figure may reproduce a diagram from a prior author work. Each is new.
3. Keep every figure legible at one column width and simple enough that the
   author can edit it: no more than about fourteen nodes, no crossing edge
   bundles, no nested subgraph deeper than two.
4. Each file states: the figure number, the caption (three lines maximum,
   balanced), the Mermaid source, the TikZ style tokens the paper will use, and
   the repository files the figure draws on.

### Commits

One commit per figure file, then one for the directory README. Push each
immediately.
