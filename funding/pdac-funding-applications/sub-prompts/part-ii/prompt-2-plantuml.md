## prompt-2-plantuml

**Stage.** PART II, Stage 2 of 8. **Output.** `funding/pdac-funding-applications/plantuml/`.

### Objective

Specify every **plantuml-type** figure. PlantUML is the right vocabulary when
the subject is **formal**: actors and their permitted use cases, states with
explicit guards, or an activity with forks and joins. It is used where the paper
must be precise about who is allowed to do what.

### Allocation

Three figures. PlantUML gets a small share because the paper has only three
genuinely formal subjects.

| File | Construct | Perspective (must be unique) |
|:--|:--|:--|
| `fig-05-actor-authority.puml.md` | use case | Every actor in the trial and the exact set of actions each is authorized to take |
| `fig-13-advisory-state-guards.puml.md` | state with guards | The advisory system's states and the guard condition on every transition, including the two that only a human can fire |
| `fig-19-award-lifecycle-activity.puml.md` | activity with fork and join | What happens in parallel once any one of the ten applications is funded |

### Rules

1. Same palette rule. No black fill.
2. Guards are written on the transition, never in a floating note that could be
   read as attached to the wrong edge.
3. Stick figures for actors, ellipses for use cases, rounded rectangles for
   states: the reader should recognize the notation without a legend.
4. Each file states the figure number, the balanced three-line caption, the
   PlantUML source, the TikZ `uml*` tokens, and the repository sources.

### Commits

One commit per figure file, then one for the directory README.
