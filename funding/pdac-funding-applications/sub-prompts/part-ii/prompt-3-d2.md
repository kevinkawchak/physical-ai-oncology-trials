## prompt-3-d2

**Stage.** PART II, Stage 3 of 8. **Output.** `funding/pdac-funding-applications/d2/`.

### Objective

Specify every **d2-type** figure. D2 is the right vocabulary when the subject is
**nesting or tabulation**: containers inside containers, a true grid, a record
with typed fields, or a layered stack. It is used where the paper must show
structure rather than motion.

### Allocation

Four figures.

| File | Construct | Perspective (must be unique) |
|:--|:--|:--|
| `fig-03-ten-application-grid.d2.md` | grid | The ten applications as a scored matrix: mechanism, term, ask, perspective |
| `fig-07-evidence-container-stack.d2.md` | nested containers | The four evidence tiers, from simulation to registrational readout, nested by strength of claim |
| `fig-11-budget-layers.d2.md` | layers | Where every dollar of the ask lands, layer by layer, with the cost-share layer separated |
| `fig-15-data-record-schema.d2.md` | sql tables | The trial's data objects and the keys that join them, for the data-sharing section |

### Rules

1. Same palette rule. No black fill.
2. A grid must be a true grid: equal cell heights, aligned columns, a header row
   in Corporate Blue with white text.
3. Container titles sit at the container's top-left corner, outside the child
   nodes, never overlapping them.
4. Each file states the figure number, the balanced caption, the D2 source, the
   TikZ `d2*` tokens, and the repository sources.

### Commits

One commit per figure file, then one for the directory README.
