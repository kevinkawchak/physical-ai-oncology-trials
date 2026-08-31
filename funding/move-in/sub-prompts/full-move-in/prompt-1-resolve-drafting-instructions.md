# Stage 2, sub-prompt 1 - resolve every drafting instruction

## Goal

Copy `draft-move-in/` to `full-move-in/` and then remove every `\draftnote` by
answering it. A `\draftnote` that survives into stage 2 is a defect. The audit
is a grep for `draftnote`, which must return zero across `full-move-in/`.

## Method

For each instruction, in section order:

1. Open the exact path the instruction names.
2. Take the number, the wording, or the structure the instruction asks for.
   Do not re-derive a figure that already exists in an author source. The
   $700,000 per year budget frame is reused verbatim, not recomputed.
3. Write the prose that carries it, with the citation key in the same sentence
   as the number.
4. Delete the instruction.

## Rules that bind the prose

| Rule | Consequence |
|:--|:--|
| Clause K | American English, La Jolla register. No British spelling survives |
| Clause D | Author qualifications appear as facts with dates, not as adjectives |
| Clause H | The award is stated as $700,000 per year for five years, $3,500,000 total, from at least one federal agency |
| Master prompt, inputs | The $36,330 figure is described as **projected**, never as estimated |
| Positioning corrections | Daraxonrasib is not described as first in human; it is investigational and already in Phase 3 evaluation. The robotic configuration is specified at the site agreement. No drug supply agreement or letter of authorization exists |
| Rule 3 | No diagram. Where the parent build would have drawn a figure, write a table |

## What must be true of every number

Every quantitative claim carries a citation to an author source or to codified
law, and every claim taken from a simulation carries the limitation the source
itself stated, in the same table row or the same sentence. A number without its
caveat is a defect at this stage, not a stage 3 polish item.

## Acceptance

- `grep -rc 'draftnote' full-move-in/` returns 0.
- Every section that was a skeleton at stage 1 now carries finished prose.
- The stage compiles at 0 errors.

## Commit

One commit per section file, seventeen in all.
