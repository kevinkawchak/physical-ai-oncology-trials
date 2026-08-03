# Figure 15 - The trial's data objects and the keys that join them

**Type.** d2-type, sql tables. **Section.** §6, Physical AI Governance.
**Perspective.** *The join graph.* Application 04's Figure 3 lists three record
types field by field; this is the whole schema, with the foreign keys that make
a released dataset re-analysable by someone who was not there.

**Caption (three balanced lines, 64 to 68 characters each).**

```
Six tables and the five keys that join them. Every released record can
be traced to one participant and one operative step, which is what
makes an independent re-analysis of the cohort possible at all.
```

## D2 source

```d2
participant: {
  shape: sql_table
  participant_id: uuid {constraint: primary_key}
  dose_level: int
  kras_variant: text
  resectability: text
}
operative_step: {
  shape: sql_table
  step_id: uuid {constraint: primary_key}
  participant_id: uuid {constraint: foreign_key}
  step_ordinal: int
  arm_ids: int[]
}
advisory_decision: {
  shape: sql_table
  decision_id: uuid {constraint: primary_key}
  step_id: uuid {constraint: foreign_key}
  accepted: bool
  latency_ms: float
}
stop_event: {
  shape: sql_table
  stop_id: uuid {constraint: primary_key}
  step_id: uuid {constraint: foreign_key}
  measured_ms: float
  scope: text
}
adverse_event: {
  shape: sql_table
  ae_id: uuid {constraint: primary_key}
  participant_id: uuid {constraint: foreign_key}
  attribution: enum
  ctcae_grade: int
}
specimen: {
  shape: sql_table
  specimen_id: uuid {constraint: primary_key}
  participant_id: uuid {constraint: foreign_key}
  pathway_pd: float
  response_grade: text
}
operative_step.participant_id -> participant.participant_id
advisory_decision.step_id -> operative_step.step_id
stop_event.step_id -> operative_step.step_id
adverse_event.participant_id -> participant.participant_id
specimen.participant_id -> participant.participant_id
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Six tables | `d2sql`, two-part split node | Participant centred at x = 6.4, y = 1.4; the five dependents on an arc below at y = -1.2 and y = -3.4 |
| Header part | `protoblue` fill, white text | The `rectangle split part fill` first element |
| Field lines | `\tiny\sffamily`, left aligned | Primary key in bold, foreign key in italic |
| Join edges | `d2edge` with a crow-foot terminator drawn as three 0.9mm strokes | Five edges; none passes under a table because the dependents fan out |

`operative_step` is placed directly beneath `participant` because it carries the
only two-hop path in the schema, and putting it anywhere else forces an edge
across another table.

## Repository sources

- `funding/pdac-funding-applications/applications/app-04-doe-genesis-mission/sections/sec-04-operation-governance.tex` - the three record types this schema completes
- `funding/pdac-funding-applications/applications/app-08-nci-ctep/sections/sec-05-budget-site.tex` - the schedule of assessments that fixes the specimen and adverse-event timing
- `funding/RFA-RM-27-001-v2/LaTeX Source Files.zip` - the data management and sharing module
