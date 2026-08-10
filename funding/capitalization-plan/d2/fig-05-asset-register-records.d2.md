# Figure 5 - The asset register as four typed records

**Type.** d2-type, sql_table records. **Section.** §2, The Entity and the Asset.
**Perspective.** *Everything the company holds, licenses, has contracted for, or
lacks, as four typed tables with an encumbrance field on every row.* No other
figure enumerates the assets; Figure 4 draws the same four classes as physical
zones, which shows the shape of the holding but not its terms.

**Caption (three balanced lines, 62 to 65 characters).**

```
The asset register as four typed records. Thirteen rows are owned
outright, three are licensed on public terms, seven are absent,
and the contracted table has no rows at all as of August 2026.
```

## D2 source

```d2
owned: "owned" {
  shape: sql_table
  style: {fill: "#FFFFFF"; stroke: "#00417A"}
  asset_id: "int, primary key"
  title: "text"
  instrument: "enum, doi or repository"
  first_deposit: "date"
  encumbrance: "enum, none"
}

licensed: "licensed" {
  shape: sql_table
  style: {fill: "#FFFFFF"; stroke: "#3C7DB2"}
  item: "text"
  licensor: "text"
  terms: "text, public"
  exclusive: "bool, false on every row"
  survives_stop: "bool"
}

contracted: "contracted" {
  shape: sql_table
  style: {fill: "#E9ECEF"; stroke: "#6C757D"}
  row_count: "int, zero"
}

absent: "absent" {
  shape: sql_table
  style: {fill: "#E9ECEF"; stroke: "#6C757D"}
  item: "text"
  blocked_milestone: "text, foreign key"
  signatory: "text"
  company_can_obtain_alone: "bool, false on every row"
}

milestones: "milestone" {
  shape: sql_table
  style: {fill: "#DCE8F1"; stroke: "#3C7DB2"}
  milestone_id: "text, primary key"
  months: "text"
  cost_usd: "int"
  artifact: "text"
}

absent.blocked_milestone -> milestones.milestone_id: "foreign key"
owned.asset_id -> milestones.artifact: "supplies evidence for"
contracted -> absent: "every contracted row would\nclose one absent row"
```

## The four tables, populated

### owned, 13 rows, encumbrance none

| Asset | Instrument | Date | DOI or path |
|:--|:--|:--|:--|
| Phase 1 protocol | Zenodo | 2026-06-21 | 10.5281/zenodo.20780121 |
| Phase 2 protocol | Zenodo | 2026-06-24 | 10.5281/zenodo.20807027 |
| IND package, drafted, not filed | Zenodo | 2026-07-01 | 10.5281/zenodo.21097442 |
| Phase 1 LLM document guidance | Zenodo | 2026-06-29 | 10.5281/zenodo.21018646 |
| PI LLM adoption guide | Zenodo | 2026-06-26 | 10.5281/zenodo.20843290 |
| QSP simulation stack and VVUQ suite | Zenodo | 2025-08 | 10.5281/zenodo.17001137 |
| Daraxonrasib identification meta-analysis | Zenodo | 2025-06 | 10.5281/zenodo.15735068 |
| Funding application v1.0 | Zenodo | 2026-07-07 | 10.5281/zenodo.21232965 |
| Funding application v2.0 | Zenodo | 2026-07-12 | 10.5281/zenodo.21317266 |
| Patient robot advocacy paper | Zenodo | 2026-07-31 | 10.5281/zenodo.21720120 |
| H. R. 9510 bill v5.0 and three companions | Zenodo | 2026-06 | 10.5281/zenodo.20619762 |
| Repository, physical-ai-oncology-trials | GitHub, MIT | 2026-08 | v4.5.0 |
| Ten funding application file sets | Repository | 2026-08-04 | funding/pdac-funding-applications |

### licensed, 3 rows, none exclusive

| Item | Licensor | Terms | Survives a stop |
|:--|:--|:--|:--|
| Base model weights | Model vendor | Vendor terms, non-exclusive | No |
| Robotic platform | Site and vendor | Site's asset, not the company's | No |
| Daraxonrasib | Revolution Medicines | No licence exists; investigational | No |

### contracted, 0 rows

No clinical trial agreement, no drug supply agreement, no letter of
authorization, no CDA, no subaward, no insurance binder specific to the trial.
The table is drawn with its header and no body row, because an empty table is a
finding and a missing table is an omission.

### absent, 7 rows, none obtainable by the company alone

| Item | Blocks | Signatory |
|:--|:--|:--|
| Site agreement, executed | M1 | UC San Diego |
| IRB approval | M2 | Site IRB |
| Drug supply agreement | M6 | Revolution Medicines |
| Letter of authorization, IND cross-reference | M5 | Revolution Medicines |
| Theatre and robotic platform time | M7 | UC San Diego |
| Trial-specific liability insurance | M6 | Carrier |
| A second full-time employee | M3 | Funded by the Phase I award |

## TikZ construction notes

Canvas 14.6 by 8.2 cm. Four `d2sql` records in two columns of two, plus one
`d2sql` milestone record set apart on the right.

| Element | Style token | Placement |
|:--|:--|:--|
| owned record | `d2sql`, `text width=34mm` | x = 0, y = 0, anchor north west |
| licensed record | `d2sql`, `text width=34mm` | x = 0, y = -4.30 |
| contracted record | `d2sql` with `pagrayl` body fill, `text width=34mm` | x = 4.85, y = -4.30 |
| absent record | `d2sql` with `pagrayl` body fill, `text width=38mm` | x = 4.85, y = 0 |
| milestone record | `d2sql` with `pablue2` body fill, `text width=34mm` | x = 10.30, y = -2.15 |
| Row counts | `d2cellk`, `minimum width=13mm` | Anchored north east on each record's header strip |
| Foreign key edge | `d2edgeb` with an open arrowhead | absent to milestone, straight, x = 8.95 to 10.30 |
| Evidence edge | `d2edge` | owned to milestone, `bend right=16` beneath the absent record |
| Would-close edge | `d2edged` | contracted to absent, vertical, the only vertical edge |
| Empty-table note | `umlnote`, `text width=32mm` | Anchored north, 3 mm beneath the contracted record |
| In-figure note | `pnote`, `text width=134mm` | x = 0, y = -7.60 |

Record discipline: every field line is `name : type`, never prose. Row counts
are carried in a separate `d2cellk` strip on the header rather than inside a
field, because a count is metadata about the table and not a column of it.

Edge routing: three edges only. The evidence edge is the one that could cross
the absent record, and it takes `bend right=16`, which carries it 8 mm below
that record's south edge. The contracted-to-absent edge is vertical and is the
only edge in the corridor between the two columns.

## Repository sources

- `funding/supplementary/Physical AI Oncology Trial Founding Documents.md` - the deposited works and their DOIs
- `funding/supplementary/source-files/Physical-AI-Oncology-Trial-Competition-Proposal.zip` - the January 13, 2026 baseline: on that date the owned table held two rows and the absent table held the same seven
- `trial-protocol/`, `trial-phase-2/`, `trial-ind/`, `trial-documents/` - the protocol, Phase 2, IND and guidance assets
- `funding/pdac-funding-applications/` - the ten application file sets, the thirteenth owned row
- `funding/potential-partners/UC-San-Diego/` - two of the seven absent rows
- `LICENSE` at the repository root - the MIT terms on the repository row
