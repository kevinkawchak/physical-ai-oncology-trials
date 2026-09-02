# Figure 11 - The Data Room as Typed Records

**Platform.** D2. **Native construct.** SQL table records: a header band over
typed rows, with a class column.

## Perspective no other figure in this day gives

Figure 10 shows the day as a process and Figure 12 shows how it could fail. This
one shows the largest artifact the day produces, and it shows it in the form a
reviewer will actually meet it in: nine records, each with a question, an access
class, and a first line that states what is not true.

A D2 SQL table draws a header band over typed rows, which is exactly a record
listing. A flowchart of a data room would imply an order in which the folders are
read, and there is none.

## Native source

```d2
dataroom: {
  shape: sql_table
  01_entity: "Who owns this, since when" {constraint: open on request}
  02_program: "What is being built" {constraint: already public}
  03_evidence: "What is known, with limits" {constraint: already public}
  04_regulatory: "Prepared, and not submitted" {constraint: open on request}
  05_capital: "The gap, and no offering" {constraint: under a CDA}
  06_correspondence: "Who was approached, and when" {constraint: under a CDA}
  07_recognition: "What correspondence exists" {constraint: open on request}
  08_team: "One person, eleven roles" {constraint: open on request}
  09_risks: "What would make this stop" {constraint: open on request}
}
```

## TikZ construction

A ten-row, three-column record table. Row pitch is 0.66 cm and the header band is
0.70 cm deep. The class column is narrowest because its values are short and
repeated; the question column is widest because it carries the sentences.

| Element | Style | Geometry |
|:--|:--|:--|
| Header band, three cells | `d2cellh` | Widths 34 mm, 52 mm, 32 mm at `y = 0` |
| Record rows 1 to 9 | `d2celll` on columns 1 and 2, `d2cell` on 3 | `y = -0.66` to `y = -5.94` |
| Public rows, 2 and 3 | `d2cellk` on column 3 | Marked by fill, because "already public" is a different class from "open on request" |
| Confidential rows, 5 and 6 | `d2cellg` on column 3 | Marked by fill |
| Frame | `d2cont` fitted, braced fit value | Encloses the header and all nine rows |
| Title | `d2title` | Above the frame |
| First-line note | `pnote`, two lines | Below the frame |

Edge routing: there are no edges. A record listing has no connectors, and drawing
one between folders would assert a reading order the data room does not have.

## The three access classes, and why there are exactly three

| Class | Rows | What it means |
|:--|:--|:--|
| Already public | 2, 3 | Deposited with a digital object identifier. Nothing to send; a link suffices |
| Open on request | 1, 4, 7, 8, 9 | Sent to a counterparty who asks, with no agreement required |
| Under a confidentiality agreement | 5, 6 | Sent only under an executed agreement, for the counterparty's protection as much as the company's: row 6 names people who did not consent to publication |

## Value provenance

| Value in the figure | Source |
|:--|:--|
| The nine folders, their questions and their classes | `../briefs/brief-01-data-room-index.md`, the nine-folder table |
| The "prepared, and not submitted" wording on row 4 | The same file, the folder 04 first line |
| The "no offering exists" wording on row 5 | The same file, the folder 05 first line |
| The "no agreement exists" wording on row 6 | The same file, the folder 06 first line |

## Caption, exactly as printed

```
Figure 11. The data room as nine typed records, each carrying the question it
answers and the access class that governs who may be sent it and when.
```

Line 1 is 75 characters, line 2 is 71 characters.

## Sources read

- `funding/auto-fund/07Sep26/briefs/brief-01-data-room-index.md`
- `funding/capitalization-plan/final-capital/sections/sec-10-build-method.tex`
- `funding/capitalization-plan/final-capital/capstyle.sty`, for the `d2*` styles
