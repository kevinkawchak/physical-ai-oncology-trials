# Figure 14 - The Weekly Cadence as Five Standing Functions

**Platform.** Diagrams, the `mingrammer/diagrams` idiom. **Native construct.**
Dashed titled clusters of glyph tiles, one cluster per weekday.

## Perspective no other figure in this day gives

Figure 13 shows one session and Figure 15 shows thirty days. This one shows the
repeating unit between them: the week, drawn as five standing functions rather
than as a schedule. A schedule is a list of dates; a standing function is a thing
that exists every week whether or not anything is due, and a company of one
person survives on the second rather than the first.

## Native source

```python
from diagrams import Diagram, Cluster
from diagrams.generic.blank import Blank

with Diagram("Weekly cadence", direction="LR"):
    with Cluster("Monday, federal, 90 min"):
        mon = Blank("Re-contacts, program questions, registrations")
    with Cluster("Tuesday, capital, 75 min"):
        tue = Blank("Instruments, counsel, treasury, filings")
    with Cluster("Wednesday, site and partner, 90 min"):
        wed = Blank("Institutions, foundations, start-up support")
    with Cluster("Thursday, preparation, 120 min"):
        thu = Blank("Data room, diligence answers, next week's drafts")
    with Cluster("Friday, execution and record, 75 min"):
        fri = Blank("Release, follow-ups, the week's record table")
    mon >> tue >> wed >> thu >> fri
```

## TikZ construction

Five clusters on a 2.85 cm horizontal pitch, each holding one glyph tile with its
label beneath and a time budget beneath that. `\dgnode` places the tile, its
pictogram and its label as three related nodes, so a braced `fit` over the tile
and its label encloses both.

| Element | Style and glyph | Geometry |
|:--|:--|:--|
| Monday tile | `dgtile` with `\glyphbank` | `(0,0)` |
| Tuesday tile | `dgtile` with `\glyphchart` | `(2.85,0)` |
| Wednesday tile | `dgtile` with `\glyphteam` | `(5.70,0)` |
| Thursday tile | `dgtilem` with `\glyphdoc` | `(8.55,0)` |
| Friday tile | `dgtile` with `\glyphclock` | `(11.40,0)` |
| Budget labels | `pnote`, centered under each label node | 1.05 cm below each tile |
| Cluster frames | `dgcluster`, `dgcluster2` on Thursday | Braced `fit` over each tile, its label and its budget |
| Cluster titles | `dgctitle` | Anchored north, above each frame |
| Sequence edges | `dgedgeb` | Four, left to right |
| Carry rule | `dgedged`, one curved edge from Friday back to the same weekday | Drawn below the row |
| Total | `pnote` | Below the row: seven and a half hours a week |

Edge routing: the four sequence edges run along the row between adjacent
clusters and touch nothing else. The single carry edge is drawn 1.4 cm below the
row and returns to Monday's south anchor, so it passes under three clusters
rather than through any of them.

## Why Thursday is drawn in the mid shade

Thursday is preparation, and it is the day that makes the following week
possible. Every other day spends what Thursday built. Marking it by fill rather
than by a note makes the asymmetry visible at a glance, which is the argument the
cadence brief makes in words.

## The carry rule the return edge represents

An artifact that does not exist by the end of its day is carried to the same day
next week rather than squeezed into the next one. A cadence that reschedules
within a week stops being a cadence by Wednesday, so the return edge goes to the
same weekday and not to the next day.

## Value provenance

| Value in the figure | Source |
|:--|:--|
| The five weekday themes | `../briefs/brief-02-weekly-cadence.md`, the frame table |
| The five time budgets | The same file |
| The seven and a half hour total | The same, summed |
| The carry rule | The same file, the rules section |

## Caption, exactly as printed

```
Figure 14. The weekly cadence as five standing functions with a time budget
each, and the carry rule that moves unfinished work a week rather than a day.
```

Line 1 is 73 characters, line 2 is 76 characters.

## Sources read

- `funding/auto-fund/08Sep26/briefs/brief-02-weekly-cadence.md`
- `funding/capitalization-plan/final-capital/sections/sec-07-operating-plan.tex`
- `funding/capitalization-plan/final-capital/capstyle.sty`, for the `dg*` styles and the vector glyphs
