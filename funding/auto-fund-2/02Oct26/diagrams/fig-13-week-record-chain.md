# Figure 13 - The Week as a Record Chain

**Platform.** Graphviz. **Native construct.** Record nodes in ranks, one rank per
weekday, with orthogonal edges.

## Perspective no other figure in this day gives

Table 21 lists the four signals. Table 23 lists the sixteen letters. Neither
shows how one day's letters reach another day. The chain does: each weekday is a
rank from signal to letters to pending answers, and the only edges that cross
ranks are the follow-ups, which is why Friday receives from Monday and Tuesday
and from nowhere else.

## Native source

```dot
digraph week {
  rankdir=TB; splines=ortho; node [fontname="Times", fontsize=9];
  subgraph cluster_mon { label="Monday";
    s1 [shape=ellipse, style=filled, fillcolor="#8C5A12", fontcolor=white, label="Two unsolicited applicants"];
    l1 [shape=record, label="{3 emails, 2 replies|Personnel, first hire, training}"];
    p1 [shape=ellipse, label="NIH, SBA, UC San Diego; each applicant"]; s1 -> l1 -> p1; }
  subgraph cluster_tue { label="Tuesday";
    s2 [shape=ellipse, style=filled, fillcolor="#8C5A12", fontcolor=white, label="The SEC office's update"];
    l2 [shape=record, label="{5 emails|Venue; 3 notices ask nothing}"];
    p2 [shape=ellipse, label="The SEC office and the FDA, on venue"]; s2 -> l2 -> p2; }
  subgraph cluster_wed { label="Wednesday";
    s3 [shape=ellipse, style=filled, fillcolor="#8C5A12", fontcolor=white, label="White House communications"];
    l3 [shape=record, label="{4 emails, 1 form|A contributor's place}"];
    p3 [shape=ellipse, label="OSTP, Genesis, Moores; due Monday"]; s3 -> l3 -> p3; }
  subgraph cluster_thu { label="Thursday";
    s4 [shape=ellipse, style=filled, fillcolor="#8C5A12", fontcolor=white, label="An NSF pilot inquiry"];
    l4 [shape=record, label="{4 emails, 2 filings|Gate, resources, disclosure}"];
    p4 [shape=ellipse, label="NSF, NCTI, NAIRR; due Tuesday"]; s4 -> l4 -> p4; }
  subgraph cluster_fri { label="Friday";
    s5 [shape=ellipse, label="No new signal"];
    l5 [shape=record, label="{3 follow-ups|Only what has earned one}"];
    p5 [shape=ellipse, label="Read first on Monday at 5:00"]; s5 -> l5 -> p5; }
  l4 -> l5 [color="#8C5A12", penwidth=1.5, label="joins"];
  p1 -> l5 [color="#8C5A12", penwidth=1.5, label="follow-up, on the third open business day"];
  p2 -> l5 [color="#8C5A12", penwidth=1.5];
}
```

## TikZ construction

| Element | Style | Geometry |
|:--|:--|:--|
| Rank headers | `gvcellh`, 22 mm | y = 0.05; x = 0.00, 3.20, 6.40, 9.60, 12.80 |
| Signals | `gvkey`, 19 mm; Friday's as `gvgray` | y = -0.95 |
| Letters | Two-field records: `gvcells` over `gvcell`, 25 mm; Friday's header field as `gvcellh` | y = -2.05 and -2.57 |
| Pending answers | `gvgray`, 19 mm; Friday's as `gvsoft` | y = -3.80 |
| Follow-up edges | `gvundirb` into a channel at y = -4.85, rising at x = 11.20 into Friday's record | One label on the channel |
| The join | `gvedgeb` from Thursday's record to Friday's | Label above the edge |

## Caption, exactly as set

> Figure 13. The week as a record chain, one rank per weekday, with the follow-up edges
> that let Monday's and Tuesday's letters, and no others, reach Friday's three follow-ups.

The two lines are 85 and 88 characters.

## Where it is used

`../packet/sections/sec-01-the-week-in-one-record.tex`, after Table 21.
