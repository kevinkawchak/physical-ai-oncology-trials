# Figure 6 - Which award unblocks which activity

**Type.** graphviz-type, directed acyclic graph. **Section.** §2, The Ten
Applications. **Perspective.** *Redundancy.* Application 01's Figure 3 shows one
award's dependencies; this shows all ten against the same activity set, and the
result is that only two activities have a single upstream source.

**Caption (three balanced lines, 63 to 67 characters each).**

```
Ten awards against eight trial activities. Six activities have two or
more upstream awards and survive any single refusal; the two that do
not are drawn heavier, and both are regulatory.
```

## DOT source

```dot
digraph funding {
  rankdir=LR; node [fontname="Times", fontsize=9]; edge [color="#6C757D"];
  subgraph cluster_awards {
    label="Ten applications"; style=dashed; color="#6C757D";
    a01 [label="01 Pioneer"]; a02 [label="02 ARPA-H"]; a03 [label="03 X-Labs"];
    a04 [label="04 Genesis"];  a05 [label="05 SBIR"];  a06 [label="06 FNIH"];
    a07 [label="07 HHMI"];     a08 [label="08 CTEP"];  a09 [label="09 FRO"];
    a10 [label="10 Moores"];
  }
  subgraph cluster_acts {
    label="Trial activities"; style=dashed; color="#6C757D";
    t1 [label="Site agreement"];        t2 [label="IND activation", penwidth=2];
    t3 [label="IRB approval", penwidth=2]; t4 [label="Interlock rig"];
    t5 [label="Logging schema"];        t6 [label="Cohort accrual"];
    t7 [label="Specimen pipeline"];     t8 [label="Public release"];
  }
  a01 -> t1; a01 -> t2; a01 -> t6;
  a02 -> t2; a02 -> t4; a02 -> t6;
  a03 -> t4; a03 -> t5; a03 -> t8;
  a04 -> t5; a04 -> t8;
  a05 -> t4; a05 -> t5;
  a06 -> t7; a06 -> t6;
  a07 -> t6; a07 -> t7;
  a08 -> t6; a08 -> t7; a08 -> t8;
  a09 -> t5; a09 -> t7; a09 -> t8;
  a10 -> t1; a10 -> t3;
}
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Award cluster | `gvcluster` dashed, corner title | Left, ten `gvnode` ellipses in two columns of five, x = 0 and 2.4, y pitch 1.15 |
| Activity cluster | `gvcluster2` dashed | Right, eight `gvbox` at x = 8.6, y pitch 1.35 |
| Single-source activities | `gvboxk` filled Corporate Blue, 0.9pt stroke | IND activation and IRB approval only |
| Edges | `gvedge` thin black | Twenty-three edges routed through a 2.2-wide corridor at x = 5.0 to 7.0 so none passes under an ellipse |
| Corridor | Unfilled | Left deliberately empty; the density of the corridor is part of the reading |

Two columns of awards keeps the left cluster the same height as the right one,
which stops the edge fan from crossing itself at the top and bottom.

## Repository sources

- Each `funding/pdac-funding-applications/applications/app-*/sections/sec-05-budget-site.tex` - the activity each ask funds
- `funding/pdac-funding-applications/applications/app-01-nih-pioneer-award/sections/sec-05-budget-site.tex` - the single-source finding this figure generalises
- `funding/potential-partners/UC-San-Diego/priority-steps.md` - the site agreement and IRB activities
