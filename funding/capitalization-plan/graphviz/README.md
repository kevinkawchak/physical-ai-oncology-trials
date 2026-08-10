# Graphviz-type figures - Capitalization Plan (v4.5.0)

[![Platform](https://img.shields.io/badge/Platform-Graphviz-3C7DB2.svg)](https://graphviz.org)
[![Figures](https://img.shields.io/badge/Figures-4-00417A.svg)](.)
[![Constructs](https://img.shields.io/badge/Constructs-records%20%2F%20DAG%20%2F%20fault%20tree-6C757D.svg)](.)
[![Stage](https://img.shields.io/badge/Stage-5%20of%208-6C757D.svg)](../sub-prompts/stage-5-graphviz)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

Four figure specifications produced by
[`../sub-prompts/stage-5-graphviz/`](../sub-prompts/stage-5-graphviz). Each is
reproduced in LaTeX by the `gv*` TikZ vocabulary in `capstyle.sty`. Graphviz is
used wherever the claim is about **dependency or propagation**: what must exist
before what, and how one failure reaches everything downstream.

## Contents

| File | Figure | § | Construct | The question it answers |
|:--|:--|:--|:--|:--|
| [`fig-03-indirect-cost-decomposition.gv.md`](fig-03-indirect-cost-decomposition.gv.md) | 3 | 1 | record nodes | What does the same direct work cost under three overhead regimes? |
| [`fig-09-work-package-dag.gv.md`](fig-09-work-package-dag.gv.md) | 9 | 3 | DAG, three clusters | Which work packages does the money reach? |
| [`fig-14-stop-condition-fault-tree.gv.md`](fig-14-stop-condition-fault-tree.gv.md) | 14 | 5 | fault tree | What has to fail, and in what combination? |
| [`fig-17-evidence-chain-records.gv.md`](fig-17-evidence-chain-records.gv.md) | 17 | 6 | record chain | Where does each published number land in a filing? |

## Why four

Graphviz's idiom is the edge, and this paper has four arguments whose whole
content is a set of edges. Two of the four are about money reaching or not
reaching something, one is about failure propagating, and one is about a
citation travelling into a filing. A fifth would have to restate one of these,
which is why the count stops at four rather than at five.

## Arithmetic that must hold

Two of these figures carry sums. A fault tree can be ugly and still be right; a
cost record cannot.

| Check | Where | Holds |
|:--|:--|:--|
| $842,000 + $288,000 + $96,000 + $86,000 + $84,000 = $1,396,000 | Figure 3, direct record | Yes |
| 57 percent of $1,300,000 MTDC = $741,000 | Figure 3, route A | Yes |
| $1,396,000 + $741,000 = $2,137,000 | Figure 3, route A total | Yes |
| 40 percent of $1,396,000 = $558,000; 7 percent of $1,954,000 = $137,000 | Figure 3, route B | Yes |
| $1,396,000 + $558,000 + $137,000 = $2,091,000 | Figure 3, route B total | Yes |
| 7.5 percent of $1,396,000 = $105,000; 7 percent of $1,501,000 = $105,000 | Figure 3, route C | Yes |
| $1,396,000 + $105,000 + $105,000 = $1,606,000 | Figure 3, route C total | Yes |
| $2,137,000 minus $1,606,000 = $531,000, that is 33.1 percent | Figure 3, premium | Yes |
| WP1 to WP5 sum to $306,000 | Figure 9, Phase I cluster | Yes |
| WP6 to WP12 sum to $1,300,000 | Figure 9, Phase II cluster | Yes |
| WP13 to WP17 sum to $2,104,000 | Figure 9, gap cluster | Yes |
| WP1 to WP12 carry the identical cost to M1 to M12 | Figures 9 and 13 | Yes, all twelve |

## Anti-defect record

| Defect class | How these four avoid it |
|:--|:--|
| Spaghetti edges | Figure 9 has exactly two edge crossings, both at a shallow angle in open canvas. Figure 17 has two convergent pairs, each drawn as one straight and one 18-degree bend so the arrivals are 4 mm apart |
| Default nodesep | Every figure states `ranksep` and `nodesep` numerically. Figure 17 uses 0.32 because a three-field record is tall, not wide |
| Missing `cluster_` prefix | Every subgraph is named `cluster_*`, so each is drawn as a box rather than silently dissolving |
| Record parser failure | Every record field is plain text. No literal brace, angle bracket or vertical bar appears inside a field |
| Rank skipping in a tree | Figure 14 is strictly layered across five ranks and no edge skips more than one rank |
| Black gate fills | `\umlgateand` takes `pagraym` and `\umlgateor` takes `pagrayl`. Neither is black, and each gate's label sits beneath its glyph at a fixed 0.22 cm offset |
| Arrow read as an equals sign | Figure 17 names the link type on all fifteen edges, and lists what each type does not license |

## Rule 5 source map

| These figures use | From | For |
|:--|:--|:--|
| `final-apply/sections/sec-08-budget-and-leverage.tex` | `../../pdac-funding-applications` | Figures 3 and 9, the direct-cost base and the four-layer frame |
| `final-apply/sections/sec-05-trial-evidence.tex` | `../../pdac-funding-applications` | Figure 17's five quantities |
| `final-apply/sections/sec-06-physical-ai-governance.tex` | `../../pdac-funding-applications` | Figure 17's 3 ms and 500 ms stop figures |
| `applications/app-05-nih-sbir-seed/` | `../../pdac-funding-applications` | Figures 3 and 9, the two award amounts |
| `chunk-03` | `../../science-golden-age` | Figure 3, the deference-to-incumbents finding it prices |
| `trial-ind/`, `trial-protocol/` | repository root | Figure 17's section numbering, Figure 14's de-escalation rule |
| 13 CFR 121.702, 21 CFR §54.2, 2 CFR 200.414 | codified | Figures 3 and 14 |
| `final-apply/applystyle.sty` | `../../pdac-funding-applications` | The `gv*` vocabulary and the two gate glyph macros |
