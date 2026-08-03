# Figure 14 - Three frontier-model roles and the artifact each produces

**Type.** diagrams (python)-type, clustered by vendor. **Section.** §6, Physical
AI Governance. **Perspective.** *Division of labour across three vendors.* No
PART I figure covers this; it is the only figure in the paper about how the
documents were made rather than about what they say.

**Caption (three balanced lines, 62 to 66 characters each).**

```
Three vendors, nine roles, and the single review chain that connects
them. No model reviews its own output, and the chain is drawn as a
cycle because the second reviewer's finding returns to the author.
```

## diagrams (Python) declaration

```python
# Specification only. Not executed, not committed as a .py file.
with Diagram("Frontier model roles", direction="TB"):
    with Cluster("A. Claude Code, author"):
        ind = Node("IND applications and protocols")
        reg = Node("FDA, ICH, NIH adaptions")
        fig = Node("Machine-readable diagrams")
    with Cluster("B. ChatGPT Codex, first reviewer"):
        vvuq = Node("Verification, validation, uncertainty")
        meta = Node("Meta-analyses of trial papers")
        pdf = Node("PDF reading and writing")
    with Cluster("C. Google Gemini, second reviewer"):
        peer = Node("Second peer review")
        metaver = Node("Meta-verification of prior stages")
        code = Node("Accelerated code problem solving")
    ind >> vvuq
    reg >> vvuq
    fig >> vvuq
    vvuq >> peer
    meta >> peer
    peer >> ind  # findings return to the author
    metaver >> vvuq  # and to the first reviewer
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Three vendor clusters | `dgcluster` for A and B, `dgcluster2` for C | Three ranks, y = 0, -3.1, -6.2; three tiles each at x = 0, 2.6, 5.2 |
| Tiles | `dgtile` for A, `dgtilem` for B, `dgtileg` for C | Fill deepens by review stage, so the chain is readable without the labels |
| Forward edges | `dgedgeb` | A to B and B to C, drawn from the cluster edge, not from individual tiles, where all three tiles feed the same target |
| Return edges | `dgedged` with `bend left=22` | Two only: C back to A, and C back to B. The bend keeps them clear of the tile field |
| Glyphs | `\glyphdoc`, `\glyphshield`, `\glyphchart`, `\glyphgear`, `\glyphflask`, `\glyphai`, `\glyphteam`, `\glyphlink`, `\glyphcpu` | One per tile |

The two return edges are the figure's point: a linear chain would imply that
review findings are recorded rather than acted on.

## Repository sources

- `funding/tripartisan-llm-support.md` - all nine roles, verbatim
- `funding/daraxonrasib-llm-story.md` - the sequence in which the three were used, June 2025 to July 2026
- `funding/pdac-funding-applications/prompts/prompt-apply.md` - the constraint that only one model authored this build
