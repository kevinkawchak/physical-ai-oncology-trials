# Figure 18 - One person's toolchain against an institutional program office

**Type.** diagrams (python)-type, clustered by layer. **Section.** §9, Build
Method. **Perspective.** *Function-for-function comparison at the layer level.*
Application 03's Figure 2 compares four functions in two flat rows; this
compares five layers with the cost and elapsed time attached, which is the
version a reviewer can price.

**Caption (three balanced lines, 64 to 68 characters each).**

```
Five layers, executed twice. The left column is one operator with an
LLM toolchain and a public repository; the right is the institutional
equivalent. Only the bottom layer is common to both.
```

## diagrams (Python) declaration

```python
# Specification only. Not executed, not committed as a .py file.
with Diagram("Two ways to run the same five layers", direction="TB"):
    with Cluster("One independent scientist, 14 months"):
        l1 = Node("Literature: 40 meta-analyses, LLM deep research")
        l2 = Node("Modeling: 3 simulations, $36,330 each")
        l3 = Node("Regulatory: protocol, IND, guidance")
        l4 = Node("Release: Zenodo DOI, GitHub repository")
        l5 = Node("Clinical: contracted to a qualified site")
    with Cluster("Institutional program office"):
        r1 = Node("Scientific writing group")
        r2 = Node("Modeling and simulation team, $120K to $2M per run")
        r3 = Node("Regulatory affairs unit")
        r4 = Node("Data management office and library")
        r5 = Node("Clinical: in-house trial unit")
    l1 >> l2 >> l3 >> l4 >> l5
    r1 >> r2 >> r3 >> r4 >> r5
    l5 - r5   # the only shared layer
```

## TikZ construction notes

| Element | Style token | Placement |
|:--|:--|:--|
| Left cluster | `dgcluster` | x = 0, five tiles stacked at y = 0 to -8.4, pitch 2.1 |
| Right cluster | `dgcluster2` | x = 6.8, same pitch, so layers align horizontally and can be compared by eye |
| Shared bottom layer | `dgtilem` on both sides | The only pair in the same fill, joined by a `dgbi` bidirectional edge |
| Cost annotations | `\pnote` | To the right of each right-hand tile, so the two columns are not crowded |
| Glyphs | `\glyphdoc`, `\glyphchart`, `\glyphshield`, `\glyphcloud`, `\glyphflask` on the left; `\glyphteam`, `\glyphcpu`, `\glyphgear`, `\glyphdb`, `\glyphuser` on the right | One per tile |

Horizontal alignment carries the comparison. A reader scanning a row sees the
same function twice; a reader scanning a column sees one organization's whole
stack.

## Repository sources

- `funding/pdac-funding-applications/applications/app-03-nsf-tip-x-labs/sections/sec-03-evidence.tex` - the four-function comparison this figure extends to five layers
- `funding/supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip` - the $36,330 figure and the $120,000 to $2,000,000 industry benchmark
- `funding/supplementary/Physical AI Oncology Trial Founding Documents.md` - the release layer
- `funding/science-golden-age/chunk-03-...md` - the indirect-cost and administrative-burden findings the right column carries
