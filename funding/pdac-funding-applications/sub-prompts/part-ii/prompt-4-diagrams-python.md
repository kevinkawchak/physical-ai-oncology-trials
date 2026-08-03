## prompt-4-diagrams-python

**Stage.** PART II, Stage 4 of 8. **Output.** `funding/pdac-funding-applications/diagrams-python/`.

### Objective

Specify every **diagrams (python)-type** figure. This vocabulary renders an icon
glyph with its label beneath and groups nodes into dashed titled clusters. It is
the right choice when the subject is a **system deployed across boundaries**:
what runs where, on whose hardware, behind which trust boundary.

**No Python file is generated.** The specification is machine-readable and the
figure is reproduced natively in TikZ, because Rule 3 forbids raster output and
the repository's lint job must stay green.

### Allocation

Three figures.

| File | Construct | Perspective (must be unique) |
|:--|:--|:--|
| `fig-09-on-premises-topology.md` | clustered infrastructure | The on-premises deployment: what sits inside the hospital boundary, what never crosses it |
| `fig-14-tripartisan-model-roles.md` | clustered by vendor | The three frontier-model roles and the artifact each produces |
| `fig-18-independent-scientist-stack.md` | clustered by layer | One person's toolchain against the equivalent institutional program office |

### Rules

1. Same palette rule. No black fill; glyph tiles use `pablue2`, `pablue1`,
   `protoblue`, `pagrayl`, or `pagraym`.
2. Every tile carries a vector pictogram drawn in TikZ. No raster, no font
   icons.
3. Cluster titles are set above the cluster, never inside the node field.
4. Each file states the figure number, the balanced caption, the
   `diagrams`-style declaration, the TikZ `dg*` tokens, and the repository
   sources.

### Commits

One commit per figure file, then one for the directory README.
