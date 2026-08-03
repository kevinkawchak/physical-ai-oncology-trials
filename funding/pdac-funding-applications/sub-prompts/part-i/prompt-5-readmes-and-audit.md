## prompt-5-readmes-and-audit

**Stage.** PART I, Stage 5 of 5. **Output.** Every README under
`funding/`, and the PART I error-fix pass.

### README coverage requirement

Every directory under `physical-ai-oncology-trials/funding/`, existing and new,
must carry a comprehensive `README.md` with badges, a repository structure
block, DOI badges where a DOI exists, and an explicit statement of which files
from other directories were used and where (Rule 5).

| Directory | State before this stage | Action |
|:--|:--|:--|
| `funding/` | short DOI list | Rewrite as a hub with structure, badges, and a source map |
| `funding/RFA-RM-27-001/` | present | Expand: what it is, what reuses it |
| `funding/RFA-RM-27-001-v2/` | present | Expand: name the files this build reads from the zip |
| `funding/pdfs/` | present | Expand: which application emails attach which PDF |
| `funding/potential-partners/` | present | Expand: the two candidate sites and their role |
| `funding/potential-partners/Scripps/` | present | Expand |
| `funding/potential-partners/UC-San-Diego/` | present | Expand: the partner of choice for all ten |
| `funding/science-golden-age/` | comprehensive | Add a section mapping each chunk to the applications that cite it |
| `funding/supplementary/` | present | Expand: the three source zips and the founding-document ledger |
| `funding/supplementary/source-files/` | present | Expand: what each zip supplies downstream |
| `funding/pdac-funding-applications/**` | new | Comprehensive README per directory |

### Audit pass (Rule 7, second-to-last commit)

Run these checks over all ten application file sets and fix every hit:

1. `grep -n "padark\|fill=black\|fill=protoblack"` across every `.sty` and
   `.tex` returns nothing.
2. Every `\end{appfig}` is followed by `\vspace{-0.7cm}` and then
   `\figcaption`, with no other token between them.
3. Every `p{...}` column specification is preceded by
   `>{\raggedright\arraybackslash}`.
4. Every `tabularx` or fixed table is set to `\textwidth` or `\linewidth`.
5. No `---`, no `--` outside a bibliography page range, no Unicode em dash or
   en dash in body text.
6. Every `\cite` key resolves in that application's `references.bib`.
7. Every caption is at most three lines and the lines are balanced.
8. `\S` is used for every codified section reference; no bare `SS`.

### Last commit of PART I

Update `applications/README.md` and the hub milestone table, then hand over to
PART II.
