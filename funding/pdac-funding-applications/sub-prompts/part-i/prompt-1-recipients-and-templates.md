## prompt-1-recipients-and-templates

**Stage.** PART I, Stage 1 of 5. **Output.**
`funding/pdac-funding-applications/applications/README.md`, the shared
`appstyle.sty` template contract, and the shared `references.bib` seed.

### Objective

Fix the ten recipients, the per-recipient template variant, and the shared
LaTeX contract before any application is written, so that the ten file sets are
unique to their recipient without diverging in quality or in palette.

### Recipient selection rule

A recipient qualifies only if the White House report *Science: A New Golden Age*
(`funding/science-golden-age/`) either names its program or names the mechanism
the program runs. Record the anchor chunk and the anchor sentence for each.
Reject any recipient that cannot be tied to a chunk.

| # | Recipient program | Perspective | Anchor chunk |
|:--|:--|:--|:--|
| 01 | NIH Common Fund, Director's Pioneer Award | surgical | `chunk-01`, `chunk-03` |
| 02 | ARPA-H mission office | surgical | `chunk-03` |
| 03 | NSF TIP Directorate, X-Labs | surgical | `chunk-03` |
| 04 | DOE Office of Science, Genesis Mission | surgical | `chunk-06`, `chunk-08` |
| 05 | NIH SEED, SBIR/STTR | surgical | `chunk-05`, `chunk-08` |
| 06 | Foundation for the NIH, AMP | medical oncology | `chunk-04` |
| 07 | HHMI Investigator Program | medical oncology | `chunk-03` |
| 08 | NCI Cancer Therapy Evaluation Program | medical oncology | `chunk-06`, `chunk-08` |
| 09 | Convergent Research, FRO program | medical oncology | `chunk-03` |
| 10 | UC San Diego Moores Cancer Center | medical oncology | `chunk-05`, `chunk-08` |

### Template contract (`appstyle.sty`, one copy per application directory)

1. Palette is the `patient-robot-advocacy` palette exactly: `protoblue` #00417A,
   `protogray` #6C757D, white, `pagrayl` #E9ECEF, `pagraym` #CED4DA, `pagrayd`
   #9AA1A8, `pablue1` #3C7DB2, `pablue2` #DCE8F1. **No `padark` token and no
   black fill anywhere**; black is permitted for strokes and text only.
2. The five TikZ diagram vocabularies (`mm*`, `uml*`, `d2*`, `dg*`, `gv*`) are
   carried over, trimmed to the constructs a five-page application needs.
3. Figure spacing invariant: every figure is `\end{appfig}` then
   `\vspace{-0.7cm}` then `\figcaption{...}`, with rigid skips, so the
   frame-to-caption distance is identical for every figure in the repository.
4. Captions are centred, italic, at most three lines, with the lines balanced to
   a similar character count.
5. Every fixed-width table column is `>{\raggedright\arraybackslash}p{...}` and
   every table is exactly `\textwidth` wide.
6. `\RaggedRight` body with `\RaggedRightRightskip=0pt plus 2em`, maximal widow,
   club and broken penalties, and a stretchable `\parfillskip` so no paragraph
   ends in a one-word or two-word line.
7. `\UrlBreaks` on every character and `\Urlmuskip=0mu plus 3mu`, re-asserted
   after `url` and `hyperref` load, so no link runs off the right margin.
8. Single dashes only. No em dash, en-dash pair, or triple dash. `\S` for every
   codified section reference.

### Per-recipient cover variant

Each application gets a different cover treatment, so the ten attachments do not
read as ten copies of one file:

| # | Cover variant |
|:--|:--|
| 01 | Full-width banner with a person-based-award badge strip |
| 02 | Milestone-ledger cover: three-column go/no-go table above the title |
| 03 | Rule-block cover with a left accent bar and an organization-type panel |
| 04 | Mission-tile cover naming the Robotics national mission |
| 05 | Two-panel commercial cover: technical objective beside the market case |
| 06 | Partnership cover: a two-party consortium strip |
| 07 | Person-first cover: investigator block above a compact title rule |
| 08 | Protocol-record cover in the style of a study registration header |
| 09 | Time-bound cover: a five-year dissolution timeline strip |
| 10 | Institutional intake cover with a Moores Cancer Center routing block |

### Deliverable checklist

- [ ] `applications/README.md` with the recipient table, the anchor sentences,
      the per-recipient variant table, and the file-set contract.
- [ ] The `appstyle.sty` contract written down in that README so every later
      stage can be checked against it.
- [ ] A shared `references.bib` seed drawn from
      `funding/RFA-RM-27-001-v2/references.bib`,
      `funding/science-golden-age/chunk-09` and `chunk-10`, and
      `funding/supplementary/Physical AI Oncology Trial Founding Documents.md`.

### Commits

One commit for the applications README, one for the style contract, one for the
bib seed. Push each the moment it is written (Rule 8).
