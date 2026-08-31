# Stage 1, sub-prompt 2 - `main.tex`, cover page, and table of contents

## Goal

Write `funding/move-in/draft-move-in/main.tex`. It carries the cover page
exactly as the master prompt specifies it, the table of contents in the
template's form, and one `\input` per section file with `\clearpage` discipline
between documents.

## The cover, line by line

The master prompt fixes the content. The theme is adapted from
`inputs/Physical-AI-Oncology-Clinical-Trial-Site-Complete-Documentation-Package.zip`,
whose cover is a centered title block over three stacked italic disclaimer
blocks, then the contents. The La Jolla cover keeps that center axis and adds a
ruled frame, a badge line, and a fifteen-cell document strip.

| Line | Text |
|:--|:--|
| Title | La Jolla Move-In: Pancreatic Oncology Clinical Trial Site Complete Documentation Package |
| Subtitle | 15 Documents for California's First PDAC LLM Oncology Clinical Trial Site |
| Third line | Legislation, Regulations, Building Code, Premises Code, Conventional Trial Requirements, and Operations |
| Author | Kevin Kawchak, CEO ChemicalQDevice, `kevink@chemicalqdevice.com` |
| Date line | August 23, 2026, Draft 1.0 |
| Scope note | Independent research paper and practical adoption guide. Not medical or regulatory advice, not endorsed by the FDA, NIH, HHS, an IRB, ICH, or any sponsor |
| Disclaimer | Independent; not endorsed or sponsored by any trial sponsor, CRO, site, IRB, regulator, or medical society; adapted using Claude Code Opus 5 |
| ORCID | `0009-0007-5457-8667`, hyperlinked to `https://orcid.org/0009-0007-5457-8667` |
| Deposit line | Paper v1.0 at `https://doi.org/10.5281/zenodo.xxxxxxxx`; repository v4.7.0 at `https://github.com/kevinkawchak/physical-ai-oncology-trials`; files in `/funding/move-in` |
| Foot | August 23, 2026 |

The DOI stays in placeholder form `10.5281/zenodo.xxxxxxxx` with a live
hyperlink to `https://doi.org/10.5281/zenodo.xxxxxxxx` (Rule 12). Do not invent
a number.

## Table of contents

Keep the template's form: `\tableofcontents` immediately after the disclaimer
blocks, with each document appearing as a part-level line and its sections
nested beneath. Use the compressed `\l@section` from `movestyle.sty` so fifteen
documents plus front and back matter do not spill into a mostly empty second
contents page unnecessarily.

## Body

One `\input` per section, in order, with `\clearpage` before every `\docpart`
so each of the fifteen documents opens on its own page:

```
\input{sections/sec-00-front}\clearpage
\input{sections/sec-01-sb-1188-authorization}\clearpage
...
\input{sections/sec-15-funding-and-lobbying}\clearpage
\input{sections/sec-16-backmatter}
```

## Acceptance

- The compiled cover shows every line in the table above, in that order.
- The three DOI and URL links are clickable and none overflows the measure.
- Each `\docpart` begins a fresh page in the compiled PDF.

## Commit

One commit for `main.tex`, message `move-in/draft: main.tex with the La Jolla
cover, contents, and 17 section inputs`.
