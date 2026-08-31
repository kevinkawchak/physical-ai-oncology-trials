# update-final - the second-prompt update over stage 8

[![Stage](https://img.shields.io/badge/Stage-Update%20over%208%20of%208-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final)
[![Paper](https://img.shields.io/badge/Paper-Draft%201.0-A32A3C.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system)
[![Compiles](https://img.shields.io/badge/pdfLaTeX-53%20pages%2C%20clean-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final)
[![Figures](https://img.shields.io/badge/Figures%20drawn-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final)
[![Tables](https://img.shields.io/badge/Tables-25-6B6B6B.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final)
[![Sections](https://img.shields.io/badge/Sections-11-2E2E2E.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final/sections)
[![Repository](https://img.shields.io/badge/Repository-v4.6.0-800020.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-C9C9C9.svg)](https://creativecommons.org/licenses/by/4.0/)

## What this stage is

A second author prompt over
[final-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial),
filed verbatim at
[new-trial-system/prompts/prompt-2-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/prompt-2-new-trial.md),
with the model's markdown output at
[output-2-new-trial.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/prompts/output-2-new-trial.md).

This directory is a complete, self-contained Overleaf project. It reads nothing
from its parent. The paper keeps its title, its cover page, its eleven sections,
its twenty-five figures and its twenty-five tables.

## What changed, and why

### 1. The bundle now compiles

This is the defect that mattered most, because it made every other property of
the prior stage unverifiable. Three pgf `\foreach` headers declared iteration
macros with a digit in the name:

| File | Header as filed | Figure affected |
|:--|:--|:--|
| `sections/sec-03-ind.tex` | `\foreach \i/\lab/\x0/\x1` (twice) | Figure 7, the IND assembly clock |
| `sections/sec-05-legislation.tex` | `\foreach \i/\ttl/\b1/\b2/\b3/\st/\ts` | Figure 16, the statute to SOP layers |
| `sections/sec-06-funding-proposals.tex` | `\foreach \i/\lab/\p0/\p1/\d1` | Figure 17, the funding artifact calendar |

A TeX control sequence cannot carry a digit, so `\x0` is `\x` followed by the
character `0`. pdfLaTeX raised `Missing control sequence` inside Figure 7 and
halted with no output file at all. The headers are re-declared with letter-only
names. The document now sets in 53 pages with no error, no undefined citation,
no undefined reference, no bibtex warning, and no overfull box above 5 pt.

### 2. Section 7, AI Peer Review, is rewritten around quantities

The final stage argued the case on latency and reviewer count. A funder cannot
act on "faster", so the section now states four costs the prior regime imposes
and three the new regime removes, each carrying a number and a date.

| The four costs | As quantified in the section |
|:--|:--|
| Patients | About 51,980 US pancreatic cancer deaths in 2025, about 142 a day and 1,000 a week; roughly 7,400 inside one 7 to 8 week best-case round, and 17,000 to 26,000 inside a typical several-month processing time |
| Visual evidence | Journals commonly hold an article to 6 to 8 display items; this paper carries 25 figures and 25 tables, because each figure is a 6.5 KB specification file rather than an image, 158 KB across all twenty-five |
| Dollars and months | List article charges of \$2,000 to \$6,000, above \$10,000 at flagship titles, against the source study's total spend of \$35 across a 14-day workflow; and 21 to 24 weeks a year lost to rounds at three papers a year |
| Direction of payment | The author pays the journal, and what the journal supplies is a page on its own site carrying the author's work |

| The three the new regime removes | As stated in the section |
|:--|:--|
| Elapsed time | This paper, with 25 figures and 25 tables, finished within three days, August 12 to 14, 2026 |
| Revision latency | Major paper and diagram code revision on the 1 to 2 day scale, evidenced by this stage itself |
| The requirement for human peer review | On the record from late 2025 to the present, human peer review is not required for this class of work, bounded so it does not touch IRB approval, an FDA information request, or a DSMB |

Table 21 is re-cut to nine axes and Figure 22's grid to seven of them in the
same order, so the figure and the table cannot drift apart. Table 22's model
roster, Figure 23's concurrency and Figure 24's disagreement tree are carried
forward unchanged: Anthropic in the production role, OpenAI as first independent
reviewer, Google as second independent reviewer, and the human principal
investigator holding final authority over all three.

### 3. Figure 14 is redrawn as a class diagram

The final stage drew Figure 14 as a use case diagram: six stick actors in two
edge columns, eleven ellipses inside a boundary, thirteen association lines. Two
defects follow from that choice. A stick actor's TikZ node is its caption, not
its glyph, so every association terminated at the text and the lines did not
appear to connect; and a five-stroke figure reads as elementary beside a bill
that amends the Federal Food, Drug, and Cosmetic Act.

Each actor is now a class: a name compartment carrying the party and its
stereotype, and a member compartment carrying the duties that party owes, with
each duty's number and the prior-law verdict on it. Ownership is read inside the
box rather than traced along a line, so thirteen association lines become six,
all edge to edge, three inside a row and three inside a column, none crossing
another and none crossing a class. The caption and the surrounding context are
unchanged, by instruction. The specification is at
[new-trial-system/plantuml/fig-14-statutory-actor-duties.md](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/plantuml/fig-14-statutory-actor-duties.md).

### 4. The first revision pass over rendered pages

Because the prior stage produced no PDF, none of its twenty-five figures had
ever been seen. Five defects were found on the compiled pages and fixed:
Figure 2's emphasis callout overlapped its in-figure note; Figure 4's loop frame
label sat on the Author activation bar; Figure 6 ran two edges through its own
subject label, ran the regulatory edge across the whole simulation cluster, and
ran the clinical edge through the device cluster's label row; Table 13's bold
`Characters` header overflowed its column by 8.19 pt; and two paragraphs each
spilled a two-line orphan onto an otherwise empty page.

## Files

| File | Contents |
|:--|:--|
| [main.tex](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/main.tex) | Cover page, badges, DOI, ORCID iD, notices, keywords, contents, eleven `\input` lines |
| [trialstyle.sty](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/trialstyle.sty) | Palette, float and caption machinery, and the five diagram vocabularies, now including the class diagram tokens `umlclshdr`, `umlclsbody`, `umlstereo`, `umlassoclbl` and the `\umlclass` constructor |
| [references.bib](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/references.bib) | 143 entries, every one with a working URL and a DOI where one exists |
| [sections/](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial/update-final/sections) | The eleven section files, one per `\input` |
| [update-final-LaTeX.zip](https://github.com/kevinkawchak/physical-ai-oncology-trials/blob/main/new-trial-system/final-new-trial/update-final/update-final-LaTeX.zip) | The whole project as one Overleaf upload |

## The paper

| Section | Title | Figures | Tables |
|:--|:--|:--|:--|
| 0 | Abstract, reader's guide, indexes | none | 1, 2, 3 |
| 1 | Introduction | 1, 2 | 4, 5 |
| 2 | Methods | 3, 4, 5 | 6, 7 |
| 3 | IND | 6, 7, 8, 9 | 8, 9, 10 |
| 4 | Trial Protocol | 10, 11, 12, 13 | 11, 12, 13 |
| 5 | Legislation | 14, 15, 16 | 14, 15, 16 |
| 6 | Funding Proposals | 17, 18, 19, 20 | 17, 18, 19, 20 |
| 7 | AI Peer Review | 21, 22, 23, 24 | 21, 22, 23, 24 |
| 8 | Limitations and Future Work | 25 | 25 |
| 9 | Conclusions | none | none |
| 10 | Back matter and references | none | glossary |

## Verification

A checker over the eleven section files confirms, on every build:

- 25 numbered figure captions and 25 numbered table captions, each opening with
  `Figure N.` or `Table N.`
- every caption exactly two lines, balanced within a four-character spread
- every figure and every table referenced by number in the body text
- `\vspace{-0.6cm}` before every caption, the same distance for every float
- every `tabularx` set to `\textwidth`, every fixed column carrying
  `>{\raggedright\arraybackslash}`
- no em dash, en dash, double hyphen or triple hyphen in prose
- no non-US clinical term from the master prompt's conversion list
- no raster anywhere in the bundle

## Compile

Upload `update-final-LaTeX.zip` to Overleaf and select pdfLaTeX, or locally:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Files from other directories used here

| Source directory | Used for |
|:--|:--|
| [new-trial-system/final-new-trial](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/final-new-trial) | The stage 8 source this update is taken from |
| [new-trial-system/template-new-system](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/template-new-system) | The cover block, contents block and one-input-per-section structure |
| [new-trial-system/prompts](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/prompts) | Both master prompts and both build outputs |
| [new-trial-system/plantuml](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/plantuml) | Figure 14's rewritten class diagram specification |
| [new-trial-system/inputs](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/inputs) | The AI peer review archive quoted throughout Section 7 |
| [funding/RFA-RM-27-001-v2](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/funding/RFA-RM-27-001-v2) | The Anthropic, OpenAI and Google role assignment of Table 22 |
| [new-trial-system/communications](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/new-trial-system/communications) | The thirteen outbound communications that carry this paper as their first attachment |

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
