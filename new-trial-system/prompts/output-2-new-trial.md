## output-2-new-trial

I began by reading rather than writing. The prior branch,
`claude/pancreatic-cancer-llm-paper-vkwc0j`, was merged into the working branch
before anything else, so that the new pull request would carry the whole of the
prior one rather than sit beside it. The first master prompt at
`new-trial-system/prompts/prompt-new-trial.md` was read in full, because the
second prompt requires the first prompt's PAPER TEMPLATE, PAPER CONTEXT, PAPER
FORMATTING, PROCESS COMMITS, SUB-PROMPT SCHEDULE and GITHUB sections to be
satisfied again rather than assumed. The stage 8 source set, the paper template
archive, the AI peer review archive under `new-trial-system/inputs`, and the
Figure 14 specification under `new-trial-system/plantuml` were read next.

A LaTeX toolchain is available in this session, which the first build did not
have. That single difference determined most of what follows, so I installed
TeX Live before writing a line, compiled the stage 8 bundle as deposited, and
found that it does not produce a PDF at all.

### The defect that came first

```
! Missing control sequence inserted.
<inserted text>
                \inaccessible
l.160 ...-0.12}) rectangle (\x1,{-0.35*\i+0.12});}
!  ==> Fatal error occurred, no output PDF file produced!
```

Three pgf `\foreach` headers in the stage 8 sections declare iteration macros
with a digit in the name:

| File | Header as filed | Figure |
|:--|:--|:--|
| `sec-03-ind.tex`, twice | `\foreach \i/\lab/\x0/\x1` | Figure 7 |
| `sec-05-legislation.tex` | `\foreach \i/\ttl/\b1/\b2/\b3/\st/\ts` | Figure 16 |
| `sec-06-funding-proposals.tex` | `\foreach \i/\lab/\p0/\p1/\d1` | Figure 17 |

A TeX control sequence cannot carry a digit, so `\x0` is `\x` followed by the
character `0`. pdfLaTeX halts inside Figure 7 and never reaches Figures 16 or
17. This is the defect that mattered most, because it made every other property
of the prior stage unverifiable: none of its twenty-five figures had ever been
rendered, and no reader could have opened a PDF of it. The headers are
re-declared with letter-only names in `update-final`, which is the first commit
on this branch.

After that repair the document sets in 53 pages with no error, no undefined
citation, no undefined reference, no bibtex warning, and after two further fixes
no overfull box above 5 pt.

### The output directory

`new-trial-system/final-new-trial/update-final` holds `main.tex`,
`trialstyle.sty`, `references.bib`, eleven section files, two READMEs and
`update-final-LaTeX.zip`. It is a complete Overleaf project and reads nothing
from its parent, so the parent stage stays exactly as deposited and remains a
readable build record. The cover page is carried forward unchanged, because the
first prompt fixes it: title, Draft 1.0, the DOI with a hyperlink, the author
with a green ORCID iD mark and a hyperlink, the independent-research notice, the
disclaimer naming Claude Code Opus 5, the deposit line, San Diego, and
August 14, 2026.

### Objective 3, the AI Peer Review section

The prior section argued the case on latency and reviewer count. That is true
and it is not actionable: a funder cannot act on "faster". Two subsections were
added that state the case in quantities with dates attached.

**What the delay costs.** Four costs, each carrying a number.

1. *Patients.* US pancreatic cancer deaths ran about 51,980 in 2025, which is
   about 142 a day and 1,000 a week. A best-case round of seven to eight weeks
   is about 52 days, so roughly 7,400 Americans die of the disease inside one
   best-case round, and a typical several-month processing time covers 17,000 to
   26,000. I wrote the paragraph so that it states plainly that peer review does
   not cause those deaths, because the honest version of the argument is
   narrower and harder to dismiss: review latency is a clinical quantity for
   this disease rather than an administrative one.
2. *Visual evidence.* Journals commonly hold a research article to six or eight
   display items. This paper carries twenty-five figures and twenty-five tables
   because each figure is a specification file rather than an image: twenty-five
   files, about 158 KB in total and about 6.5 KB each, measured from the
   deposited specification directories.
3. *Dollars and months.* List article processing charges commonly run \$2,000 to
   \$6,000 per article and exceed \$10,000 at several flagship titles. The
   comparison figure is exact and comes from the source study's own evaluation
   standard: `n_c = $35` across a fourteen-day workflow. The second half of the
   cost is the months, and the source study names them: the new regime
   "eliminates publication lull and cognitive load of uncertainty".
4. *The direction the money runs.* Under the author-pays model the researcher
   transfers the charge to the journal, and what the journal supplies is a page
   on its own site carrying the researcher's work. The source study states the
   alternative without qualification, that authors can now gain reputation
   "without the need for paid review or journals".

**What three days buys.** Three claims, each with a date.

1. This paper was finished within three days, August 12 to August 14, 2026, with
   twenty-five figures, twenty-five tables, eleven section files and about
   324,000 characters of LaTeX source.
2. Major paper and diagram code revisions run on the 1 to 2 day scale. The
   evidence for that claim is this stage rather than an assertion of it: a
   second prompt on August 13 required Figure 14 redrawn, this section
   rewritten, Table 21 and Figure 22 re-cut, and the compile defect found and
   fixed, and the result was deposited the following day.
3. On the record from late 2025 to the present, human peer review is not
   required for this class of work. This is the author's position and I wrote it
   plainly rather than hedged, because hedging it would misrepresent how the
   repository is run. I bounded it in the same paragraph: it concerns journal
   peer review of the author's own deposited works and does not touch
   institutional review board approval, an FDA information request, or a data
   and safety monitoring board.

Table 21 is re-cut to nine axes and Figure 22's grid to seven of them in the
same order, so the figure and the table cannot drift apart. Table 22's model
roster, Figure 23's concurrency and Figure 24's disagreement tree are carried
forward unchanged, by instruction: Anthropic in the production role, OpenAI as
first independent reviewer, Google as second, human PI holding final authority.

### Objective 4, Figure 14

The instruction is that the stick figures go, that the lines do not connect to
them, and that the humans look elementary. Both observations are correct and
they have the same cause. `\umlactor` draws five strokes and then places a
separate label node beneath them; the TikZ node that carries the actor's name is
the label, not the glyph, so every one of the thirteen associations terminated
at text rather than at the figure it pointed at.

I did not patch the anchor. A use case diagram states who touches what, and it
cannot state what each party owns without a line for every ownership relation.
A class diagram states ownership directly: the duty is written inside the party
that owes it. Six actor classes now carry their duties in a member compartment
with the duty numbers and the prior-law verdict on each, and the only lines left
are the six places where two parties actually interact. Thirteen lines become
six, all edge to edge, three inside a row and three inside a column, none
crossing another and none crossing a class. The 1.15 cm gutter is set by the
widest horizontal label; the 1.7 cm corridor between rows by the widest vertical
one.

The caption is unchanged, by instruction, and so is the surrounding context.
The specification at `new-trial-system/plantuml/fig-14-statutory-actor-duties.md`
was rewritten to match, with valid PlantUML class-diagram source, the
absolute-coordinate construction table, and the edge-routing argument.

### The revision pass

The first prompt requires at least one revision, and the second requires it
again. Because the prior stage produced no PDF, this is the first pass over
rendered pages that this paper has ever had. I rendered every figure page and
read them. Five defects:

| Where | Defect | Fix |
|:--|:--|:--|
| Figure 2 | The emphasis callout and the in-figure note overlapped across a 0.10 cm band | Callout moved below the grid's own bottom rule, note 0.50 cm further down |
| Figure 4 | The loop frame reaches the Author lifeline because row 7 returns to it, so the frame label sat on the Author activation bar | Label set 9.5 mm inside the west edge instead of at the corner |
| Figure 6 | The subject label sat 5.4 mm below the node center, inside the halo and inside the corridor four inbound edges converge through, so two edges ran through the words "IND v1.0, twelve modules"; the regulatory edge ran across the whole simulation cluster; the clinical edge crossed the device cluster's label row | Label moved above the halo on a 17 mm measure; every inbound edge re-terminated on the halo at its own bearing, 145, 180, 215 and 250 degrees; regulatory edge routed over the top at y = 2.05 and down at x = 11.70; clinical edge routed through the empty band between the device cluster's two tile rows |
| Table 13 | The bold `Characters` header did not fit a 1.55 cm column and overflowed by 8.19 pt | Columns re-cut |
| Sections 4 and 8 | Two paragraphs each spilled a two-line orphan onto an otherwise empty page | Both tightened |

A checker runs over the eleven section files on every build and reports zero
findings: 25 numbered figure captions and 25 numbered table captions, each two
lines within a four-character spread; every figure and table referenced by
number in the body; `\vspace{-0.6cm}` before every caption; every `tabularx` at
`\textwidth` with `\raggedright\arraybackslash` on every fixed column; no em
dash, en dash, double hyphen or triple hyphen; no non-US clinical term from the
first prompt's conversion list; and no raster anywhere.

### One instruction I could not satisfy exactly

The first prompt asks that each section marked "This is a main section" carry a
similar number of characters. The second prompt requires Section 7 to grow.
After the expansion, and after merging two of Section 7's older subsections and
trimming four more, the prose measures are:

| Section | Prose characters |
|:--|:--|
| 3, IND | about 12,500 |
| 4, Trial Protocol | about 12,400 |
| 5, Legislation | about 14,400 |
| 6, Funding Proposals | about 12,000 |
| 7, AI Peer Review | about 18,700 |

Section 7 runs longest. I closed part of the gap from the other side by giving
Sections 3, 4 and 6 a closing paragraph each tying their artifact to the review
clock, which is substantive cross-reference rather than padding, and I did not
close it further because doing so would have meant deleting material the second
prompt asks for. The prior stage's own spread was 11,325 to 14,414; this one is
12,000 to 18,700, and I am recording that rather than presenting it as met.

### Objective 5, the communications

Thirteen plain-text files under `new-trial-system/communications`, in three
sub-directories. Nine funding follow-up emails, one per organization in
`funding/pdac-funding-applications/applications/emailed-source`, which holds nine
archives; UC San Diego Moores is the intended feasibility partner rather than a
funding organization, so it has no email in the set. Each email follows its own
organization's `email-app-xx` format: `FROM`, `TO` and `CC` carried forward from
the original application, a subject, a body closing with the author's name,
company and August 14, 2026, an ordered attachment list with the paper first,
and a `BEFORE SENDING` checklist. Each body makes the argument that organization
actually cares about rather than the same argument nine times: Gate 1's critical
path for ARPA-H, the \$1,606,000 ask and the milestone ladder for SBIR, the
codified content-item crosswalk for CTEP, a shareable evidence layer for FNIH,
and what a time-bound organization can produce that the existing system
structurally cannot for the FRO.

One LinkedIn post, written to be posted with the paper attached as a document
rather than linked. Three brief general messages: patient and caregiver,
clinical investigator, and policy and press. The investigator message leads with
four checks that would falsify the paper's claims, ordered cheapest first,
rather than with the claims themselves.

Every message repeats the same four bounds. Model review is not regulatory
review. The investigational agent is never described as first-in-human. UC San
Diego is not a sponsor, site or endorser. H. R. 9510 is an author-drafted
proposal that has not been introduced. Nothing has been sent, and every address
carries an instruction to re-confirm it before use.

### Objective 6, the checks and the pull requests

Every check the repository runs was run locally before delivery: `ruff check .`
passed, `ruff format --check .` reported 525 files already formatted, `yamllint`
passed on `configs/` and the physics parameter mapping, `python -m py_compile`
passed on `scripts/verify_installation.py`, and `pytest tests/` reported 1,608
passed and 80 skipped. Pull request #75's own checks were green, so there was no
failing check to repair there; the work of #75 is carried into this branch by
merge rather than by copy, so the new pull request contains all of it plus this
update. #75 is closed with a comment pointing at its successor.

### Commit record

Every file was committed and pushed the moment it was finished, so the branch is
the production record rather than a report about it. The order was: the merge of
the prior branch; the compile fix with the new style vocabulary and the redrawn
Figure 14; the Figure 14 specification; the Section 7 rewrite with Table 21 and
Figure 22; the revision pass over the rendered pages; the stage READMEs and the
Overleaf bundle; `prompt-2-new-trial.md`; the thirteen communications; the
repository documentation; and this file.
