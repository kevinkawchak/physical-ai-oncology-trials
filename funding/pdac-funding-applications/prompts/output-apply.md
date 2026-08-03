## output-apply

I'll start by exploring the repository structure and all the source materials referenced in the prompt.

PR [#72](https://github.com/kevinkawchak/physical-ai-oncology-trials/pull/72) is open for live monitoring. Now the Process A sub-prompt schedules.

Now Part I Stage 1 — the shared style contract. This is the workhorse file.

TeX Live installed — I can verify every LaTeX source actually compiles. Let me test the style.

Found it — the `apptable` environment was picking up an interword space. Fixing.

Now Application 01 — NIH Director's Pioneer Award (surgical perspective).

Application 01 needs to fit 5 pages. Let me tighten it structurally.

The figures are stranding page space. Let me use the float carrier, as the parent work's final stage did.

`[!tb]` is the right placement — it recovers the stranded space. Applying.

App-01 is clean at 5 pages. Now the email and README.

Now applications 02–05 (surgical set). Building app-02, ARPA-H.

App-02 has room. Let me enrich it to full 5-page depth.

App-02 clean at 5 pages with 4 figures. Now its email, README, and zip.

Set A complete. Now Set B — medical oncology perspective. App-06, FNIH/AMP.

Now app-10 — the partner institution itself, grounded in the author's own priority-steps.

All ten applications are in. Now the Part I audit pass.

Only two categories flagged: the `--- BODY ---` delimiters in the `.txt` files, and BibTeX page ranges. The second is correct; the first is now `=== BODY ===`.

All ten now compile clean and within the five-page limit. Let me refresh the zips and commit the audit pass.

PART I is complete. Now PART II — Stage 1, the mermaid-type figures.

All 20 figure specifications done. Now PART II Stage 6 — `draft-apply`. First the paper style.

The `apptable` fix, the `\appfile` character scanner, and the table-of-contents lead reduction were the three changes needed to reach a clean draft compile.

Now Stage 7 — `full-apply`. This resolves every drafting instruction and draws all twenty figures.

All 20 figures drawn, 18 tables populated.

Clean: 0 errors, 0 overfull, 0 underfull. Full-apply README, zip, and commit.

Now Stage 8 — `final-apply`, the senior-author pass. Setting up and beginning the tightening.

The source breaks aren't the rendered breaks. Let me set explicit balanced three-line captions.

---

### What was built

**PART I — ten application file sets.** Ten recipient-unique Phase 1 pancreatic
cancer trial funding application email file sets under
`funding/pdac-funding-applications/applications/`, each carrying an email `.txt`
(recipients, subject, body, four-line closing ending `July 10th, 2026`, compiled
and manual attachment lists, pre-send checklist), a five-page LaTeX attachment
with its own cover variant, a self-contained `appstyle.sty` and
`references.bib`, one `.tex` per section, a README, and an Overleaf zip.

Applications 01 to 05 lead with the operation, 06 to 10 with the drug and
patient selection; both sets describe the same hybrid procedure. Every one is
dated August 3, 2026, names UC San Diego Moores Cancer Center as the intended
partner of choice at the feasibility stage only, and carries no DOI of its own.

**PART II — the summary paper.** Eight stages: five diagram-specification
directories (`mermaid/`, `plantuml/`, `d2/`, `diagrams-python/`, `graphviz/`),
then `draft-apply/`, `full-apply/`, and `final-apply/`. Twenty figures across
five machine-readable platforms in an uneven 6/3/4/3/4 split chosen by purpose.
No `publication/` subdirectory, no PDFs, no raster images anywhere.

### Verification performed

Every LaTeX source in this build was compiled with pdfLaTeX and BibTeX before it
was committed, not assumed to compile. Final state:

| Source set | Errors | Overfull | Underfull | Undefined citations | Pages |
|:--|:--|:--|:--|:--|:--|
| Ten application attachments | 0 | 0 | 0 | 0 | 4 to 5 each |
| `draft-apply` | 0 | 0 | 0 | 0 | 20 |
| `full-apply` | 0 | 0 | 0 | 0 | 33 |
| `final-apply` | 0 | 0 | 0 | 0 | 34 |

Audited and passing across all sources: no black fill outside comments; every
figure followed by exactly `\vspace{-0.7cm}` then `\figcaption`; every
fixed-width column ragged-right; every table at `\textwidth`; at most five
figures per application; four-line unlabelled closing; sent from
`kevink@chemicalqdevice.com`; manual-attachment instructions present in all ten.

### Six defects found and fixed during the build

1. **`apptable` overfull by 2.74pt in every table.** The newline after
   `\end{tabularx}` contributed an interword space past the right margin. Fixed
   with `\unskip` in the environment's closing code.
2. **Figures stranding up to a page of whitespace.** Inline figures with
   `\needspace` forced breaks. Fixed by carrying every figure in an `[!tb]`
   float, which is what the parent work's own final stage does.
3. **`\appfile` overflowing by up to 188pt.** A repository path has no spaces;
   both `\nolinkurl` and `\path` failed inside a macro argument. Fixed with a
   character scanner that inserts a `\penalty300` break after every character.
4. **`\apptimeline` failing to compile.** It chose fills with `\ifnum` inside a
   TikZ key list, which expands too late, and passing braced arguments through
   `\foreach` broke `\XC@col@rlet`. Rewritten to place five cells explicitly.
5. **Missing glyph macros.** `\glyphlink`, `\glyphcloud`, `\glyphsignal`,
   `\glyphhand` and `\glyphteam` were dropped when adapting the parent style and
   are restored.
6. **Two underfull boxes from unbreakable `\href` display strings.** Changed to
   `\url`, which is clickable under `hyperref` and honours `\UrlBreaks`.

### One substantive correction, recorded rather than absorbed

The author's own `funding/potential-partners/UC-San-Diego/priority-steps.md`
records that daraxonrasib must not be described as first-in-human: it is
investigational and already in Phase 3 evaluation. Nine of the ten applications
carried that error until the PART I audit pass. Each now carries a positioning
note, every email body and subject is corrected, and all ten carry the
instruction in their pre-send checklist. The correction is stated in §7 of the
summary paper rather than quietly fixed.

### Two things I could not do as literally specified

1. **`robotic-surgeries/tree/main/patient-robot-advocacy` READMEs.** The prompt
   asks for a comprehensive README in every directory of that path, but Rule 1
   restricts commits to `physical-ai-oncology-trials`. `robotic-surgeries` is a
   different repository, so nothing was written there. Every new and existing
   directory under `physical-ai-oncology-trials/funding/` did receive one.
2. **The one-quarter length target.** `final-apply` reaches 1/3.14 on prose and
   1/2.85 on source against a target of approximately 1/4. The exact figures are
   recorded in the stage README along with the two levers that would close the
   gap and what each would cost, rather than cutting further, because what
   remains is argument rather than words.

### Recipient addresses

Every address used is a published office or programme mailbox, never a personal
address invented for a named individual, except in application 10, where the two
named UC San Diego clinicians and the Clinical Trials Office address are taken
verbatim from the author's own `priority-steps.md`. Every `.txt` opens its
pre-send checklist with an instruction to confirm the address against the
recipient's current contact page, because programme mailboxes change and a
bounced inquiry is not resubmitted automatically.
