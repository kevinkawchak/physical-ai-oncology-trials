## prompt-4-email-txt-and-attachments

**Stage.** PART I, Stage 4 of 5. **Output.** The ten
`email-app-NN-<slug>.txt` files and the ten Overleaf `.zip` bundles.

### The `.txt` contract

Every file is plain UTF-8 text, sent from `kevink@chemicalqdevice.com`, and has
exactly these blocks in this order and nothing else before them:

```
FROM: kevink@chemicalqdevice.com

TO: <exact recipient address>
CC: <exact recipient address, when a second office must see it>

SUBJECT: <exact subject line>

=== BODY ===
<exact body contents>

Sincerely,
CEO Kevin Kawchak
ChemicalQDevice
July 10th, 2026
=== END BODY ===

ATTACHMENTS COMPILED FROM THIS DIRECTORY
1. <pdf the author compiles from main.tex>

ATTACHMENTS THE AUTHOR MUST ADD BY HAND
1. <title>. <DOI or URL>.
...

BEFORE SENDING
- <verification steps>
```

The four closing lines carry no labels: `Sincerely,` `CEO Kevin Kawchak`
`ChemicalQDevice` `July 10th, 2026`, each on its own line.

### Address discipline

Use only an office or program mailbox that a funder publishes for unsolicited
scientific correspondence. Never invent a named individual's personal address.
Every `.txt` closes with a `BEFORE SENDING` block whose first line instructs the
author to confirm the address against the funder's current contact page, because
program mailboxes change.

### Manual-attachment selection rule

Each application names between three and six prior works, drawn from
`funding/supplementary/Physical AI Oncology Trial Founding Documents.md` and the
two application PDFs in `funding/pdfs/`, chosen for that recipient:

| Recipient kind | Attach |
|:--|:--|
| Person-based award | Funding Application v2.0; Phase 1 Protocol; PI Adoption Guide |
| Milestone agency | IND; Phase 1 Protocol; Phase 1 Guidance |
| Organization-type program | Bill v5.0; Founding Documents; Funding Application v2.0 |
| Mission platform | IND; Phase 1 Protocol; Bill v5.0 |
| Small business | Funding Application v2.0; Phase 1 Guidance; Efficient LLM Trial Simulations |
| Partnership vehicle | Funding Application v2.0; Phase 2 Protocol; Clinician narrative |
| Person-based philanthropy | Phase 1 Protocol; PI Adoption Guide; Founding Documents |
| Clinical evaluation program | IND; Phase 1 Protocol; Phase 2 Protocol; RASolute 302 |
| Time-bound organization | Founding Documents; Bill v5.0; Efficient LLM Trial Simulations |
| Host institution | Phase 1 Protocol; IND; Funding Application v2.0; UC San Diego priority steps |

### Zip contract (Rule 13)

Each application directory ships `app-NN-<slug>-LaTeX.zip` containing `main.tex`,
`appstyle.sty`, `references.bib`, and `sections/`, and nothing else, so the
author can drop it straight into Overleaf and compile with
`pdflatex -> bibtex -> pdflatex -> pdflatex`.

### Commits

One commit for the ten `.txt` files, one for the ten zips, one for the
per-application READMEs. Push each immediately.
