# CHANGELOG entry - v3.9.2

This file is a standalone changelog entry that complements `CHANGELOG.md`. It is
intended to be merged into the top of the root `CHANGELOG.md` at the next
aggregate update. The full release notes for this version live at
`releases/v3.9.2.md`.

## [3.9.2] - 2026-05-11

### Added

- Ten distinct single column LaTeX academic paper template directories under
  `generative-ai/templates/`:
  - `01-clinical-trial-protocol/` - Clinical Trial Protocol perspective
    (Times Roman classic, neutral black).
  - `02-robotic-surgery-engineering/` - Robotic Surgery Engineering perspective
    (modern sans-serif with **dark blue** section accents on title, abstract
    heading, section headings, and rules; HTML 0A3D62).
  - `03-bioinformatics-genomics/` - Bioinformatics and Genomics perspective
    (MDPI-inspired clean look with italic abstract).
  - `04-regulatory-fda/` - Regulatory and FDA Submission perspective
    (Palatino serif with **dark blue navy** accents; HTML 0B2545).
  - `05-patient-centered-outcomes/` - Patient-Centered Outcomes Research
    perspective (slim minimalist Charter body).
  - `06-health-economics/` - Health Economics and Outcomes Research
    perspective (bold display, 22 pt sans-serif title).
  - `07-ai-ml-methods/` - AI and Machine Learning Methods perspective
    (technical monospaced section headings).
  - `08-digital-twin-simulation/` - Digital Twin and Simulation perspective
    (elegant Palatino serif).
  - `09-multi-site-federation/` - Multi-Site Federation and Federated Learning
    perspective (journal-style uppercase headings).
  - `10-translational-oncology/` - Translational Oncology perspective
    (Computer Modern serif, LaTeX default).
- `generative-ai/templates/README.md` template family index documenting the
  shared invariants, perspective table, file layout, compile recipe, and zip
  generation recipes for Linux/MacOS and Windows PowerShell.
- Per-template `main.tex`, `new_paper.sty`, `references.bib`, `README.md`,
  and a `sections/` subdirectory holding eight per-section `.tex` files:
  `abstract.tex`, `introduction.tex`, `methods.tex`, `results.tex`,
  `discussion.tex`, `limitations_future.tex`, `conclusions.tex`,
  `back_matter.tex`. Every section ships with one illustrative paragraph and
  a three-row "Table 2" layout using the column width pattern
  `>{\raggedright\arraybackslash}p{3.4cm}` / `p{4.6cm}` / `p{5.4cm}` to
  prevent rivers.
- Blank single-line `\keywords{ }` slot on page one of every template
  directly under the abstract.
- `releases/v3.9.2.md` canonical record for the v3.9.2 release notes block
  in the standard Title / Summary / Features / Contributors / Notes format.
- One Overleaf-ready zip bundle alongside template 01
  (`generative-ai/templates/01-clinical-trial-protocol.zip`) as a starting
  point; the templates README documents the recipe to generate the other
  nine bundles locally on demand.

### Changed

- `README.md` (root) release badge bumped from v3.9.1 to v3.9.2, with a new
  `5/11: v3.9.2 (Generative AI LaTeX Paper Templates)` entry directly above
  the existing `5/10: v3.9.1` block, plus a `Generative AI LaTeX Paper
  Templates (v3.9.2)` ASCII pipeline diagram and a top-level repository
  structure listing that names the new `generative-ai/templates/` directory.

### CI and quality

- The pull request adds only LaTeX, Markdown, and one binary zip file under
  `generative-ai/templates/` and `releases/`. No Python or YAML files are
  added or modified.
- The repository CI matrix (`ruff format --check`, `ruff check`, `yamllint
  -d relaxed`) on Python 3.10, 3.11, and 3.12 remains green because all
  new files are outside the lint scope.

### Notes

- The example template at
  `kevinkawchak/robotic-surgeries/tree/main/2030-gbm-1min/paper/full-paper/final-paper`
  was used as a read-only reference for the title page metadata pattern,
  the back-matter layout, the column type macros, the ORCID iD fallback
  block, and the formatting invariants (widows, orphans, rivers,
  raggedright tables, hyperlink color). No content was committed to the
  `robotic-surgeries` repository as part of this work, per the explicit
  instruction.
- Two of the ten templates (02 Robotic Surgery Engineering and 04
  Regulatory and FDA Submission) specify dark blue text for relevant
  sections per the requirement. The other eight templates render all body
  text in black with no color accents.
- All ten templates are single column format, do not include line numbers
  in columns, and do not include preprint editing formatting.
