## master-prompt

> This file preserves the single submitted Master prompt word-for-word, filed
> per Rule 14 of the prior build ("File this submitted prompt and all AI outputs
> using the prior prompt's instructions as a template"). The prior template is
> the `auto-bill-01/master-prompt.md` convention: a `## master-prompt` heading
> followed by the prompt verbatim, then the derived rules and the derived prompt
> schedule that govern the run. Each stage of this run follows the same
> convention under `auto-bill-02/sub-prompts/` (the generated sub-prompts) and
> the per-stage `output-*.md` files (the narrative outputs).

"PROCESS"
Generate H. R. 9510 v5.0 (single-prompt-bill/tree/main/auto-bill-02, repo
v2.0.0) that a) Comprehensively focuses on financial data, and b) Contains
tables, ASCII, and mermaid diagrams. Build from
cancer-automated/tree/main/papers/VVUQ-05/final-bill (no /deliverables); and
adapts to single-prompt-bill/blob/main/auto-bill-01/master-prompt.md (update
sub-prompts, auto-commit/PR schedule; no shortcuts).
"PROCESS"

## Derived rules (adapted from `auto-bill-01/master-prompt.md`)

The seventeen rules of the prior Master prompt carry forward, re-targeted for
the v5.0 build. Where the prior rule named v4.0 objects (the twelve
deliverables, Mermaid-only diagrams), the rule is updated to the v5.0 objects
(the comprehensive financial-data focus; tables, ASCII, and Mermaid together).

1. Only commit to the `kevinkawchak/single-prompt-bill` repository, with new
   files going into `main/auto-bill-02` and a comprehensive README at each
   level. Subdirectories also have READMEs. Do not commit to other
   repositories.
2. The bill must comprehensively focus on financial data: an operative
   financial core (financial data records and transparency, device user-fee
   treatment, an authorization of appropriations, and a budgetary-effects
   clause) plus non-operative financial appendices (cost estimate and
   budgetary data, verification economics, a financial-data transparency
   standard, and a financial research-influence matrix), every part
   represented by a properly formatted, clickable table of contents in the
   bill.
3. Appendices, "Authorship, Notices, and Provenance", and References must be
   present and correct. Follow the prior improvements regarding reference
   white-space formatting and no single words per line.
4. Only the single Claude Code agent session that received this Master prompt
   may author the build, end to end. Do not stall, ask questions, or go into
   plan mode.
5. No raster images are allowed anywhere. The permitted visual media are
   exactly three, each used where it is the correct representation: full-width
   white-background tables; monospace ASCII figures (preserved spacing, black
   rules on white); and gray-scale Mermaid diagrams (real `mermaid` blocks in
   Markdown, matching gray-scale TikZ figures in the compiled LaTeX).
6. Each README.md for each directory must be comprehensive and state which
   files from other directories were used and where. For instance: No. 03
   Mermaid diagram - `Clinical-AI-Demos/tree/main/ai-outputs/output-01`.
7. 1 commit must be used for each distinguishable task (each `.tex`, `.md`,
   etc.).
8. The prior one-PR-per-stage schedule is updated for this run's execution
   environment, which permits exactly one development branch
   (`claude/funny-meitner-7ywlgc`): each prior PR becomes one stage milestone
   (a contiguous, labeled set of real-time commits), and all nine milestones
   are tracked in one continuously updated pull request whose description is
   refreshed as each milestone lands.
9. All commits must be submitted to GitHub in real-time the moment they are
   generated for user viewing, and the single pull request must be opened at
   the first milestone and kept current. Do not hold commits from GitHub as
   they are completed.
10. There must be a single last v2.0.0 update that provides the CHANGELOG.md
    entry, the releases.md entry, and the repository updates.
11. The agent has permission to commit to the designated branch, push in real
    time, and open and update the single pull request in GitHub.
12. Do not make shortcuts from prompt to prompt: every stage must be fully
    developed following prior specified prompts. All files generated from
    prompts must be present and working. Each set of LaTeX files must compile
    properly in Overleaf.
13. Don't stop until all tasks are completed. The user will continue the
    session by using the phrase "Continue." if tokens are exhausted. This is a
    lengthy process.
14. File this submitted prompt and all AI outputs using the prior prompt's
    instructions as a template.
15. Leave the v5.0 DOI in the format: 10.5281/zenodo.xxxxxxxx (with Hyperlink:
    https://doi.org/10.5281/zenodo.xxxxxxxx) pending deposit. Known prior
    DOIs are carried: Bill v4.0 10.5281/zenodo.20576907, Bill v3.0
    10.5281/zenodo.20535429, Bill v2.0 10.5281/zenodo.20485580, repository
    10.5281/zenodo.20576648.
16. Learn and implement the author's `\clearpage`, table-formatting column
    widths, and other `\vspace` and `\hspace` formatting methods, and the
    author's other corrections and proof-reading techniques, to create
    polished final-bill source files. The draft-bill, full-bill, and
    final-bill developments must each have their own `.tex` outputs and a
    LaTeX zip file that runs properly in Overleaf.
17. The agent will be judged on how well it followed these rules and the
    prompt schedule after it finishes, by the author and other people.

## Derived prompt schedule (updated for v5.0)

The seven-stage schedule of the prior build carries forward with updated
sources and the financial-data target. Stage 3 widens from Mermaid-only to
Mermaid and ASCII selection, because Bill v5.0 restores the ASCII medium
alongside Mermaid and tables. Milestones M1 through M9 are the auto-commit
schedule of Rule 8.

| Milestone | Stage | Adapted from (prior source) |
|:--|:--|:--|
| M1 | Bootstrap (Process A): README, this file, seven sub-prompts; open the PR | `auto-bill-01/master-prompt.md` |
| M2 | 1. Automatic financial research (`01-research/`) | `cancer-automated/.../VVUQ-05/update-bill/next-steps/prompt-1-next-steps.md`; `auto-bill-01/sub-prompts/prompt-1-research.md` |
| M3 | 2. Template bill selection (`02-template-selection/`) | candidates: `cancer-automated/.../VVUQ-05/final-bill` (v3.0 ASCII and tables) vs `auto-bill-01/final-bill` (v4.0 Mermaid TikZ) |
| M4 | 3. Mermaid and ASCII selection (`03-mermaid-selection/`) | `Clinical-AI-Demos/.../ai-outputs/output-01`; `cancer-automated/.../VVUQ-05/update-bill/figures-bill` |
| M5 | 4. Figure selection (`04-figure-selection/`) | `cancer-automated/.../VVUQ-05/update-bill/figures-bill/prompt-figures-bill.md` |
| M6 | 5. Bill draft template (`draft-bill/`) | `cancer-automated/.../VVUQ-05/draft-bill/prompt-draft-bill.md` |
| M7 | 6. Bill full version (`full-bill/`) | `cancer-automated/.../VVUQ-05/full-bill/prompt-full-bill.md` |
| M8 | 7. Bill final version (`final-bill/`, repository v2.0.0) | `cancer-automated/.../VVUQ-05/final-bill`; the prior draft-to-full-to-final corrections |
| M9 | Release: CHANGELOG.md, releases.md, repository updates (Rule 10) | `auto-bill-01` release milestone |
