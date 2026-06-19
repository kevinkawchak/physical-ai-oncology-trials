## output-final-protocol

Stage 4 produced the polished, final Phase 1 protocol under
`trial-protocol/final-protocol/`. It begins from the full protocol and applies the
corrections a senior clinical-regulatory author makes on a last pass, learning
from what the full stage revealed.

The substantive corrections carried from the full stage are two figure
expansions. In the full stage the counterfactual-scenarios figure and the
Physical-AI-concerns figure had each been node-capped for the first pass; the
final stage restores them to the full fidelity of their Mermaid sources. Figure 3
now shows all three counterfactual scenarios (resection-window collapse,
vascular-injury cascade, and drug-restart mistiming), each as a clean
input-to-fork layout in which the human-only path shortens progression-free and
overall survival while the combination path preserves them, with no text-box or
arrow overlap. Figure 4 now shows the complete eight-pair mapping of every
Physical AI concern to its mitigation, matching the eight rows of Table 1, with
the hash-chained audit trail emphasized in Corporate Blue as the answer to the
black-box concern.

The formatting corrections are the senior-author techniques the build was asked
to learn. `main.tex` now starts every major NIH section on a fresh page with
`\clearpage`, so each of the thirteen sections is self-standing and no content is
stranded across a section boundary. `protostyle.sty` adds `\raggedbottom`, which
removes the large inter-paragraph white gaps a justified bottom would otherwise
create, directly addressing the instruction to avoid large empty white spaces.
The table-column widths were already tuned to the body measure in the full stage
and were confirmed here; the bibliography is set ragged-right so no reference line
ends with a single stranded word; the ORCID iD mark is rendered without any PNG
as a TikZ disc and is paired with the ORCID URL in the back matter; and the DOI
placeholder, the section symbol for every codified reference, and the single-hyphen
rule were verified across all thirteen sections.

A comprehensive static review confirmed the final state: balanced braces and
environments (47 begin, 47 end), every cite key resolving against
`references.bib`, twenty TikZ figures and eleven full-width tables across the
thirteen sections, fourteen `\clearpage` boundaries, and the daraxonrasib dose
ladder (160, 220, 300 mg) and sample size (n up to 18) consistent throughout. The
second-to-last commit recorded the consolidated error pass, and the final commit
carries the remaining repository updates: the directory README and Overleaf zip,
the root `CHANGELOG.md` and `releases.md` v4.0.0 entries, the root `README.md`
refresh, and the `prompts/output-protocol.md` narrative. This is the polished
source the author compiles in Overleaf.
