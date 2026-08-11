# Capitalization Plan - From Independent Scientist to Novel Performer (v4.5.0)

[![Version](https://img.shields.io/badge/Repository-v4.5.0-00417A.svg)](https://github.com/kevinkawchak/physical-ai-oncology-trials)
[![Paper](https://img.shields.io/badge/Paper-Draft%201.0-00417A.svg)](final-capital)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21887807-3C7DB2.svg)](https://doi.org/10.5281/zenodo.21887807)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-A6CE39.svg)](https://orcid.org/0009-0007-5457-8667)
[![Figures](https://img.shields.io/badge/Figures-20-6C757D.svg)](#the-twenty-figures)
[![Platforms](https://img.shields.io/badge/Diagram%20platforms-5-6C757D.svg)](sub-prompts)
[![Tables](https://img.shields.io/badge/Tables-21-6C757D.svg)](final-capital)
[![Stages](https://img.shields.io/badge/Build%20stages-8-9AA1A8.svg)](sub-prompts)
[![Rasters](https://img.shields.io/badge/PNG%20%2F%20JPG-none-9AA1A8.svg)](.)

**From Independent Scientist to Novel Performer: A Small-Business Operating,
Milestone, and Capitalization Plan for a Phase 1 LLM-Advised Robotic Whipple
(ChemicalQDevice).** Draft 1.0. Kevin Kawchak, ChemicalQDevice, San Diego,
August 11, 2026. DOI [10.5281/zenodo.21887807](https://doi.org/10.5281/zenodo.21887807).

## What this is, and what changed

The ten applications filed in
[`../pdac-funding-applications`](../pdac-funding-applications) were written by an
independent scientist. Nine of the ten are addressed to mechanisms that fund a
person or an institution. One, application 05, is addressed to the only
mechanism in the set that funds a **company**, and it was the shortest of the
ten.

This paper converts the author from independent scientist to small-business
operator and rewrites that one application as the document it should have been:
an operating plan, a milestone schedule, and a capitalization plan for
ChemicalQDevice. The clause it turns on is not the report's Pioneer language and
not its Novel Performers chapter. It is the SBIR clause, which appears in three
separate chapters of *Science: A New Golden Age* and is the only one in the
report written for a firm.

The cost of the conversion is stated in [§9](final-capital/sections) rather than
hidden: the individual-scientist framing that made the ten applications
distinctive is abandoned, capitalization detail the author might prefer private
is published, and three of the ten prior recipients will now read this as
off-mechanism.

## The single arithmetic the paper is built on

| Quantity | Value | Source |
|:--|:--|:--|
| Programme, five years, direct | $3,500,000 | `../pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` |
| Programme, per year, direct | $700,000 | Same |
| SBIR Phase I, total cost, 9 months | $306,000 | `../pdac-funding-applications/applications/app-05-nih-sbir-seed` |
| SBIR Phase II, total cost, 24 months | $1,300,000 | Same |
| SBIR route, total cost, 33 months | $1,606,000 | Sum |
| SBIR route, direct work inside that award | $1,396,000 | §3, Tables 8 and 11 |
| Delta, direct, that $1.606M does not buy | $2,104,000 | §3, Table 11 |
| Private capital raised behind the firewall | $5,900,000 | §4, Table 13 |
| Private to federal leverage | 3.67 to 1 | Against the annex's 3:1 target |

Every table in the paper reconciles to these nine numbers. Where a figure
carries money, it carries one of them.

## Repository structure

```
funding/capitalization-plan/
├── README.md                          this file
├── prompts/
│   ├── README.md
│   ├── prompt-capital.md              the master prompt, verbatim
│   ├── update-capital.md              the update prompt, verbatim
│   └── output-capital.md              the full Claude Code output
├── sub-prompts/
│   ├── README.md                      the eight-stage schedule
│   ├── stage-1-mermaid/README.md
│   ├── stage-2-plantuml/README.md
│   ├── stage-3-d2/README.md
│   ├── stage-4-diagrams-python/README.md
│   ├── stage-5-graphviz/README.md
│   ├── stage-6-draft-capital/README.md
│   ├── stage-7-full-capital/README.md
│   └── stage-8-final-capital/README.md
├── mermaid/                           Figures 1, 7, 12, 13, 19
├── plantuml/                          Figures 6, 11, 14
├── d2/                                Figures 2, 5, 8, 10, 16
├── diagrams-python/                   Figures 4, 18, 20
├── graphviz/                          Figures 3, 9, 15, 17
├── draft-capital/                     stage 6: skeleton, 20 sized slots
│   ├── main.tex  capstyle.sty  references.bib  README.md
│   ├── sections/sec-00 .. sec-11.tex
│   └── draft-capital-LaTeX.zip
├── full-capital/                      stage 7: 20 figures drawn, 21 tables
│   ├── main.tex  capstyle.sty  references.bib  README.md
│   ├── sections/sec-00 .. sec-11.tex
│   └── full-capital-LaTeX.zip
└── final-capital/                     stage 8: senior-author pass
    ├── main.tex  capstyle.sty  references.bib  README.md
    ├── sections/sec-00 .. sec-11.tex
    ├── main.pdf                       44 pages, 0 overfull boxes
    └── final-capital-LaTeX.zip
```

## The update pass

One update pass followed the eight-stage build, driven by
[`prompts/update-capital.md`](prompts/update-capital.md). It changed the paper
in [`final-capital`](final-capital) and the documentation in this directory;
[`draft-capital`](draft-capital) and [`full-capital`](full-capital) are the
record of what stages 6 and 7 produced and are unchanged.

| What changed | Where | Result |
|:--|:--|:--|
| Every caption opens with its own number | 20 `\figcaption`, 21 `\tabcap` | `Figure N.` and `Table N.`, three centred lines each |
| Horizontal centring fixed at source | `final-capital/capstyle.sty` | All 20 frames at 306.00 pt, all 41 captions within 0.53 pt |
| Four figures renumbered to printed order | §4 and §5 | 10 and 11 swapped, 14 and 15 swapped, everything moved with them |
| Every float referred to in the running text | all twelve sections | 20 of 20 figures, 21 of 21 tables |
| DOIs and URLs made clickable | `sec-11` and `capstyle.sty` | `unsrturl` in place of `unsrt`: 20 DOIs, 17 URLs, all linked |
| The $36,330 figure described as projected | §2 and Table 17 | "projected", not "estimated" |
| `main.pdf` and the zip rebuilt together | [`final-capital`](final-capital) | One pass, one source set, 44 pages |

The horizontal-centring defect is worth stating plainly, because it had been
invisible in the source and was only found by measuring the compiled PDF. Both
caption carriers and the figure frame centred themselves with a leading and a
trailing `\hfil`. TeX deletes a glue item that ends a paragraph, so the trailing
one never survived, the leading one absorbed the whole slack, and every figure,
table caption and figure caption in the paper sat 13.1 pt right of centre. All
three carriers now close with `\null`.

## The twelve sections

| § | Section | Outline item | Figures |
|:--|:--|:--|:--|
| 0 | Abstract and Reader's Guide | Front matter | Tables 1, 2, 3 |
| 1 | The Novel-Performer Case | A | 1, 2, 3 |
| 2 | The Entity and the Asset | B | 4, 5, 6 |
| 3 | The $1.6M Gate and the $3.5M Programme | C | 7, 8, 9 |
| 4 | Non-Dilutive to Dilutive Bridge | D | 10, 11, 12 |
| 5 | Twelve Milestones a Program Officer Can Audit | E | 13, 14, 15 |
| 6 | The Clinical Evidence a Funder Is Buying | Supporting | 16, 17 |
| 7 | Small-Business Operating Plan | Supporting | 18 |
| 8 | San Diego and the August 2026 Record | Supporting | 19 |
| 9 | Risks, Stop Conditions, and What This Is Not | Supporting | none |
| 10 | Build Method and Reproducibility | Supporting | 20 |
| 11 | Back Matter | Supporting | none |

## The twenty figures

| # | § | Platform | Perspective |
|:--|:--|:--|:--|
| 1 | 1 | Mermaid | Three candidate clauses through one eligibility filter |
| 2 | 1 | D2 | This company scored against the report's own institutional-form table |
| 3 | 1 | Graphviz | The same direct work under a university rate and under this company's |
| 4 | 2 | Diagrams | Owned, licensed, contracted, absent, as four zones |
| 5 | 2 | D2 | The asset register as typed records with encumbrance |
| 6 | 2 | PlantUML | What the sponsor may do alone and what only the site may do |
| 7 | 3 | Mermaid | The Phase I to Phase II award state machine and its four guards |
| 8 | 3 | D2 | One programme, two prices, and the delta as a third column |
| 9 | 3 | Graphviz | Work packages reachable at $306K, at $1.606M, and not at all |
| 10 | 4 | D2 | Three capital tiers with the firewall drawn as the gap |
| 11 | 4 | PlantUML | The part 54 capital firewall as five states with guards |
| 12 | 4 | Mermaid | Who signs what, in what order, during a financing event |
| 13 | 5 | Mermaid | Thirty-three months, twelve milestones, twelve artifact dates |
| 14 | 5 | PlantUML | Evidence production and audit running concurrently |
| 15 | 5 | Graphviz | What has to fail for the programme to stop, by combination |
| 16 | 6 | D2 | Six published quantities with intervals a funder can check |
| 17 | 6 | Graphviz | Published quantity to IND to protocol to milestone |
| 18 | 7 | Diagrams | Where the 2.6 FTE, the compute, and the site functions sit |
| 19 | 8 | Mermaid | What each July and August 2026 contact unlocks |
| 20 | 10 | Diagrams | Which custodian holds which artifact if the programme stops |

Five mermaid, three plantuml, five d2, three diagrams, four graphviz. The split
follows purpose, not quota, and the reasoning is in
[`sub-prompts/README.md`](sub-prompts/README.md).

## Rule 5 source map

| Used | From | Where it appears |
|:--|:--|:--|
| `LaTeX Source Files.zip` | `../RFA-RM-27-001-v2` | The cover theme varied from, in `main.tex` of all three stages |
| `final-apply/publication/LaTeX Source Files.zip` | `../pdac-funding-applications` | `capstyle.sty` inherits its five TikZ vocabularies and its spacing invariant |
| `final-apply/references.bib` | `../pdac-funding-applications` | `references.bib`, extended with nine regulatory and policy entries |
| `final-apply/sections/sec-08-budget-and-leverage.tex` | `../pdac-funding-applications` | The four-layer $3,500,000 frame, reused verbatim in §3 |
| `final-apply/publication/useredits.md` | `../pdac-funding-applications` | The correction list applied in stage 8 |
| `applications/emailed-source/` | `../pdac-funding-applications` | The nine applications sent 8/4/26 to 8/8/26, cited in §8 |
| `applications/app-05-nih-sbir-seed/` | `../pdac-funding-applications` | The $306,000 and $1,300,000 split, expanded in §3 |
| `chunk-01`, `03`, `04`, `05`, `08` | `../science-golden-age` | Every policy quotation in §1 |
| `Physical-AI-Oncology-Trial-Competition-Proposal.zip` | `../supplementary/source-files` | The January 13, 2026 baseline in §2 |
| `UC-San-Diego/` | `../potential-partners` | The Moores record and the positioning correction in §8 |
| `trial-protocol/`, `trial-ind/`, `trial-phase-2/` | repository root | The asset register in §2 and the evidence chain in §6 |

## Positioning constraints

Nothing here is a submission of record and nothing here is an agreement. UC San
Diego and Moores Cancer Center are named as the intended partner of choice at
the feasibility stage only; no agreement of any kind exists. No drug supply
agreement, letter of authorization, or regulatory cross-reference is in place
with the agent's developer. No robotic configuration has been specified or
cleared. No patient has been treated. Daraxonrasib is investigational and
already in Phase 3 evaluation and is nowhere described as first in human; the
supportable novelty claim concerns the integrated surgical and advisory
workflow. The capitalization figures are a plan, not a completed raise: no term
sheet, SAFE, or subscription agreement exists.
