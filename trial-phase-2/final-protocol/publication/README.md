# Zenodo Publication with Author Edits - Phase 2 (paper URL directory) (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Stage](https://img.shields.io/badge/Stage-4%20final%20%2B%20author%20edits-800020.svg)](../../sub-prompts/prompt-4-final-protocol.md)
[![Design](https://img.shields.io/badge/Design-Phase%202%20Randomized%20Multicenter-800020.svg)](.)
[![Sections](https://img.shields.io/badge/NIH%20sections-13-800020.svg)](sections)
[![Figures](https://img.shields.io/badge/TikZ%20figures-22-6B6B6B.svg)](.)
[![Tables](https://img.shields.io/badge/Full--width%20tables-11-6B6B6B.svg)](.)
[![Compiles](https://img.shields.io/badge/Overleaf-pdfLaTeX-6B6B6B.svg)](main.tex)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-A6CE39.svg)](https://orcid.org/0009-0007-5457-8667)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20807027-800020.svg)](https://doi.org/10.5281/zenodo.20807027)
[![Release](https://img.shields.io/badge/Release-v4.1.0-orange.svg)](../releases.md)

[Publication with Author Edits](https://github.com/kevinkawchak/physical-ai-oncology-trials/tree/main/trial-phase-2/final-protocol/publication/author) (final-protocol/publication/author; the paper URL directory).

This directory is the **paper URL directory** for the Phase II protocol: the
author-edited, publication-ready build of the *Phase 2, Multicenter, Randomized,
Controlled Clinical Trial Protocol of On-Premises LLM-Directed Robotic
Pancreaticoduodenectomy (Whipple) with Perioperative Daraxonrasib (RMC-6236) in
KRAS-Mutated Pancreatic Ductal Adenocarcinoma*. It is the maximum-quality output
of the mermaid -> draft -> full -> final pipeline plus a senior-author edit pass,
recolored to the five-step Phase II palette with Burgundy `#800020` as the
document color.

- **Author:** Kevin Kawchak, CEO ChemicalQDevice
  ([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667))
- **DOI:** [`10.5281/zenodo.xxxxxxxx`](https://doi.org/10.5281/zenodo.xxxxxxxx)
  (pending deposit) - **Date:** June 23, 2026 - **Version:** v1.1.0
- **Predicate:** the Phase 1 protocol (v1.0.0,
  [`10.5281/zenodo.20780121`](https://doi.org/10.5281/zenodo.20780121)) that
  established the daraxonrasib RP2D and the device feasibility this trial builds on.

## What this protocol is

It is the **randomized controlled efficacy study** that the Phase 1 first-in-human
protocol explicitly deferred. The Phase 1 protocol established the daraxonrasib
recommended Phase 2 dose (RP2D, 300 mg once daily) and the feasibility and safety
of the on-premises LLM-directed eight-arm robotic Whipple; with those questions
answered, genuine clinical equipoise exists for a randomized comparison.

## Protocol at a glance

| Element | Value |
|:--|:--|
| Design | Phase 2, multicenter (8 centers), randomized 1:1, parallel-group, controlled, open-label with BICR |
| Arm A (experimental) | Perioperative daraxonrasib at RP2D (300 mg once daily) + on-premises LLM-directed eight-arm robotic Whipple |
| Arm B (control) | Modified FOLFIRINOX + institutional-standard high-volume pancreaticoduodenectomy |
| Sample size | 220 randomized (110 per arm); up to about 245 enrolled |
| Primary endpoint | Progression-free survival (PFS); HR 0.60; 85 percent power; two-sided alpha 0.05; about 140 events; one group-sequential interim |
| Key secondary (hierarchical) | OS; R0 rate; ISGPS grade B/C fistula; major pathologic response; ctDNA clearance |
| Device readiness | Phase 0 USL >= 8.0; >= 5000 sims; >= 3 frameworks; sim-to-real < 1.5 mm / < 0.4 N; fleet harmonization |
| Safety envelope | 3 N per-arm / 18 N cumulative caps; <= 3 ms cross-arm E-stop; five-vessel no-fly gate |
| Funding | Patient-Aligned Co-Investment Facility behind a capital firewall (21 CFR part 54; H.R. 9510 VVUQ standard) |

## Files

```
publication/
  main.tex                  cover, clickable TOC, \clearpage per section
  protostyle.sty            recolored #800020; raggedbottom; mermaidfig, tables, TikZ ORCID
  references.bib            daraxonrasib (5), main documents (3), Phase 1 predicate, author works, clinical, methods, standards
  sections/                 sec-00 .. sec-12 (13 NIH sections, 22 figures, 11 tables)
  README.md                 this file
  LaTeX Source Files.zip    Overleaf bundle
```

## Files from other directories used here

| Source | Used for |
|:--|:--|
| `../../../trial-protocol/final-protocol/publication` | the Phase 1 paper whose structure and figure alignments this build adapts |
| `../../mermaid/` | the 24 Phase 2 Mermaid figures reproduced here as TikZ `mermaidfig` |
| `../../../trial-protocol/inputs/2030-pdac-1min-final-paper` | the quantitative clinical and telemetry data |
| `../../../trial-protocol/inputs/21cfr312_adapt` | the Physical AI Subpart J overlay |
| `../../../trial-protocol/inputs/auto-bill-02` | the VVUQ and co-investment financial framing |
| `../../../trial-protocol/nih-protocol/` | the NIH-FDA Phase 2/3 IND/IDE template (section order and required content) |

## Quality verification (static)

13 NIH sections; 22 TikZ figures numbered 1 through 22; 11 full-width tables (all
defined and referenced); every `\cite` key resolves; balanced environments; the
locked constants (n = 220, HR 0.60, 85 percent power, about 140 events, RP2D 300
mg, eight sites, USL >= 8.0, >= 5000 sims, >= 3 frameworks) are consistent across
every section; single hyphens only; the section symbol for codified references; no
raster images; white background; Burgundy `#800020` document color.

## Compile (Overleaf, pdfLaTeX)

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## License

Released under CC BY 4.0. Author: Kevin Kawchak, CEO ChemicalQDevice
([ORCID 0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)).
