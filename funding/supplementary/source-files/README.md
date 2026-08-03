# source-files - the three LaTeX source sets v4.4.0 is built from

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Zips](https://img.shields.io/badge/Source%20zips-3-00417A.svg)](.)
[![Palette source](https://img.shields.io/badge/Palette%20source-patient--robot--advocacy-3C7DB2.svg)](.)
[![Vocabularies](https://img.shields.io/badge/Diagram%20vocabularies-5-6C757D.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../../README.md)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21720120-blue.svg)](https://doi.org/10.5281/zenodo.21720120)

Three LaTeX source sets. Everything in
[`../../pdac-funding-applications/`](../../pdac-funding-applications) is built
from them, and this README records exactly what each one supplies.

## 1. `patient-robot-advocacy.zip`

The parent publication, *Patient Robot Advocacy: A Phase 1, First-in-Human,
PDAC Clinical Trial Protocol of a LLM-Directed Robotic Whipple with Daraxonrasib
(RMC-6236)*, Draft 1.0, thirty figures, thirteen sections,
[10.5281/zenodo.21720120](https://doi.org/10.5281/zenodo.21720120).

| Content | What v4.4.0 takes |
|:--|:--|
| `patientstyle.sty` palette | The eight colour tokens: `protoblue` #00417A, `protogray` #6C757D, white, `pagrayl` #E9ECEF, `pagraym` #CED4DA, `pagrayd` #9AA1A8, `pablue1` #3C7DB2, `pablue2` #DCE8F1. The parent's ninth token, a near-black fill, is **dropped**: the v4.4.0 master prompt forbids black filled boxes |
| The five TikZ diagram vocabularies | `mm*` mermaid-type, `uml*` plantuml-type, `d2*` d2-type, `dg*` diagrams-python-type, `gv*` graphviz-type, all carried into `appstyle.sty` and `applystyle.sty` |
| The figure spacing invariant | `\end{...fig}` then `\vspace{-0.7cm}` then `\figcaption`, with rigid skips giving an identical frame-to-caption distance everywhere |
| Table column convention | Every fixed-width column `>{\raggedright\arraybackslash}p{...}`; every table exactly `\textwidth` |
| Senior-author formatting | RaggedRight body, widow and club penalties at 10000, stretchable `\parfillskip`, `\UrlBreaks` on every character, float carriage so no page is stranded behind a figure |
| Cover and back-matter furniture | The idiom, not the layout: v4.4.0 defines ten new cover variants and its own back-matter block |
| Character count | 301,310 characters, the basis for the summary paper's one-quarter target of roughly 75,000 |

## 2. `Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`

The simulation source set. Supplies every quantitative claim in the ten
applications and in the summary paper's §5.

| Result | Value | Used in |
|:--|:--|:--|
| QSP simulation, 10 arms, 250 ODEs per patient | mOS 12.8 vs 5.4 months; best arm HR 0.25; 6 of 7 archetypes below HR 1.00 | §3 of all ten applications |
| Digital twin, 1000 patients | Best mPFS HR 0.31; mOS HR 0.34; credibility 81.9 under ASME V&V 40 and FDA M15; 55 verification notebook tests | §3 of applications 01, 02, 04, 06, 07, 08, 10 |
| Empirical 100,000-patient triplicate | 5-arm consistency 8.95 to 9.45; overall reproducibility 8.65; G3+ AE 8.0% vs 25.0% | Summary paper §5 |
| Cost benchmarks | Industry: >$120,000 per empirical run, >$2M per QSP trial, $28K to $700K per digital twin. Author: $36,330 estimated per run, about one month | §3 of applications 01, 02, 05, 07; the productivity argument throughout |
| Stated limitations | No acquired resistance, ideal pharmacodynamics, no patient-specific PK or tumour growth parameters | Carried into the evidence tables of applications 08 and 10, in the same row as the result |

## 3. `Physical-AI-Oncology-Trial-Competition-Proposal.zip`

The original January 13, 2026 proposal: an end-to-end hybrid edge-cloud AI
architecture. Used as the **first released proposal time point** in the
programme chronology, which is what makes the June 2025 to August 2026 sequence
verifiable rather than asserted.

## Rule 3 note

No raster output is derived from any of these sources. Every figure in v4.4.0 is
pure TikZ, written new, and no diagram is copied from a prior author work.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
