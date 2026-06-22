# inputs - source documents for the Phase 2 protocol (v1.1.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Main documents](https://img.shields.io/badge/Main%20documents-4-800020.svg)](.)
[![Predicate](https://img.shields.io/badge/Predicate-Phase%201%20v1.0.0-6B6B6B.svg)](../../trial-protocol)
[![Sources](https://img.shields.io/badge/Sources-canonical%20in%20trial--protocol-6B6B6B.svg)](../../trial-protocol/inputs)

This directory documents the source materials the Phase II protocol builds on. The
large source files are canonical in the Phase I tree under
[`../../trial-protocol/inputs`](../../trial-protocol/inputs) and are referenced
here rather than duplicated, so there is a single source of truth.

## Main documents (and where they are used)

| # | Document | Canonical location | Used in Phase II for |
|:--|:--|:--|:--|
| 1 | Phase 1 predicate protocol | [`../../trial-protocol/final-protocol/publication`](../../trial-protocol/final-protocol/publication) | the RP2D (300 mg once daily) and the device feasibility and safety that establish equipoise for this randomized efficacy study |
| 2 | 2030 60-second PDAC robotic Whipple + daraxonrasib simulation | [`../../trial-protocol/inputs/2030-pdac-1min-final-paper`](../../trial-protocol/inputs/2030-pdac-1min-final-paper) | the clinical subject, the platform, and the quantitative telemetry, force, vessel, and advisory data |
| 3 | Adaption: 21 CFR Part 312 (Physical AI overlay) | [`../../trial-protocol/inputs/21cfr312_adapt`](../../trial-protocol/inputs/21cfr312_adapt) | the Subpart J overlay, the Phase 0 gate, the USL readiness rating, the consent opt-out, and the audit-trail and AE-reporting requirements |
| 4 | H. R. 9510 bill (Verification Before Generation, financial data amendment) | [`../../trial-protocol/inputs/auto-bill-02`](../../trial-protocol/inputs/auto-bill-02) | the VVUQ ten-gate framing and the co-investment financial-data standard behind the capital firewall |

## Author works (LLM-trust record, August 2024 to June 2026)

The author works that evidence established LLM trust for oncology trials are
collected in [`../../trial-protocol/inputs/author_works.bib`](../../trial-protocol/inputs/author_works.bib);
the directly relevant subset, the five daraxonrasib program works, and the Phase 1
predicate are carried into the Phase II bibliography at
[`../final-protocol/publication/references.bib`](../final-protocol/publication/references.bib).

## What changed from Phase 1 inputs

The Phase II build adds the **Phase 1 protocol itself** as the predicate document
(it did not exist when Phase 1 was authored) and adds randomized-trial and
reporting-method references (CONSORT, group-sequential methods, the EORTC
quality-of-life instrument, and circulating-tumor-DNA evidence) to support the
confirmatory design. All daraxonrasib citations are carried without searching, as
in Phase 1.

## License

Released under CC BY 4.0. Reproduced regulatory text is U.S. Government work under
17 U.S.C. 105. Author: Kevin Kawchak, CEO ChemicalQDevice.
