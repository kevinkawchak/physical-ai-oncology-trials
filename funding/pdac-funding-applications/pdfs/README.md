# pdac-funding-applications - 10 Independent-Scientist Applications + Summary Paper (v4.4.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Applications](https://img.shields.io/badge/Applications-10%20email%20file%20sets-00417A.svg)](applications)
[![Policy basis](https://img.shields.io/badge/Policy-Science%3A%20A%20New%20Golden%20Age-3C7DB2.svg)](../science-golden-age)
[![Mechanism](https://img.shields.io/badge/Mechanism-Independent%20Scientist%20%2F%20%24200B-6C757D.svg)](../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md)
[![Partner](https://img.shields.io/badge/Partner%20of%20choice-UC%20San%20Diego%20Moores-6C757D.svg)](../potential-partners/UC-San-Diego)
[![Figures](https://img.shields.io/badge/Summary%20paper%20figures-20-00417A.svg)](final-apply)
[![Method](https://img.shields.io/badge/Method-5%20diagram%20stages%20%E2%86%92%20draft%20%E2%86%92%20full%20%E2%86%92%20final-6C757D.svg)](sub-prompts)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--5457--8667-6C757D.svg)](https://orcid.org/0009-0007-5457-8667)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-6C757D.svg)](../../README.md)
[![Summary paper DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21787424-blue.svg)](https://doi.org/10.5281/zenodo.21787424)

This directory builds two deliverables from one master prompt.

- **PART I.** Ten complete, recipient-unique **Phase 1 pancreatic cancer trial
  funding application email file sets** (no DOIs), each written in Kevin
  Kawchak's name as an **independent scientist** under the funding approach set
  out in the White House report *Science: A New Golden Age*, and each stating
  the intent to partner at **UC San Diego Moores Cancer Center**. Five are
  written from a surgical perspective and five from a medical oncology
  perspective; both sets describe the same hybrid procedure, which carries
  surgical and medical oncology arms together.
- **PART II.** One **summary paper** (one DOI) describing the ten applications,
  built through the eight-stage sub-prompt schedule below, at approximately one
  quarter of the character count of the parent source set.

Nothing here is a submission of record. Every application is a draft the author
compiles, verifies, and sends; every recipient address must be confirmed against
the funder's current published contact page before use.

---

## 1. Build pipeline

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px','lineColor':'#6C757D'}}}%%
flowchart TB
    MP["Master prompt<br/>prompts/prompt-apply.md"]:::goal
    PA["Process A<br/>write Part I + Part II sub-prompts"]:::proc
    P1["PART I<br/>10 application file sets"]:::accent
    D1["Stage 1 mermaid"]:::input
    D2["Stage 2 plantuml"]:::input
    D3["Stage 3 d2"]:::input
    D4["Stage 4 diagrams-python"]:::input
    D5["Stage 5 graphviz"]:::input
    S6["Stage 6 draft-apply"]:::soft
    S7["Stage 7 full-apply"]:::soft
    S8["Stage 8 final-apply"]:::accent
    REL["Release v4.4.0<br/>README + CHANGELOG + releases"]:::proc
    MP --> PA
    PA --> P1
    PA --> D1 --> D2 --> D3 --> D4 --> D5 --> S6 --> S7 --> S8 --> REL
    P1 --> REL
    classDef goal fill:#00417A,stroke:#00417A,stroke-width:1.5px,color:#FFFFFF
    classDef proc fill:#6C757D,stroke:#00417A,stroke-width:1px,color:#FFFFFF
    classDef accent fill:#3C7DB2,stroke:#00417A,stroke-width:1px,color:#FFFFFF
    classDef soft fill:#DCE8F1,stroke:#3C7DB2,stroke-width:1px,color:#00417A
    classDef input fill:#E9ECEF,stroke:#6C757D,stroke-width:1px,color:#000000
```
