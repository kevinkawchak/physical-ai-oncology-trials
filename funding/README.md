# funding - ChemicalQDevice PDAC Trial Funding (repository v4.7.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Applications](https://img.shields.io/badge/Applications-12%20total-00417A.svg)](pdac-funding-applications/applications)
[![Policy basis](https://img.shields.io/badge/Policy-Science%3A%20A%20New%20Golden%20Age-3C7DB2.svg)](science-golden-age)
[![Partner of choice](https://img.shields.io/badge/Partner-UC%20San%20Diego%20Moores-6C757D.svg)](potential-partners/UC-San-Diego)
[![Repository](https://img.shields.io/badge/Repository-v4.7.0-6C757D.svg)](../README.md)
[![Capitalization plan](https://img.shields.io/badge/v4.5.0-Capitalization%20Plan-00417A.svg)](capitalization-plan)
[![La Jolla Move-In](https://img.shields.io/badge/v4.7.0-La%20Jolla%20Move--In-00417A.svg)](move-in)
[![Site documents](https://img.shields.io/badge/Site%20documents-15-3C7DB2.svg)](move-in/final-move-in)
[![DOI v2.0](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21317266-blue.svg)](https://doi.org/10.5281/zenodo.21317266)
[![DOI v1.0](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21232965-blue.svg)](https://doi.org/10.5281/zenodo.21232965)

Everything the *Daraxonrasib Phase 1 LLM-Directed Robotic Whipple in
KRAS-Mutated PDAC* programme uses to ask for money: two completed NIH
applications with DOIs, ten independent-scientist application file sets, the
company-conversion capitalization plan that rewrites the one of those ten
addressed to a company mechanism, the fifteen-document La Jolla move-in package
that spends the award once it is won, the White House policy corpus they are all
written against, the partner-site research, and the supplementary source sets
that supply their quantitative evidence.

---

## 1. Directory structure

```
funding/
  README.md                       this hub
  RFA-RM-27-001/                  Application v1.0: LaTeX source zip + README
  RFA-RM-27-001-v2/               Application v2.0: LaTeX source zip + README
  pdfs/                           Compiled PDFs of both applications
  science-golden-age/             The July 2026 White House report, chunked
  supplementary/                  Founding documents, two compiled PDFs
    source-files/                 Three LaTeX source zips
  potential-partners/             Site research
    UC-San-Diego/                 Partner of choice: overview + priority steps
    Scripps/                      Alternate site: overview + priority steps
  pdac-funding-applications/      v4.4.0: ten application file sets + summary paper
    prompts/  sub-prompts/  applications/
    mermaid/  plantuml/  d2/  diagrams-python/  graphviz/
    draft-apply/  full-apply/  final-apply/
  capitalization-plan/            v4.5.0: the ChemicalQDevice conversion paper
    prompts/  sub-prompts/
    mermaid/  plantuml/  d2/  diagrams-python/  graphviz/
    draft-capital/  full-capital/  final-capital/
  move-in/                        v4.7.0: the La Jolla site documentation package
    inputs/  inputs/READMES/      the three source artifacts and their READMEs
    prompts/  sub-prompts/        prompt of record; three stage schedules
    draft-move-in/  full-move-in/  final-move-in/
  daraxonrasib-llm-story.md       June 2025 to July 2026 drug chronology
  tripartisan-llm-support.md      The three frontier-model roles
```

## 2. Completed applications with DOIs

**7/12: (Clinical Trial Funding Application v2.0)** *RFA-RM-27-001, Kawchak K.
The application proposes a first-in-human, combined drug-device investigation of
perioperative daraxonrasib and an eight-arm robotic pancreaticoduodenectomy,
with an on-premises, repository-pinned LLM acting only as a second-opinion
advisory system.*
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21317266-blue)](https://doi.org/10.5281/zenodo.21317266)
Kawchak, K. (2026). Clinical Trial Funding Application v2.0, RFA-RM-27-001,
Kawchak K. Zenodo. https://doi.org/10.5281/zenodo.21317266

**7/7: (Clinical Trial Funding Application)** *RFA-RM-27-001, Kawchak K.
ChemicalQDevice respectfully submits this application to the NIH Director's
Pioneer Award for "Daraxonrasib Phase 1 LLM-Directed Robotic Whipple in
KRAS-Mutated PDAC."*
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21232965-blue)](https://doi.org/10.5281/zenodo.21232965)
Kawchak, K. (2026). Clinical Trial Funding Application, RFA-RM-27-001,
Kawchak K. Zenodo. https://doi.org/10.5281/zenodo.21232965

## 3. New in v4.7.0: the La Jolla move-in package

[`move-in/`](move-in) is what the programme does with the money once an award
lands. Fifteen documents stand up California's first pancreatic ductal
adenocarcinoma large language model oncology clinical trial site, in La Jolla,
for a chief executive and ten coworkers, against the same $700,000 per year and
$3,500,000 over five years frame the ten applications ask for. The frame is
reused verbatim from
[`pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex`](pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex)
and is not re-derived.

| Part | Documents | What they carry |
|:--|:--|:--|
| I | 01 SB 1188, 02 AB 3162, 03 SB 964, 04 H. R. 10412 | Site authorization, participant rights and robotic safety, model transparency and data protection, and the federal demonstration act succeeding H. R. 9510 |
| II | 05 San Diego Municipal Code, 06 Title 22 Chapter 15, 07 FDA compliance guide | Entitlement, the department's survey machinery, and a fourteen-row map from federal law to a document in the package |
| III | 08 Building code, 09 Premises code, 10 Parking, 11 Emergency plan | What is built, who may be where, how a participant arrives and leaves, and what happens when something fails |
| IV | 12 Activation and SOPs, 13 Conventional trial requirements, 14 Staffing, 15 Funding and lobbying | Five gates, a manual that stands without the model, eleven people, and the separation of federal from non-federal funds |

| Stage | Directory | Compile |
|:--|:--|:--|
| 1 | [`move-in/draft-move-in/`](move-in/draft-move-in) | 0 errors, 27 pages |
| 2 | [`move-in/full-move-in/`](move-in/full-move-in) | 0 errors, 71 pages |
| 3 | [`move-in/final-move-in/`](move-in/final-move-in) | 0 errors, 67 pages, 0 short pages |

## 4. New in v4.4.0: ten independent-scientist application file sets

[`pdac-funding-applications/`](pdac-funding-applications) adds ten complete
funding application email file sets, each unique to its recipient, each written
in Kevin Kawchak's name as an **independent scientist** under the funding
approach set out in *Science: A New Golden Age*, and each naming **UC San Diego
Moores Cancer Center** as the intended partner. Applications 01 to 05 lead with
the operation; 06 to 10 lead with the drug and patient selection. Both sets
describe the same hybrid procedure. **PART I carries no DOIs**; the DOIs that
appear in it are citations of prior published work.

| # | Recipient | Perspective | Ask |
|:--|:--|:--|:--|
| 01 | NIH Common Fund, Director's Pioneer Award | Surgical | $700K per year, 5 years |
| 02 | ARPA-H mission office | Surgical | $2.1M over 36 months, three gates |
| 03 | NSF TIP Directorate, X-Labs | Surgical | $700K per year, 5 years |
| 04 | DOE Office of Science, Genesis Mission | Surgical | $700K per year, 5 years |
| 05 | NIH SEED, SBIR/STTR | Surgical | $306K Phase I, $1.3M Phase II |
| 06 | Foundation for the NIH, AMP | Medical oncology | $3.5M cash, rest contributed |
| 07 | HHMI Investigator Program | Medical oncology | Person-based, 7-year horizon |
| 08 | NCI Cancer Therapy Evaluation Program | Medical oncology | $700K per year, 5 years |
| 09 | Convergent Research, FRO programme | Medical oncology | $3.5M, dissolves in year 5 |
| 10 | UC San Diego Moores Cancer Center | Medical oncology | A 45-minute feasibility meeting |

## 5. What each directory supplies to the others (Rule 5)

| Directory | Supplies | Consumed by |
|:--|:--|:--|
| [`science-golden-age/`](science-golden-age) | Ten verbatim chunks of the July 2026 report plus 186 BibTeX entries | Every application's §1 and §2; the summary paper's §1 and §2 |
| [`RFA-RM-27-001-v2/`](RFA-RM-27-001-v2) | Trial synopsis, endpoint wording, budget frame, 58 bibliography entries | Every application's §3 to §5; the summary paper's §5 |
| [`RFA-RM-27-001/`](RFA-RM-27-001) | The first application, as a time point | Programme chronology |
| [`pdfs/`](pdfs) | Compiled PDFs of both applications | Manual-attachment lists in the ten emails |
| [`supplementary/`](supplementary) | Founding-document ledger and two compiled PDFs | Prior-work tables and attachment lists |
| [`supplementary/source-files/`](supplementary/source-files) | Three LaTeX source zips: patient-robot-advocacy, daraxonrasib simulations, the original proposal | Palette and the five diagram vocabularies; all quantitative evidence; the first proposal time point |
| [`potential-partners/UC-San-Diego/`](potential-partners/UC-San-Diego) | Partnership sequence, required positioning, priority steps, named contacts | §5 of every application; application 10 in full |
| [`potential-partners/Scripps/`](potential-partners/Scripps) | The alternate-site comparison | The second-site branch in application 03 |
| [`daraxonrasib-llm-story.md`](daraxonrasib-llm-story.md) | The June 2025 to July 2026 chronology and the three stated differences from RASolute 302 | §3 of every application |
| [`tripartisan-llm-support.md`](tripartisan-llm-support.md) | The three frontier-model roles | Governance sections and the summary paper's §6 |
| [`pdac-funding-applications/`](pdac-funding-applications) | Ten application file sets and the summary paper | The v4.4.0 release; the `$700,000` budget frame and the four-source evidence table reused by `move-in/` |
| [`capitalization-plan/`](capitalization-plan) | The company conversion, the 21 CFR part 54 firewall wording, the three cost benchmark rows and the projected `$36,330` figure | The v4.5.0 release; documents 13, 14 and 15 of `move-in/` |
| [`move-in/`](move-in) | Fifteen site documents, eleven roles, and the six-line split of the annual `$700,000` | The v4.7.0 release |

## 6. Positioning constraints that apply to everything here

- Daraxonrasib is **not** first-in-human. It is investigational and already in
  Phase 3 evaluation. The supportable claim is the first prospective clinical
  evaluation of the integrated surgical and LLM advisory workflow, subject to
  FDA and institutional confirmation.
- The robotic configuration is specified by manufacturer, model, cleared
  configuration, arm count, instruments, and software version at the site
  agreement, not asserted in advance.
- No drug supply, letter of authorization, or regulatory cross-reference is in
  place with the agent's developer.
- UC San Diego and Moores Cancer Center are named as the intended partner of
  choice and nothing more. Neither is described as a partner, sponsor, trial
  site, or endorser, and neither will be without written authorization.
- The AI system is bounded and advisory throughout. Licensed clinicians retain
  final authority over diagnosis, treatment, surgery, and safety decisions.
- Simulation results, draft protocol concepts, unvalidated software, proposed
  clinical research, and established clinical evidence are labelled separately
  wherever they appear.

## 7. License

Creative Commons Attribution 4.0 International (CC BY 4.0).
