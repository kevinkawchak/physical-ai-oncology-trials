# applications - PART I: ten independent-scientist funding application file sets (v4.4.0)

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0/)
[![File sets](https://img.shields.io/badge/File%20sets-10-00417A.svg)](.)
[![Set A](https://img.shields.io/badge/Set%20A-surgical%20perspective-3C7DB2.svg)](.)
[![Set B](https://img.shields.io/badge/Set%20B-medical%20oncology%20perspective-6C757D.svg)](.)
[![Dated](https://img.shields.io/badge/Dated-August%203%2C%202026-6C757D.svg)](.)
[![Pages](https://img.shields.io/badge/Attachment-%E2%89%A45%20pages%20each-6C757D.svg)](.)
[![DOIs](https://img.shields.io/badge/DOIs-none%20in%20PART%20I-9AA1A8.svg)](.)
[![Repository](https://img.shields.io/badge/Repository-v4.4.0-blue.svg)](../../../README.md)

Ten complete funding application email file sets. Each is unique to its
recipient, written in **Kevin Kawchak's** name as an **independent scientist**
under the funding approach set out in the White House report *Science: A New
Golden Age*, and each states the intent to partner at **UC San Diego Moores
Cancer Center**. Every set is dated **August 3, 2026**. PART I carries **no
DOIs of its own**; the DOIs that appear are citations of prior published work.

---

## 1. The argument every application makes

The report's transmittal letter states that the U.S. research system should
**prioritize the individual scientist over legacy institutions**, and its
Chapter II states that federal agencies distribute **approximately $200 billion
in annual R&D funding** with "no systematic framework for identifying where
those dollars could catalyze the greatest scientific returns, with deference
instead given to the same incumbents that consume the funding."

Each application answers that in the same shape: a single scientist, working
from one public repository, has already produced the document package that a
legacy institution's program office produces with a team, and asks for the
portion of the realigned portfolio needed to take it into a first-in-human
Phase 1 trial at a partner site.

## 2. The ten recipients

| # | Directory | Recipient program | Perspective | *Golden Age* anchor |
|:--|:--|:--|:--|:--|
| 01 | [`app-01-nih-pioneer-award`](app-01-nih-pioneer-award) | NIH Common Fund, Director's Pioneer Award | Surgical | Summary of the Report: "scaling long-horizon grants for the best and brightest modeled on National Institutes of Health (NIH) Director's Pioneer Award" |
| 02 | [`app-02-arpa-h`](app-02-arpa-h) | ARPA-H mission office | Surgical | Chapter II: the ARPA program-manager model, ARPA-H and ARPA-E |
| 03 | [`app-03-nsf-tip-x-labs`](app-03-nsf-tip-x-labs) | NSF TIP Directorate, X-Labs | Surgical | Chapter II: X-Labs, "the first federal program designed to fund independent research organizations outside traditional academia" |
| 04 | [`app-04-doe-genesis-mission`](app-04-doe-genesis-mission) | DOE Office of Science, Genesis Mission | Surgical | Chapter V and the Annex: EO 14363 and the Robotics mission, "to initiate the era of physical AI-driven scientific discovery" |
| 05 | [`app-05-nih-sbir-seed`](app-05-nih-sbir-seed) | NIH SEED, SBIR/STTR | Surgical | Chapter IV: "Programs like SBIR open doors for technician-founded ventures" |
| 06 | [`app-06-fnih-amp`](app-06-fnih-amp) | Foundation for the NIH, AMP | Medical oncology | Chapter III: FNIH and the Accelerating Medicines Partnership held up as the model |
| 07 | [`app-07-hhmi-investigator`](app-07-hhmi-investigator) | HHMI Investigator Program | Medical oncology | Chapter II: person-based funding of roughly $10 million over seven years |
| 08 | [`app-08-nci-ctep`](app-08-nci-ctep) | NCI Cancer Therapy Evaluation Program | Medical oncology | Annex: prioritized biological sciences; Chapter I's cancer framing |
| 09 | [`app-09-convergent-fro`](app-09-convergent-fro) | Convergent Research, FRO program | Medical oncology | Chapter II: focused research organizations as time-bound nonprofit research startups |
| 10 | [`app-10-ucsd-moores-engine`](app-10-ucsd-moores-engine) | UC San Diego Moores Cancer Center | Medical oncology | Chapter IV and the Annex: regional innovation clusters and non-federal cost share |

Set A leads with the operation and Set B leads with the drug, but **both sets
describe the same hybrid procedure**, which carries surgical and medical
oncology arms together: an eight-arm robotic pancreaticoduodenectomy with
perioperative daraxonrasib (RMC-6236) and an on-premises, advisory-only LLM.

## 3. What each directory contains

```
app-NN-<slug>/
  README.md                    what this application asks for, and from what
  email-app-NN-<slug>.txt      recipients, subject, body, closing, attachments
  main.tex                     the compiled attachment, at most five pages
  appstyle.sty                 the shared style (one self-contained copy)
  references.bib               the shared bibliography (one self-contained copy)
  sections/                    one .tex per section of the attachment
  app-NN-<slug>-LaTeX.zip      Overleaf-ready bundle of the four items above
```

Compile with `pdflatex -> bibtex -> pdflatex -> pdflatex`.

## 4. Cover variants

The ten attachments use ten different cover treatments, all defined in
`appstyle.sty`, so no two read as copies of one template. All ten vary in
appearance from the centred form-field theme of
[`../../RFA-RM-27-001-v2`](../../RFA-RM-27-001-v2).

| # | Macro | Treatment |
|:--|:--|:--|
| 01 | `\appbanner` | Full-width Corporate Blue banner with a badge strip |
| 02 | `\appledger` | Three-column milestone ledger above the title |
| 03 | `\appaccentblock` | Left accent bar beside a rule block |
| 04 | `\appmissiontile` | National-mission tile above the title |
| 05 | `\apptwopanel` | Technical objective beside the commercial case |
| 06 | `\appconsortium` | Two-party consortium strip |
| 07 | `\appperson` | Investigator block above a compact title rule |
| 08 | `\apprecord` | Four-field study-registration header |
| 09 | `\apptimeline` | Five-year dissolution timeline strip |
| 10 | `\approuting` | Institutional intake routing block |

## 5. Style contract (`appstyle.sty`)

| Rule | Implementation |
|:--|:--|
| Palette | `protoblue` #00417A, `protogray` #6C757D, white, `pagrayl` #E9ECEF, `pagraym` #CED4DA, `pagrayd` #9AA1A8, `pablue1` #3C7DB2, `pablue2` #DCE8F1 |
| No black fill | The parent style's near-black fill token is deleted; black survives as stroke and text only. Audit with `grep -c padark`, which must return 0 |
| Figure spacing | `\end{appfig}` then `\vspace{-0.7cm}` then `\figcaption`; rigid 26pt close and `\nointerlineskip` open give an identical 6.1pt frame-to-caption distance everywhere |
| Captions | Centred italic, at most three lines, lines balanced to a similar character count |
| Tables | Exactly `\textwidth`; every fixed column `>{\raggedright\arraybackslash}p{...}`; Corporate Blue header row |
| Body | `\RaggedRight` with `\RaggedRightRightskip=0pt plus 2em`; widow, club and broken penalties at 10000; stretchable `\parfillskip` so no paragraph ends in one or two words |
| Links | `\UrlBreaks` on every character, re-asserted after `url` and `hyperref`; `\dlink` prints a DOI as text with a clickable target |
| Symbols | Single dashes only; `\S` for every codified section reference |
| Diagrams | Five TikZ vocabularies: `mm*` mermaid, `uml*` plantuml, `d2*` d2, `dg*` diagrams-python, `gv*` graphviz |

## 6. Figure budget

At most **five figures per application**, chosen by what the figure has to
answer rather than by an equal quota across the five platforms.

| # | Figures | Types used |
|:--|:--|:--|
| 01 | 3 | mermaid, d2, graphviz |
| 02 | 4 | mermaid, d2, graphviz, plantuml |
| 03 | 3 | d2, diagrams-python, mermaid |
| 04 | 4 | diagrams-python, mermaid, graphviz, d2 |
| 05 | 3 | mermaid, d2, graphviz |
| 06 | 3 | d2, mermaid, graphviz |
| 07 | 3 | mermaid, d2, plantuml |
| 08 | 4 | mermaid, d2, graphviz, plantuml |
| 09 | 3 | mermaid, d2, diagrams-python |
| 10 | 4 | mermaid, d2, plantuml, graphviz |

## 7. Files used from other directories (Rule 5)

| Source file or directory | Where it is used |
|:--|:--|
| [`../../science-golden-age/chunk-01`](../../science-golden-age/chunk-01-front-matter-and-summary.md) | §1 of every application: the individual-scientist goal and the Pioneer Award sentence |
| [`../../science-golden-age/chunk-03`](../../science-golden-age/chunk-03-chapter-two-revitalizing-science-and-technology-enterprise.md) | §1 and §2: the $200 billion portfolio, the incumbency tax, mid-scale science, X-Labs, FROs, HHMI person funding |
| [`../../science-golden-age/chunk-04`](../../science-golden-age/chunk-04-chapter-three-securing-dominance-in-critical-and-emerging-technologies.md) | §2 of 05 and 06: trial economics, FDA reform, FNIH and AMP |
| [`../../science-golden-age/chunk-05`](../../science-golden-age/chunk-05-chapter-four-science-and-technology-better-lives-of-all-americans.md) | §2 of 05 and 10: craft, technicians, regional clusters |
| [`../../science-golden-age/chunk-06`](../../science-golden-age/chunk-06-chapter-five-a-new-golden-age.md) | §2 of 04 and 09: Genesis Mission, Gold Standard Science, closed-loop experimentation |
| [`../../science-golden-age/chunk-08`](../../science-golden-age/chunk-08-annex-fiscal-year-2028-research-and-development-budget-priorities.md) | §5 of every application: long-duration grants, fast grants, cost share, the Robotics mission |
| [`../../RFA-RM-27-001-v2/LaTeX Source Files.zip`](../../RFA-RM-27-001-v2) | §3 and §5: trial synopsis (up to 18 treated participants, 3+3 design, 28-day screening, 30/90-day safety, 24-month OS) and the $700,000 per year, $3,500,000 total budget frame |
| [`../../supplementary/source-files/patient-robot-advocacy.zip`](../../supplementary/source-files) | `appstyle.sty` palette, the five diagram vocabularies, the cover and back-matter idiom, the table-column convention |
| [`../../supplementary/source-files/Daraxonrasib-Efficient-LLM-Trial-Simulations.zip`](../../supplementary/source-files) | §3 evidence tables: QSP mOS 12.8 months and HR 0.25, digital-twin mPFS HR 0.31, credibility score 81.9, 55 verification tests, LLM cost comparisons |
| [`../../supplementary/source-files/Physical-AI-Oncology-Trial-Competition-Proposal.zip`](../../supplementary/source-files) | §3 chronology: the first released proposal time point |
| [`../../supplementary/Physical AI Oncology Trial Founding Documents.md`](../../supplementary) | §3 prior-work ledger and the manual-attachment lists in every `.txt` |
| [`../../daraxonrasib-llm-story.md`](../../daraxonrasib-llm-story.md) | §3: the June 2025 to July 2026 chronology and the QSP versus RASolute 302 comparison, including the three stated differences |
| [`../../tripartisan-llm-support.md`](../../tripartisan-llm-support.md) | §4: the three frontier-model roles table |
| [`../../potential-partners/UC-San-Diego/README.md`](../../potential-partners/UC-San-Diego) | §5 of every application: the partnership sequence and the required positioning |
| [`../../potential-partners/UC-San-Diego/priority-steps.md`](../../potential-partners/UC-San-Diego/priority-steps.md) | Application 10 in full |
| [`../../pdfs/`](../../pdfs) | The manual-attachment ledger in each `.txt` |

## 8. Positioning constraint carried into every file

UC San Diego is named as the **partner of choice** and nothing more. No
application, and no email, describes UC San Diego or Moores Cancer Center as a
partner, sponsor, trial site, or endorser, because no written authorization
exists. The AI system is described as bounded and advisory throughout, with
licensed clinicians retaining final authority over diagnosis, treatment,
surgery, and safety decisions. Simulation results, draft protocol concepts,
unvalidated software, proposed clinical research, and established clinical
evidence are labelled separately wherever they appear.

## 9. License

Creative Commons Attribution 4.0 International (CC BY 4.0).
