# Six Checkable Quantities

**ChemicalQDevice, San Diego.** Kevin Kawchak, CEO.
No argument is made in this brief. Every row is a published or deposited
quantity, its comparator, its evidence tier, and the limitation its own authors
stated. Independent work, not medical or regulatory advice, and not endorsed by
the FDA, NIH, HHS, an IRB, ICH, or any sponsor.

---

| Source | Test arm | Comparator | Tier | The limitation its own authors gave it |
|:--|:--|:--|:--|:--|
| RASolute 302, May 2026 | 13.2 mo median OS | 6.6 mo median OS | Trial | Metastatic and previously treated; silent on the resectable setting |
| Ten-arm QSP simulation, 250 ODEs, August 2025 | 12.8 mo median OS, HR 0.25 | 5.4 mo median OS | In silico | Assumes no acquired resistance and ideal pharmacodynamics; grade 3 and above high in all arms |
| Digital twin, 1000 patients | 12.1 mo median OS | Not applicable | Twin | No patient-specific PK, PD, or tumor growth parameters; simple Emax model |
| Digital twin, progression-free survival | HR 0.31 | Not applicable | Twin | Same source and same limitation; no immune compartments |
| VVUQ credibility, 55 tests | Score 81.9 | ASME V and V 40 gate | Twin | A pre-trial credibility score, not a post-trial validation |
| Empirical triplicate, 100,000 records | Grade 3 and above, 8.0 percent | 25.0 percent | In silico | The G12C log favors experimental while the KRAS-mutant report favors control |

## The one comparison worth making, with its three disqualifiers

The August 2025 simulation returned a 2.4-fold median overall survival ratio. The
May 2026 RASolute 302 readout returned 2.0-fold. Nine months separate them and
the simulation came first.

That is a chronology observation and a hypothesis-supporting one. **It is not a
validation claim.** Three differences are material:

- Sample size: 1000 simulated against 241 enrolled.
- Intervention: a daraxonrasib combination against daraxonrasib as a single agent.
- Molecular selection: KRAS G12C against a primarily G12D and G12V population.

## Deposits

| Quantity | Deposit |
|:--|:--|
| RASolute 302 | [10.1056/NEJMoa2605555](https://doi.org/10.1056/NEJMoa2605555) |
| QSP simulation | [10.5281/zenodo.17001137](https://doi.org/10.5281/zenodo.17001137) |
| Digital twin, both rows, and the VVUQ score | [10.5281/zenodo.20780121](https://doi.org/10.5281/zenodo.20780121) |
| Empirical triplicate | [10.5281/zenodo.15735068](https://doi.org/10.5281/zenodo.15735068) |
| Drug identification, June 2025 | [10.5281/zenodo.15735068](https://doi.org/10.5281/zenodo.15735068) |
| FDA approval, August 26, 2026 | [FDA press announcement](https://www.fda.gov/news-events/press-announcements/fda-approves-first-class-targeted-therapy-metastatic-pancreatic-cancer) |

## Cost, for completeness

The company's comparable virtual trial work is **projected** at $36,330 per run
against an industry benchmark above $120,000 per run. That figure describes
simulation work already completed and deposited and is not a projection of
clinical trial cost.

---

**Source.** Every row is reused without re-derivation from
`funding/capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`.
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).

*Disclaimer: This work is independent and is not endorsed or sponsored by any
trial sponsor, CRO, site, IRB, regulator, or medical society; and was adapted
using Claude Code Opus 5.*
