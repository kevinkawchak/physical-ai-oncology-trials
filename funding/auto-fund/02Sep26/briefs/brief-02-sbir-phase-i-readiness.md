# SBIR Phase I Readiness: What Exists Now, and What Nine Months Would Buy

**ChemicalQDevice, San Diego.** Kevin Kawchak, CEO.
Prepared for SBIR program technical staff. Independent work, not medical or
regulatory advice, and not endorsed by the FDA, NIH, HHS, an IRB, ICH, or any
sponsor.

---

## The ask, unchanged

| Phase | Total cost | Duration | Milestones |
|:--|:--|:--|:--|
| Phase I | $306,000 | 9 months | 5 |
| Phase II | $1,300,000 | 24 months | 7 |
| Both | $1,606,000 | 33 months | 12 |

The five-year program this Phase I feeds is $700,000 per year in direct cost,
$3,500,000 over five years. The SBIR route therefore buys $1,396,000 of direct
work inside the award, and $2,104,000 of the program's direct work sits outside
it. That gap is stated rather than hidden, and it is what the private position in
the capitalization plan is for.

## What already exists, before any award

A Phase I that begins by writing documents that already exist is a Phase I that
spends its first quarter on nothing. None of the following would be produced with
award funds, because all of it is deposited and public.

| Artifact | Status | Deposit |
|:--|:--|:--|
| Phase 1 protocol, combined IND and IDE | Complete | [10.5281/zenodo.20780121](https://doi.org/10.5281/zenodo.20780121) |
| Investigational new drug application, Phase 1 PDAC | Complete | [10.5281/zenodo.21097442](https://doi.org/10.5281/zenodo.21097442) |
| Phase 2 randomized protocol | Complete | [10.5281/zenodo.20807027](https://doi.org/10.5281/zenodo.20807027) |
| Ten-arm QSP simulation with verification notebooks | Complete | [10.5281/zenodo.17001137](https://doi.org/10.5281/zenodo.17001137) |
| Capitalization and milestone plan | Complete | [10.5281/zenodo.21887807](https://doi.org/10.5281/zenodo.21887807) |
| Site documentation package, fifteen documents | Complete | [10.5281/zenodo.22216519](https://doi.org/10.5281/zenodo.22216519) |

Six deposited artifacts, all dated before any award, all citable by a reviewer
without contacting the applicant.

## What nine months would actually buy

Five milestones, each with an artifact a program officer can open. None of them
is a document that already exists.

| # | Milestone | Artifact at completion |
|:--|:--|:--|
| 1 | Verification harness for the advisory layer, executable | A test suite and its run log, with pass and fail counts per assertion |
| 2 | Interface-enforced authority boundary, demonstrated | A network capture showing no write path from the model process to data capture or robot control |
| 3 | Bench and simulation evidence of surgeon-retained control | A recorded session set with per-motion approval events timestamped |
| 4 | Stop-latency measurement at both tiers | Measured distributions against the 3 ms arm-level and 500 ms system-wide specifications |
| 5 | Pre-submission package for the combination-product route | A Pre-Request for Designation ready to file, with each component described separately |

Milestone 2 is the load-bearing one. A governance claim that rests on a policy
document is worth less than a governance claim that rests on the absence of a
credential, and the difference is testable.

## Who does the work

The eleven-role site roster sums to 3.95 award-funded full-time equivalents
against the five-year program. A nine-month Phase I does not staff eleven roles.
It staffs four.

| Role | Phase I FTE | What this role is for in Phase I |
|:--|:--|:--|
| Chief executive and sponsor representative | 0.35 | Regulatory strategy, the developer relationship, the FDA route |
| Robotics and physical AI systems engineer | 0.60 | Milestones 2, 3 and 4 |
| LLM verification and model governance lead | 0.55 | Milestones 1 and 2 |
| Regulatory affairs and quality manager | 0.25 | Milestone 5 and the quality record for milestones 1 to 4 |

1.75 full-time equivalents. The other seven roles begin at site activation, which
is a Phase II event and is budgeted there.

## The cost argument, stated once

The company's comparable virtual trial work is **projected** at $36,330 per run
against an industry benchmark above $120,000 per run, with a time to decision of
about one month against 4.5 months. Those figures describe simulation work
already completed and deposited; they are not a projection of what a clinical
Phase I costs, and no part of this brief presents them as one.

## What this brief does not claim

No institution is a partner, sponsor, site, or endorser, and no agreement of any
kind exists. No drug supply agreement, letter of authorization, or regulatory
cross-reference is in place with the agent's developer. No robotic configuration
is specified. Daraxonrasib is approved in metastatic disease and is nowhere
described as first in human; the perioperative use proposed here remains
investigational and still requires an investigational new drug application.

---

**Sources.** `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed`,
`funding/capitalization-plan/final-capital` §3 and §5,
`funding/move-in/final-move-in/sections/sec-14-staffing-and-roles.tex`.
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).

*Disclaimer: This work is independent and is not endorsed or sponsored by any
trial sponsor, CRO, site, IRB, regulator, or medical society; and was adapted
using Claude Code Opus 5.*
