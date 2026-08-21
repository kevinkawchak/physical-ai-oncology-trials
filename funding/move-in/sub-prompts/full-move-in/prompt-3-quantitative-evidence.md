# Stage 2, sub-prompt 3 - the quantitative case for the Phase 1 robotic trial

## Goal

The master prompt asks that the quantitative data and tables from author sources
be sufficient for a funder to be convinced of the robotic Phase 1 clinical
trial. This sub-prompt fixes which numbers carry that weight and where each one
is printed.

## The evidence tables

| Table | Carries | Source |
|:--|:--|:--|
| Simulation to observation | 100,000-patient triplicate, ten-arm QSP at 12.8 months median overall survival against 5.4, 1000-patient digital twin at hazard ratio 0.31, RASolute 302 at 13.2 against 6.6 | `funding/pdac-funding-applications/final-apply/sections/sec-05-trial-evidence.tex` |
| Credibility | ASME V\&V 40 and ICH M15 aligned credibility score 81.9 across 55 verification notebook tests | The digital twin paper, DOI 10.5281/zenodo.17239510 |
| Cost benchmark | $36,330 **projected** per run against industry benchmarks above $120,000 for an empirical triplicate, above $2,000,000 for a ten-arm QSP trial, and $28,000 to $700,000 for a digital-twin platform | `funding/capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex` |
| Twenty-paper readiness | Each paper, its deposit date, its DOI, and the requirement the La Jolla site inherits from it | The seminar deck README in `inputs/READMES/` |
| Site capacity | 14 robot instances across 8 types, 2 procedure suites, extended-day operation, up to 18 treated participants in a 3+3 escalation | This paper, derived in document 08 and document 13 |
| Five-year budget | $700,000 per year, $3,500,000 over five years, personnel at $521,000 per year across 3.95 award-funded full-time equivalents | `funding/pdac-funding-applications/final-apply/sections/sec-08-budget-and-leverage.tex` |
| Staffing | Eleven named roles, their full-time equivalent fraction on the award, and the qualification each must hold | This paper, document 14 |
| Format comparison | Conventional Phase 1 site staffing of 80 to 120 full-time equivalents against this site's roster, and the per-participant cost consequence | The San Francisco package README in `inputs/READMES/` |

## The honesty constraint

Three claims must appear beside the numbers, every time:

1. The simulated 2.4-fold ratio and the observed 2.0-fold ratio are close, and
   that proximity is a chronology observation and hypothesis-supporting. It is
   not a validation claim. Three differences are material: 1000 simulated
   against 241 enrolled, a combination against a single agent, and KRAS G12C
   against a primarily G12D and G12V population.
2. RASolute 302 is metastatic and previously treated, and is silent on the
   resectable setting this trial addresses.
3. The contributed non-federal column carries no dollar figure, because no
   agreement exists and an invented cost-share number is worse than none.

## Acceptance

- Every table above exists, is set at the body measure, and is referred to by
  number at least once in the running text.
- Every simulation number appears in the same row or sentence as the limitation
  its own authors stated.
- The word "estimated" does not appear next to $36,330 anywhere.

## Commit

Folded into the seventeen section commits, then audited in the stage error pass.
