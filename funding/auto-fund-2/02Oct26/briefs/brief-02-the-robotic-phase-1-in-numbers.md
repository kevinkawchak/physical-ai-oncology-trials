# The Robotic Phase 1, in Numbers

**ChemicalQDevice, San Diego.** Kevin Kawchak, CEO.
For a funder, and for the NCI program director today's letter 1 asks to be
directed to. About 1,000 words, of which the last section is the one-page
Specific Aims.

---

## Why numbers

A funder deciding whether a first-in-human robotic trial is worth its money
needs to see that every part of it has been specified to a number someone can
check. Every figure below is taken from a deposited document with a persistent
identifier, and each table names its source.

## The design

| Parameter | Value |
|:--|:--|
| Phase and type | Phase 1, first-in-human, combined IND and IDE |
| Design | Open-label, single-arm, 3+3 dose escalation with staggered sentinel enrollment |
| Dose levels of daraxonrasib | 160, 220 and 300 mg |
| Participants | Up to 18 |
| Dose-limiting toxicity window | 28 days |
| Primary endpoints | Dose-limiting toxicity rate; device- and procedure-related serious adverse events through day 30, including Clavien-Dindo grade III or higher |
| Key secondary endpoints | R0 resection at day-90 pathology; ISGPS grade B or C fistula; 90-day mortality; progression-free and overall survival to 24 months |
| Follow-up | Every 12 weeks to 24 months |
| Postoperative restart | Day 7, 14 or 21, keyed to fistula status and drug trough |

Source: the Phase 1 protocol, https://doi.org/10.5281/zenodo.20780121

## The procedure and its safety limits

| Parameter | Value |
|:--|:--|
| Arms and degrees of freedom | 8 arms, 56 degrees of freedom |
| Sensor channels | 640, 80 per arm |
| Positional accuracy | 0.05 mm root mean square at 1,200 mm/s |
| Heartbeat bus | 10 kHz |
| Cross-arm emergency stop | 3 ms or less |
| System stop | 500 ms |
| Tip force | 3 N or less per arm; 18 N or less across all arms |
| Vascular no-fly gating | A five-vessel gate, including the superior mesenteric and portal veins |
| Gate before any patient | At least 1,000 simulated procedures across at least 2 independent frameworks, with a Unified Safety Level of at least 7.0 |
| Language model | On premises, advisory only; no write credential to data capture; no route to robot control |

Sources: the Phase 1 protocol and the investigational new drug application,
https://doi.org/10.5281/zenodo.21097442

## The evidence behind the choice of agent

| Source | Result | What it is |
|:--|:--|:--|
| Company QSP simulation, August 2025, 10 arms | 12.8 against 5.4 months median overall survival; hazard ratio 0.25 | In silico |
| Company digital twin, 1,000 patients | 12.1 months median overall survival; progression-free hazard ratio 0.31 | In silico |
| Company credibility battery | 81.9 over 55 tests | A pre-trial score, not a validation |
| RASolute 302, May 2026, RAS G12 population | 13.2 against 6.6 months | A Phase 3 trial |
| FDA approval, August 26, 2026, overall population | 13.2 against 6.7 months | The label |

The simulated and reported figures are close, and the company calls that
closeness a chronology, not a validation: the populations, regimens and sample
sizes differ.

## The money

| Item | Amount | Term |
|:--|:--|:--|
| NIH SBIR Phase I, in the plan | $306,000, of which $266,000 direct | 9 months |
| NIH SBIR Phase II, in the plan | $1,300,000, of which $1,130,000 direct | 24 months |
| Clinical conduct inside Phase II | $288,000 for 6 participants, $48,000 each | Within Phase II |
| Program direct cost | $3,500,000, or $700,000 a year | 60 months |
| Personnel inside the annual cost | $521,000 across 3.95 full-time equivalents | Per year |
| Delta the SBIR route does not buy | $2,104,000 | 27 months |
| Private capital behind the firewall | $5,900,000 | A plan, not a completed raise |
| Comparable virtual trial work | $36,330 per run, projected, against above $120,000 | About one month, against 4.5 |

If an NSF Phase I is awarded for the verification work, the rig and
verification lines of the NIH Phase I budget are revised so that nothing is paid
twice.

Sources: `funding/capitalization-plan/final-capital`, §3 and §6;
`funding/move-in/final-move-in`, §14.

## Specific Aims, one page

**A governed, advisory-only language model layer for a first-in-human robotic
pancreaticoduodenectomy with perioperative daraxonrasib.**

Pancreatic ductal adenocarcinoma is the third leading cause of cancer death in
the United States, and fewer than 13 percent of patients live five years. On
August 26, 2026 the FDA approved daraxonrasib as Rasonque for metastatic
pancreatic adenocarcinoma after prior systemic therapy, on overall survival of
13.2 against 6.7 months. Its use around surgery for resectable disease is
untested. ChemicalQDevice has deposited a Phase 1 protocol and an
investigational new drug application that pair perioperative daraxonrasib with a
staged eight-arm robotic pancreaticoduodenectomy, in which an on-premises
language model advises the surgical team and has no route to act.

The gap is that no evidence yet shows such a layer can be used inside a trial
safely: which decisions it may advise on, how a surgical team uses its advice,
and what credibility evidence supports that use. The objective of this Phase I
is to close that gap before the first participant is enrolled.

**Aim 1. Fix the layer's context of use inside the protocol.** Define the
decisions it may advise on, and those it may never touch: dose assignment,
dose-limiting toxicity calls, and analysis. Define the form in which advice
reaches the surgeon and how every piece of advice is recorded. *Milestone:* a
protocol section and an IRB-ready description accepted by the site investigator.

**Aim 2. Evaluate the layer in simulated use with the site's surgical team.**
Run the staged procedure in simulation, recording every piece of advice, every
case in which advice was withheld, and the time from advice to decision, as part
of the protocol's gate of at least 1,000 simulated procedures across at least two
independent frameworks. *Milestone:* a simulated-use report in which no advice
reached a safety control and every withheld case is explained.

**Aim 3. Assemble the credibility evidence for that context of use.** Extend the
company's 55-test credibility battery, which scores 81.9 today, to the layer's
advice on the protocol's decisions, framed on ASME V&V 40 and read against the
FDA's draft guidance on artificial intelligence in regulatory decision-making.
Cite, and do not repeat, any NSF-funded verification results. *Milestone:* a
credibility report ready for an amendment to the investigational new drug
application.

**Expected outcome.** A trial-ready advisory layer with a defined context of use,
measured simulated-use performance, and a credibility report, so that a Phase II
can fund clinical conduct for the first six participants. **Budget:** $306,000
over nine months in the company's plan.

## Sources

- Phase 1 protocol: https://doi.org/10.5281/zenodo.20780121
- Investigational new drug application: https://doi.org/10.5281/zenodo.21097442
- Ten-arm simulation: https://doi.org/10.5281/zenodo.17001137
- RASolute 302: https://doi.org/10.1056/NEJMoa2605555
- The capitalization plan: https://doi.org/10.5281/zenodo.21887807
