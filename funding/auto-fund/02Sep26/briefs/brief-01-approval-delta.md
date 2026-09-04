# What the Rasonque Approval Changes for This Program

**ChemicalQDevice, San Diego.** Kevin Kawchak, CEO.
Prepared for technical reviewers. Independent work, not medical or regulatory
advice, and not endorsed by the FDA, NIH, HHS, an IRB, ICH, or any sponsor.

---

## The fact

On August 26, 2026 the FDA approved Rasonque (daraxonrasib), Revolution
Medicines' RAS(ON) multi-selective inhibitor, as the first-in-class targeted
therapy for metastatic pancreatic cancer.
[FDA press announcement](https://www.fda.gov/news-events/press-announcements/fda-approves-first-class-targeted-therapy-metastatic-pancreatic-cancer).

## The chronology this company can date

| When | What | Where it is deposited |
|:--|:--|:--|
| June 2025 | Forty PDAC meta-analyses, over 400,000 words. A daraxonrasib combination ranked as the lead funding candidate | [10.5281/zenodo.15735068](https://doi.org/10.5281/zenodo.15735068) |
| August 2025 | Ten-arm QSP simulation, 250 ODEs. Median overall survival 12.8 months against 5.4 for chemotherapy | [10.5281/zenodo.17001137](https://doi.org/10.5281/zenodo.17001137) |
| May 2026 | RASolute 302 reports 13.2 months against 6.6 in the RAS G12 previously treated metastatic population | [10.1056/NEJMoa2605555](https://doi.org/10.1056/NEJMoa2605555) |
| August 2026 | FDA approval of Rasonque in metastatic pancreatic cancer | FDA press announcement, linked above |

The simulated ratio was 2.4-fold. The observed ratio was 2.0-fold.

**This is a chronology observation and a hypothesis-supporting one. It is not a
validation claim.** Three differences are material and none of them is small:
1000 simulated participants against 241 enrolled; a combination against a single
agent; and KRAS G12C selection against a primarily G12D and G12V population.

## What the approval changes

| Dimension | Before August 26, 2026 | After |
|:--|:--|:--|
| Agent status | Investigational, Phase 3 ongoing | Approved and labeled in metastatic disease |
| Risk a funder underwrites | Agent risk, device risk, workflow risk | Device risk and workflow risk |
| Supply conversation | Whether an investigational agent can be obtained | Whether a perioperative investigational use of an approved agent can be supported |
| Method evidence | An argued method | A method with one dated external checkpoint |

## What the approval does not change

Four items, stated plainly because each of them is a place where an enthusiastic
reading of the approval would be wrong.

1. **The proposed use remains investigational.** The approval covers metastatic
   disease. This program proposes perioperative use in resectable and
   borderline-resectable disease. An investigational new drug application is
   still required, and no document from this company says otherwise.
2. **No supply arrangement exists.** No drug supply agreement, letter of
   authorization, or regulatory cross-reference is in place with the agent's
   developer, and no approach has been made.
3. **The device and software questions are untouched.** The advisory boundary,
   the 3 millisecond arm-level stop, the 500 millisecond system-wide stop, the
   verification harness, and the human-authority guarantee are exactly where they
   were before the approval.
4. **The novelty claim is unchanged.** Daraxonrasib is nowhere described as
   first in human. The supportable claim concerns the first prospective clinical
   evaluation of the integrated surgical and advisory workflow, subject to FDA
   and institutional confirmation.

## Why this matters for a Phase I award specifically

A Small Business Innovation Research Phase I exists to retire a technical risk
cheaply enough that a Phase II is worth funding. Before the approval, a Phase I
in this program was carrying part of an agent question it could not answer and
was not designed to answer. After the approval, the whole of a Phase I can be
spent on the question it is good at: whether a governed advisory layer can be
verified, bounded, and shown to leave surgical authority intact.

That is a narrower Phase I than the one described in the August inquiry, and this
company is proposing the narrowing rather than waiting to be asked for it.

## The advisory boundary, in one paragraph

The model process holds no write credential to the electronic data capture system
and no route to the robot control network. That is a property of the wiring, not
of a policy document, and it can be tested by an auditor who is given the network
diagram and no cooperation from the operator. Arm-level stop is specified at 3
milliseconds and system-wide stop at 500 milliseconds. The operating surgeon
approves every motion. There is no configuration of the system in which a model
output reaches an actuator.

---

**Sources.** Repository sources for every figure above:
`funding/capitalization-plan/final-capital`,
`funding/pdac-funding-applications/applications/app-05-nih-sbir-seed`,
`funding/daraxonrasib-llm-story.md`.
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).

*Disclaimer: This work is independent and is not endorsed or sponsored by any
trial sponsor, CRO, site, IRB, regulator, or medical society; and was adapted
using Claude Code Opus 5.*
