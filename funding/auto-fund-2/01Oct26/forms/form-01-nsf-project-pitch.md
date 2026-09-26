# Form Pack 01: The NSF SBIR Project Pitch

**Portal:** The NSF SBIR/STTR Project Pitch submission, under NSF 26-510.
**Status:** Prepared. **Submit on day 4**, after the Research.gov organization
record in `form-02-research-gov-organization.md` is confirmed.

---

## What the Pitch is, and what it is not

A Project Pitch is the required first step before an NSF SBIR Phase I proposal.
NSF answers each Pitch by email; only an invited Pitch may become a full
proposal. A company may submit no more than two Pitches a year, so this one is
written to be the right one. Full proposals then fall on NSF's published
deadlines: the current solicitation lists November 4, 2026, and then the first
Wednesday in November and the first Thursday in March of later years. Confirm the
next deadline on the solicitation page on the day of submission.

The Pitch asks NSF to fund technology, because NSF's SBIR program does not fund
clinical trials. It describes the governed, advisory-only language model layer
and the method that verifies it, and it proposes no trial and uses no patient
data.

## The identifying fields

| Field | Answer |
|:--|:--|
| Company name | ChemicalQDevice LLC |
| Company address | The company's registered San Diego address |
| Primary contact | Kevin Kawchak, Chief Executive Officer, `kevink@chemicalqdevice.com` |
| Program | SBIR |
| Technology topic | Artificial Intelligence; if the dropdown offers a clinical decision support or digital health subtopic, choose the one that names clinical AI |
| Prior NSF SBIR or STTR award | None |
| Referral | NSF program staff inquiry about a pilot program |

## The four narrative fields

Each field is pasted exactly as below. The counts include spaces and were
measured on the text as written here.

| Field | Limit | This draft | Headroom |
|:--|:--|:--|:--|
| Technology Innovation | 3,500 characters | 2,144 characters | 1,356 |
| Technical Objectives and Challenges | 3,500 characters | 1,772 characters | 1,728 |
| Market Opportunity | 1,750 characters | 1,082 characters | 668 |
| Company and Team | 1,750 characters | 1,032 characters | 718 |

### Technology Innovation

```
Robot-assisted cancer surgery is starting to use large language models as advisors, but no method yet exists to show that such a model stays inside its boundaries and that its advice can be trusted before it reaches an operating room. ChemicalQDevice proposes a governed, advisory-only language model layer for robotic oncologic surgery, and a verification method that makes its behavior checkable. The layer runs on premises at a clinical site. It produces text recommendations to the operating surgeon and nothing else: it holds no write credential to the electronic data capture system, no route to the robot control network, and no ability to act on equipment. Every output is checked by two independent models from different developers before it is shown, and every check is written to an audit trail that can be replayed. The innovation is the combination of three things that current clinical AI systems treat separately: a hard architectural boundary, enforced by network segmentation and credential design rather than by instructions to the model; cross-model verification, in which disagreement between independent models is treated as a signal to withhold advice rather than as noise; and a credibility assessment framed on the ASME V&V 40 approach to computational models, extended from simulation to advisory text. The company has built and deposited the documentation this layer sits inside: a Phase 1 protocol for a staged robotic pancreaticoduodenectomy, an investigational new drug application, and site documentation, each with a persistent identifier. Its survival simulation of the agent in that protocol, run in August 2025, returned 12.8 months of median overall survival against 5.4 for chemotherapy, and the agent's later Phase 3 trial reported 13.2 months against 6.6. The company treats that proximity as a dated chronology, not a validation, and the verification method proposed here is how it would turn chronology into evidence. What exists today is a documented design and a public record. What does not yet exist is a tested verification method with measured failure rates, and that is what a Phase I would build.
```

### Technical Objectives and Challenges

```
The Phase I would retire four technical risks, each with a measure and a threshold. First, boundary integrity. The model must hold zero write credentials and have zero network paths to robot control. The objective is an automated test suite that attempts every known path across the boundary and reports none open, run on every build. Second, verification yield. Cross-model checking only helps if disagreement is rare enough to be usable and informative when it occurs. The objective is to measure, on synthetic cases generated from published parameters, how often two independent verifier models disagree with the primary model, how often disagreement coincides with an error seeded into the case, and at what threshold advice should be withheld. Third, credibility. The company's existing digital twin scored 81.9 on a 55-test credibility battery framed on ASME V&V 40. The objective is to extend that battery from simulation outputs to advisory text and to decision rules in the protocol, and to publish the results with their limitations. Fourth, timing. The layer must never delay a safety response. The protocol fixes a 3 ms cross-arm stop and a 500 ms system stop, a 3 N per-arm and 18 N cumulative force limit, and a five-vessel no-fly gate, all enforced outside the model. The objective is to show, on a hardware-in-the-loop test bench, that advisory latency cannot interact with any of them. The main technical challenges are building synthetic cases realistic enough to expose failure, choosing verifier models that fail independently rather than together, and measuring rare events without patient data. No clinical trial is proposed, and no patient data is used. The work produces a verification method, its test suite, and a public report of measured rates.
```

### Market Opportunity

```
The first customers are academic cancer centers and trial sponsors that want to use language models in regulated clinical research and cannot yet show a regulator or an institutional review board that the model stays within bounds. A verification method they can adopt, with a test suite and a report format, lowers that barrier for every trial that uses one. The second market is robotic surgery developers, who need a way to add advisory software to a platform without putting the platform's clearance at risk. An advisory layer that is architecturally incapable of acting on equipment is easier to place inside an existing regulatory boundary than one that is merely instructed not to. Competing approaches rely on prompt instructions, model fine-tuning, or human review of every output. The first two are not verifiable from outside, and the third does not scale. The company's approach is verifiable by design and publishes its failure rates. Revenue would come from licensing the method and test suite to sites and sponsors, and from services that apply it to their protocols.
```

### Company and Team

```
ChemicalQDevice is a San Diego small business led by its founder and chief executive, Kevin Kawchak, who designs, writes and verifies the company's clinical AI systems and has deposited a continuous public record of them since June 2025, including a Phase 1 protocol, an investigational new drug application, a Phase 2 protocol, and site documentation, each with a persistent identifier. The company works to a fixed early-start schedule under written operating rules, and its method pairs one production model with two independent reviewing models from different developers, with every change recorded in a public repository. Its staffing plan names eleven roles, six of which, including a systems engineer and site safety officer and a model governance lead, are the roles this Phase I would draw on. Two prospective staff approached the company without a posting. A clinical investigator and host institution would hold every clinical decision in any later trial; the company's role is the advisory software and its verification.
```

## Before submitting

- Confirm the Research.gov organization record is complete, per form pack 02.
- Confirm the next full-proposal deadline on the NSF 26-510 page.
- Confirm every number in the four fields against its source: the credibility
  score and the survival figures from
  `funding/capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`,
  and the force, stop and no-fly limits from the deposited Phase 1 protocol,
  https://doi.org/10.5281/zenodo.20780121.
- Confirm the Company and Team field's sentence about the two prospective staff
  is still true on the day of submission, and names neither person.
- Confirm no field proposes a clinical trial, uses patient data, or claims an
  award, a partnership, or an endorsement.
- Confirm no field describes Rasonque, or the agent, as approved for the
  perioperative use the company's protocol proposes.
- After submitting, record the submission date in the day's record block, and
  send letter 4 to NIH.

## Sources

- NSF 26-510: https://www.nsf.gov/funding/opportunities/small-business-innovation-research-small-business-technology/nsf26-510/solicitation
- NSF SBIR, America's Seed Fund: https://seedfund.nsf.gov/contact/
- ASME V and V 40: https://www.asme.org/codes-standards/find-codes-standards/v-v-40-assessing-credibility-computational-modeling-verification-validation-application-medical-devices
