# What an External Standard Requires

**ChemicalQDevice, San Diego.** Kevin Kawchak, CEO.
For the FDA program and any technical reviewer. About 850 words.

---

## Why this brief exists

A request to be listed as an external standard invites one question before any
other: a standard by what measure? This brief sets out eight criteria that any
body of work would have to meet before a reviewer could reference it as a
benchmark for oncology clinical trial research, and it scores the company's
record against each one. Three criteria are met, three are met in part, and two
are not yet met. The brief says so rather than rounding up.

## The eight criteria

| # | Criterion | What it asks | The record today | Status |
|:--|:--|:--|:--|:--|
| 1 | Persistent identifiers | Can each work be cited permanently? | Eleven DOIs across the program | Met |
| 2 | Dated versions | Can a reader tell which version was cited, and when? | Deposit dates; versioned releases v4.0.0 to v4.9.0 | Met |
| 3 | Open method | Can the method be inspected and rerun? | Public repository with source, prompts and build records | Met |
| 4 | Credibility assessment | Has the model been scored against a credibility framework? | 55-test VVUQ score of 81.9 framed on ASME V and V 40 | In part |
| 5 | Regulatory mapping | Is each document mapped to the rule it serves? | 21 CFR Part 312, ICH adaptations, FDA AI draft guidance | In part |
| 6 | Independent review | Has anyone outside the company reviewed it? | Two independent model reviewers; no human external review | In part |
| 7 | Outcome validation | Has any prediction been tested against a trial result? | One chronology observation; no prospective test | Not yet |
| 8 | Adoption | Has any other group used it as a reference? | The company's statement about OpenAI and Anthropic, not independently confirmed | Not yet |

## Reading the three partial scores

**Credibility assessment.** The digital twin behind the survival simulation was
scored at 81.9 over 55 verification, validation and uncertainty quantification
tests, framed on the ASME V and V 40 approach to computational model
credibility. That is a pre-trial credibility score. It is not a post-trial
validation, and it covers the model rather than the trial documents.

**Regulatory mapping.** The investigational new drug application and the two
protocols are mapped to 21 CFR Part 312 and to adapted ICH guidance, and the use
of language models is described against the FDA's draft guidance on AI to
support regulatory decision-making. No regulator has reviewed that mapping.

**Independent review.** Every document is produced by one model and reviewed by
two others from different developers, and the division of work is published. A
model reviewer is not a human expert, and a standard would need both.

## Reading the two unmet scores

**Outcome validation.** The only comparison between a company prediction and a
trial result is the simulation's 12.8 months against RASolute 302's 13.2 months.
The populations, regimens and sample sizes differ, so it is a chronology, not a
test. A prospective test would require the company to publish a prediction
before a trial reports, and to be scored on it afterward.

**Adoption.** The request's basis is the company's statement that its papers
have been used extensively by OpenAI and Anthropic. The company can document its
own use of their tools. It cannot document the two companies' internal use of
its papers, and it has written to both inviting a correction. Until either
confirms, criterion 8 is unmet on the independent evidence.

## What the company can do in the next ninety days

| Criterion | Action | Owner |
|:--|:--|:--|
| 6 | Ask one clinical investigator to review the Phase 1 protocol and publish the review with the investigator's consent | The chief executive, through the site conversations on day 3 |
| 7 | Publish a dated prediction for one endpoint of a trial that has not yet reported | The chief executive |
| 8 | Record any reply from OpenAI or Anthropic exactly as received | The chief executive |
| 5 | Ask the FDA program which venue owns the question | Letter 2, today |

## Sources

- ASME V and V 40: https://www.asme.org/codes-standards/find-codes-standards/v-v-40-assessing-credibility-computational-modeling-verification-validation-application-medical-devices
- ICH M15: https://database.ich.org/sites/default/files/ICH_M15_Step2_draft_Guideline_2024_0523.pdf
- FDA AI draft guidance: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological
- The credibility score and the six checkable quantities: `funding/capitalization-plan/final-capital/sections/sec-06-clinical-evidence.tex`
