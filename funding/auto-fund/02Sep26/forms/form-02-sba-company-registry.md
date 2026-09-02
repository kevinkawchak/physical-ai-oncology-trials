# Form Pack 02: SBA Company Registry

**Portal.** [https://www.sbir.gov/registration](https://www.sbir.gov/registration)
**What it produces.** An SBIR/STTR Company Registry control number, required on
every SBIR application at submission.
**Support when it fails.** SBA SBIR support through
[https://www.sbir.gov/contact](https://www.sbir.gov/contact).

---

## Why this pack is on day 1

An SBIR application without a current Company Registry control number is not
accepted. The number is free and the registration is short, but it depends on the
SAM.gov Unique Entity Identifier from
[`form-01-sam-gov-entity-validation.md`](form-01-sam-gov-entity-validation.md),
so the two are done in that order and both are done before the first SBIR
receipt date is chosen.

## Before the first field

| Prerequisite | Source |
|:--|:--|
| Unique Entity Identifier | SAM.gov, from pack 01 |
| Taxpayer identification number | IRS record |
| Ownership percentages summing to 100 | The operating agreement |
| Number of employees, including affiliates | Payroll record |

## Field answers

| Field | Answer | Note |
|:--|:--|:--|
| Company legal name | ChemicalQDevice LLC | Must match SAM.gov exactly |
| Unique Entity Identifier | From pack 01 | Do not type a placeholder |
| Business address | The company's physical address of record | No post office box |
| Company website | https://github.com/kevinkawchak/physical-ai-oncology-trials | The public repository is the company's public record of work |
| Year founded | 2021 | October 2021, California |
| Number of employees including affiliates | 1 | The 500-employee SBIR ceiling is not close |
| Is the company majority owned and controlled by one or more individuals who are citizens or permanent residents of the United States | Yes | 13 CFR 121.702 ownership and control |
| Is the company majority owned by multiple venture capital operating companies, hedge funds, or private equity firms | No | This answer changes if a priced round is taken; see the note below |
| Is the company a joint venture | No | |
| Primary contact | Kevin Kawchak, kevink@chemicalqdevice.com | |
| Principal Investigator primary employment | The Principal Investigator's primary employment must be with the small business during the award | Confirm against the effort commitment before answering |
| NAICS code | 541715 | Same as SAM.gov primary |
| SBIR agencies of interest | HHS, NIH; DOD; NSF; DOE | Selection, multiple |

## The one answer that a future financing can change

The ownership question is the reason day 2 of this block sets out three
instruments side by side rather than one. Under 13 CFR 121.702, majority
ownership by multiple venture capital operating companies, hedge funds, or
private equity firms changes SBIR eligibility and, at some agencies, requires a
separate authorization. A convertible instrument that has not converted does not
change the answer today; a priced round that transfers majority ownership does.

**Practical rule carried into day 2.** No instrument is signed before the effect
of that instrument on this field is written down. The registry answer is
re-checked at every financing event, not annually.

## Narrative field: technology area

Paste the following. **388 characters with spaces.**

> Verification-first artificial intelligence and physical AI for oncology
> clinical trials. Current program: a governed, advisory-only large language
> model layer supporting robotic pancreaticoduodenectomy with a perioperative
> RAS(ON) inhibitor in KRAS-mutated pancreatic ductal adenocarcinoma, with
> surgical authority retained at the interface rather than by policy.

## After submission

| What is returned | Typical time | What to record |
|:--|:--|:--|
| Company Registry control number | Immediate on completion | The control number, and the date it was issued |
| Registry record expiration | Stated on the record | A reminder at 60 days before |

The control number is entered on the SBIR application cover page. Record it in
the repository entity record on the day it is issued, because an application
assembled from memory on a receipt date is an application with a typo in it.

## The three ways this goes wrong

1. **A Unique Entity Identifier typed from memory.** One transposed character
   produces a registry record that does not match SAM.gov, and the mismatch
   surfaces at application submission rather than at registration.
2. **The Principal Investigator employment answer given optimistically.** SBIR
   requires the Principal Investigator's primary employment to be with the small
   business during the award. Answer it against the actual effort commitment.
3. **An ownership answer that is true today and false after a financing.** See
   the section above. The answer is re-checked at every financing event.

---

**Sources.** `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed`;
`funding/potential-partners/UC-San-Diego/priority-steps.md` §13;
`funding/capitalization-plan/final-capital` §2 and §4.
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).

*Nothing in this pack has been submitted. Every value must be confirmed against
the company's filed records on the day of entry.*
