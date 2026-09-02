# Form Pack 01: Grants.gov Workspace Setup

**Portal.** [https://www.grants.gov](https://www.grants.gov)
**Support.** `support@grants.gov`, 800-518-4726. **Closed today.**
**Status.** Completed offline. **Not submitted.**

---

## The dependency to check first

| Check | Where | If it fails |
|:--|:--|:--|
| SAM.gov registration is active, not expired | [sam.gov](https://sam.gov) entity record | Nothing below can be completed. Renewal is the whole of the next session's work |
| Unique Entity Identifier matches the SAM record character for character | The entity record | A mismatch reports as a generic error and costs an hour to diagnose |
| SBA Company Registry control number is current | [sbir.gov](https://www.sbir.gov/registration) | An SBIR application cannot be submitted without it |

## What a workspace is, and why it is created before it is needed

A Grants.gov workspace is the container an application is assembled in. It can be
created against an opportunity, populated over weeks, and submitted at the end. A
workspace created on the day an application is due is a workspace created under
time pressure, and the forms inside it inherit that.

## Role assignments

| Role | Person | Note |
|:--|:--|:--|
| Authorized Organization Representative | Kevin Kawchak | The role that submits. Requires SAM.gov eBiz point of contact approval, which is a separate step and is not instant |
| Workspace Owner | Kevin Kawchak | Creates and manages the workspace |
| Workspace Participant | None | Added when a subaward or a consultant joins |

The Authorized Organization Representative approval is the item most likely to
delay a first submission, because it is granted by the entity's eBiz point of
contact in SAM.gov rather than by Grants.gov, and a one-person company is both
parties. Confirm the approval exists before an opportunity is chosen.

## Standard form answers to prepare

| Form | Field | Answer |
|:--|:--|:--|
| SF-424 | Applicant legal name | ChemicalQDevice LLC |
| SF-424 | Unique Entity Identifier | From SAM.gov |
| SF-424 | Type of applicant | Small business, for-profit organization |
| SF-424 | Congressional district | The district of the company's address of record |
| SF-424 | Project director | Kevin Kawchak, Chief Executive Officer |
| SF-424 | Areas affected | California, San Diego County |
| SF-424 | Is the applicant delinquent on any federal debt | No |
| R&R Other Project Information | Human subjects | Yes |
| R&R Other Project Information | Clinical trial | Yes |
| R&R Other Project Information | Vertebrate animals | No |
| R&R Other Project Information | Environmental impact | No |
| R&R Senior Key Person | ORCID | 0009-0007-5457-8667 |

## The narrative attachments to have ready

| Attachment | Source in this repository |
|:--|:--|
| Project summary and abstract | `02Sep26/briefs/brief-02-sbir-phase-i-readiness.md` |
| Specific aims | The Phase 1 protocol, [10.5281/zenodo.20780121](https://doi.org/10.5281/zenodo.20780121) |
| Research strategy | The same, plus the evidence pack |
| Facilities and other resources | `funding/move-in/final-move-in` |
| Equipment | The same |
| Biosketch | To be assembled; ORCID-linked publication record |
| Budget and justification | `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed` |

## The three ways this goes wrong

1. **No Authorized Organization Representative approval.** The workspace exists,
   the application is complete, and nobody can press submit.
2. **An expired SAM.gov registration discovered on the due date.** Renewal is not
   instant.
3. **A PDF that fails the portal's own validation.** Grants.gov rejects some
   generated PDFs. Validate every attachment in the workspace well before a due
   date, not on it.

---

**Sources.** `funding/potential-partners/UC-San-Diego/priority-steps.md` §13;
`funding/auto-fund/02Sep26/forms/form-01-sam-gov-entity-validation.md`;
[Grants.gov](https://www.grants.gov).
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).

*Nothing in this pack has been submitted.*
