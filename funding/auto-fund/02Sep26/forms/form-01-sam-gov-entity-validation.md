# Form Pack 01: SAM.gov Entity Validation, and the Three Registrations Behind It

**Portal.** [https://sam.gov](https://sam.gov) - Entity Registration.
**Follow-through.** [https://grants.gov](https://grants.gov) organization
registration, then [https://commons.era.nih.gov](https://commons.era.nih.gov).
**Support when it fails.** Federal Service Desk,
[https://www.fsd.gov](https://www.fsd.gov). Grants.gov support,
`support@grants.gov`, 800-518-4726.

---

## Before the first field

| Prerequisite | Where it comes from | Check |
|:--|:--|:--|
| Legal business name, exactly as filed | California Secretary of State record | Must match the state filing character for character, including "LLC" |
| Physical address, no post office box | The company's address of record | SAM.gov rejects a PO box for the physical address field |
| Taxpayer identification number | IRS record | Must match the IRS name control, which is derived from the legal name |
| Bank routing and account, for electronic funds transfer | The corporate account | Entity registration is incomplete without it |
| Unique Entity Identifier | Assigned by SAM.gov at validation | If one exists already, do not request a second |

## Field answers

| Field | Answer | Limit |
|:--|:--|:--|
| Legal business name | ChemicalQDevice LLC | Portal-validated against IRS |
| Doing business as | ChemicalQDevice | 120 |
| Entity structure | Limited liability company | Selection |
| State of incorporation | California | Selection |
| Date of incorporation | October 2021 | Date |
| Entity type for federal awards | Small business, for-profit organization other than small business is not selected | Selection |
| Business type, additional | Sole proprietorship is not selected; the entity is a single-member LLC | Selection |
| Purpose of registration | All awards | Selection |
| Primary NAICS | 541715, Research and Development in the Physical, Engineering, and Life Sciences except Nanotechnology and Biotechnology | Selection |
| Secondary NAICS | 541714, Research and Development in Biotechnology except Nanobiotechnology | Selection |
| Electronic Business point of contact | Kevin Kawchak, kevink@chemicalqdevice.com | Contact block |
| Government Business point of contact | Kevin Kawchak, kevink@chemicalqdevice.com | Contact block |
| Past Performance point of contact | Kevin Kawchak, kevink@chemicalqdevice.com | Contact block |
| Size metrics, average annual receipts | Report the three-year average from the filed returns; do not estimate | Numeric |
| Size metrics, average number of employees | 1 | Numeric |
| Financial assistance certification | Complete; the entity seeks financial assistance awards | Selection |

## Narrative field: entity description

Paste the following. **412 characters with spaces**, inside the common 500
limit.

> ChemicalQDevice LLC is a California single-member limited liability company
> formed in October 2021, developing verification-first artificial intelligence
> and physical AI methods for oncology clinical trials. Its current program is a
> Phase 1 governed surgical advisory system for robotic pancreaticoduodenectomy
> in KRAS-mutated pancreatic ductal adenocarcinoma. All work is published in a
> public repository.

## The two registrations that follow, in order

| Order | Registration | Portal | Depends on |
|:--|:--|:--|:--|
| 1 | Grants.gov organization registration | grants.gov | An active SAM.gov entity with a Unique Entity Identifier |
| 2 | eRA Commons organization, plus separate Signing Official and PD/PI accounts | commons.era.nih.gov | Grants.gov organization registration |

If the chief executive will occupy both the Signing Official and the PD/PI roles,
**two separate eRA Commons accounts are required**, not one account with two
roles. Then link ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667)
to the PD/PI profile; NIH matches publications through that link and an unlinked
profile shows an empty record.

## After submission

| What is returned | Typical time | What to record |
|:--|:--|:--|
| Entity validation outcome | 1 to 10 business days | The validation ticket number |
| Unique Entity Identifier | On successful validation | The identifier, in the repository entity record |
| Registration active date and expiration date | On activation | The expiration date, and a reminder 60 days before it |

## The three ways this goes wrong

1. **A name mismatch of one character.** "ChemicalQDevice LLC" against
   "Chemical Q Device, LLC" fails IRS validation and the failure message does not
   say which field caused it. Compare against the state filing before typing.
2. **An expired registration nobody noticed.** SAM.gov registration lapses
   annually and a lapsed entity cannot receive an award. Federal submissions do
   not always warn. Set the 60-day reminder at activation, not later.
3. **A single eRA Commons account for both roles.** The submission is accepted
   and then rejected at validation, which costs a receipt date. Create both
   accounts before the first application, not during one.

---

**Sources.** `funding/potential-partners/UC-San-Diego/priority-steps.md` §13;
`funding/capitalization-plan/final-capital/sections/sec-02-entity-and-asset.tex`.
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).

*Nothing in this pack has been submitted. Every value must be confirmed against
the company's filed records on the day of entry.*
