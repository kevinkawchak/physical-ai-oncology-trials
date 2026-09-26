# 28Sep26 - Day 1, The Unsolicited Applicants (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-2E5E4E.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-2E5E4E.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-3-6C757D.svg)](emails)
[![LinkedIn](https://img.shields.io/badge/LinkedIn%20replies-2-6C757D.svg)](linkedin)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](forms)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20Unsolicited%20Applicants-2E5E4E.svg)](packet)
[![Privacy](https://img.shields.io/badge/Applicant%20names-not%20in%20repository-9AA1A8.svg)](#privacy-and-equal-treatment)

In the week of September 21, 2026 two people wrote to the chief executive on
LinkedIn, each on their own initiative, asking to work for ChemicalQDevice. The
company had posted no position and published no hiring information. Both found
the company through its public work: the deposited papers, the repository, and
the program built around a molecule the FDA approved on August 26, 2026 as
Rasonque.

That is a small event, and it is the most fundable thing that happened all week.

## The one approval step

> **Approve the same reply to both applicants, and the three letters that open
> routes to fund them.**

No offer is made, no interview is promised, and no position is posted. The
replies say there is no open position today, describe the eleven roles the
company's federal applications would fund, and ask each applicant for a resume,
the role closest to their experience, and consent to be named as prospective
personnel contingent on an award.

## Why this matters to a funder

Every federal small-business reviewer scores the team. Until last week the
company's answer was a plan: eleven roles at 3.95 award-funded full-time
equivalents and a $521,000 personnel line inside a $700,000 annual direct cost.
Two unsolicited inquiries do three things that plan could not do alone.

| What a reviewer asks | What the plan could say | What the two inquiries add |
|:--|:--|:--|
| Can this company attract people? | That it intends to recruit | That two people came before it recruited anyone |
| Does its public work reach anyone? | Download counts and citations | Two people who read it and asked to join |
| Will the roles be filled on award? | A recruiting timeline | A named, consenting pipeline, if both agree |

It also matters for the other three signals of the week. The SEC request, the
White House correspondence and the NSF inquiry each ask, in different words,
whether this company can work inside a structure. The first two people who asked
to join it are where that answer is first tested.

## The run order

| Order | Item | Where | Depends on |
|:--|:--|:--|:--|
| 1 | Fill each applicant's first name and paste the reply into the LinkedIn conversation they opened | [`linkedin/`](linkedin) | Nothing |
| 2 | Send the NIH SBIR question on naming contingent personnel | [`emails/email-01-nih-sbir-contingent-personnel.txt`](emails/email-01-nih-sbir-contingent-personnel.txt) | Nothing |
| 3 | Send the first-hire readiness questions | [`emails/email-02-first-hire-readiness.txt`](emails/email-02-first-hire-readiness.txt) | Nothing |
| 4 | Send the research staff training question to UC San Diego | [`emails/email-03-ucsd-research-staff-training.txt`](emails/email-03-ucsd-research-staff-training.txt) | Nothing |
| 5 | Read the role-fit brief before any conversation with either applicant | [`briefs/brief-02-role-fit-and-structured-interview.md`](briefs/brief-02-role-fit-and-structured-interview.md) | An applicant's reply |
| 6 | Approve the payroll earmark as a rule, with no purchase | [`investing/capital-01-contingent-payroll-earmark.md`](investing/capital-01-contingent-payroll-earmark.md) | Nothing |
| 7 | Hold both form packs until a start date exists | [`forms/`](forms) | An award and an accepted offer |

## Privacy and equal treatment

Neither applicant is named in this repository, and neither will be. The
repository is public, and a person who writes privately to a chief executive has
not agreed to appear in it. The files refer to them as Applicant A and Applicant
B in the order their messages arrived, and the names live only in the LinkedIn
conversations.

The two replies are identical in substance. A first employer that answers two
unsolicited applicants differently creates the first unequal-treatment record in
its history, and it does so before it has any policy to point to. The interview
plan in [`briefs/`](briefs) asks both applicants the same questions in the same
order and scores them against the same rubric.

## Directory contents

```
28Sep26/
├── README.md              this approval sheet
├── linkedin/              2 .txt replies, one per applicant, pasted into LinkedIn
├── emails/                3 .txt letters, each with at least three verified addresses
├── briefs/                3 .md briefs: why it matters, role fit, contingent personnel
├── forms/                 2 .md packs: EDD employer registration, I-9 and new hire report
├── investing/             1 .md capital instruction: the contingent payroll earmark
├── diagrams/              3 .md figure specifications
└── packet/                The Unsolicited Applicants: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 28Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../move-in`](../../move-in) | The eleven roles in both replies, brief 2, Table 3 and Figure 2 |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../move-in`](../../move-in) | The separation of award funds from the company reserve |
| `applications/app-05-nih-sbir-seed/` | [`../../pdac-funding-applications`](../../pdac-funding-applications) | Letter 1 and Table 5 |
| `02Sep26/emails/email-01-nih-seed-sbir-recontact.txt` | [`../../auto-fund`](../../auto-fund) | The earlier re-contact letter 1 refers to |
| `04Sep26/emails/email-01-ucsd-moores-escalation.txt` | [`../../auto-fund`](../../auto-fund) | The earlier feasibility contact letter 3 refers to |
| `02Sep26/investing/capital-01-treasury-ladder.md` | [`../../auto-fund`](../../auto-fund) | The ladder the payroll earmark sits inside |
| `02Sep26/forms/form-01-sam-gov-entity-validation.md` | [`../../auto-fund`](../../auto-fund) | The NAICS codes in the EDD form pack |

## Positioning, carried into every file in this directory

No position is open and no offer is made. Naming a person as prospective
personnel in a federal application creates no obligation on either side until an
award is made and an offer is accepted. No agreement of any kind exists with UC
San Diego or any other institution. Rasonque is approved in the metastatic
setting, and the perioperative use this program proposes remains
investigational. No order in [`investing/`](investing) is investment advice.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
