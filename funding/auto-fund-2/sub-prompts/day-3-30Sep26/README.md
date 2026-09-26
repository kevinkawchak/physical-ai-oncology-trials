# Sub-prompt 3 - 30Sep26, The Structured Team (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-3%20of%205-7A1F2B.svg)](../../30Sep26)
[![Accent](https://img.shields.io/badge/Accent-Mission%20Oxblood%20%237A1F2B-7A1F2B.svg)](../../30Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-4-6C757D.svg)](../../30Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../30Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-1-6C757D.svg)](../../30Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../30Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../30Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../30Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

The day that answers the White House correspondence which followed the chief
executive's message to the President of September 12, 2026. The message named
three priority changes for the chief executive and the startup: obedience over
hard work and responsibilities; adaptation to existing authority, order, and
structure; and a teamwork-based early-rise, scheduled routine.

## The single decision this day asks for

**Does the chief executive approve answering the White House through its
contact form and the Office of Science and Technology Policy, and placing the
company, by letter, inside four established structures as a contributor rather
than a lead?**

## Why this day is shaped the way it is

A message that promises obedience to structure is tested by what follows it.
This day turns each of the three priorities into operating rules that a
counterparty can check, and then writes to four structures the company would
work inside: the federal AI-for-science effort coordinated from the White House,
the Department of Energy's Genesis Mission, the academic cancer center that
would hold the trial, and the National Cancer Institute's investigator-initiated
pathway. Each letter offers a contributor's place and asks for nothing larger.

The White House publishes no email address for correspondence. It is answered
through its own contact form, and the form pack carries the exact text.

## What this day produces

| # | Deliverable | Format | Recipient |
|:--|:--|:--|:--|
| 1 | `emails/email-01-ostp-response.txt` | `.txt` | OSTP, copied to DOE Genesis Mission partnerships and OPM AI workforce |
| 2 | `emails/email-02-genesis-mission-contributor.txt` | `.txt` | DOE Genesis Mission partnerships and funding opportunity contacts, copied to OSTP |
| 3 | `emails/email-03-ucsd-contributor-role.txt` | `.txt` | Moores Clinical Trials Office, ACTRI, Office of Clinical Trials Administration |
| 4 | `emails/email-04-nci-ctep-structure-followup.txt` | `.txt` | The three CTEP addresses of the existing thread |
| 5 | `forms/form-01-white-house-contact.md` | `.md` | The whitehouse.gov contact form |
| 6 | `briefs/brief-01-three-priorities-as-rules.md` | `.md` | Any reader of the September 12 message |
| 7 | `briefs/brief-02-the-structure-map.md` | `.md` | Anyone asking who governs what in the program |
| 8 | `briefs/brief-03-the-early-rise-weekday.md` | `.md` | The chief executive and every future team member |
| 9 | `investing/capital-03-quarter-end-policy-review.md` | `.md` | The chief executive and the brokerage |
| 10 | `diagrams/fig-07` .. `fig-09` | `.md` | The author |
| 11 | `packet/` | `.tex`, `.pdf`, `.zip` | Attached to letters 1, 2 and 3 |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform |
|:--|:--|:--|:--|
| 7 | Graphviz | Directed tree | Authority is a hierarchy, and a dot tree is the plainest way to draw one |
| 8 | D2 | Grid timetable | A weekday routine is a grid of hours against days |
| 9 | PlantUML | Activity diagram with swimlanes | A letter moves between parties in order, with one decision |

Mermaid and Diagrams are not used on day 3.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 11 | The three priorities, each with its operating rule and its evidence | 5.6 cm |
| 12 | The day's action register | 6.4 cm |
| 13 | The structure map: each authority, what it governs, and the company's place | 5.2 cm |
| 14 | The early-rise weekday, block by block | 5.8 cm |
| 15 | Decisions inside the Phase 1 and who holds each | 5.4 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Mission Oxblood `#7A1F2B`, with `#A85A63` and `#F3E4E6` as its two lighter shades |
| 2 | Addresses | Three per email, all in [`../../inputs`](../../inputs) |
| 3 | The message | Quoted verbatim, including its line breaks, and never paraphrased |
| 4 | The White House communications | Described as communications received, never as an award, an appointment, or a commitment, and never quoted, since their text is not in the repository |
| 5 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 6 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 7 | Dialect and punctuation | La Jolla usage; single hyphens only |
| 8 | Rasters | None |

## Commit order

| Order | Commit |
|:--|:--|
| 1 | This sub-prompt README |
| 2 | `30Sep26/README.md` |
| 3 | `30Sep26/emails/README.md` |
| 4 to 7 | The four letters, one commit each |
| 8 to 11 | `briefs/`, `forms/`, `investing/`, `diagrams/` |
| 12 to 15 | `packet/main.tex`, `fundstyle.sty`, `references.bib`, `packet/README.md` |
| 16 to 22 | `sec-00` through `sec-06`, one commit each |
| 23 | `packet/sections/README.md` |
| 24 | `main.pdf` and `30Sep26-packet-LaTeX.zip` |

## Rule 5 source map

| Used | From | Where it appears in day 3 |
|:--|:--|:--|
| [`../../prompts/prompt-auto-fund-2.md`](../../prompts/prompt-auto-fund-2.md) | This block | The September 12 message, quoted verbatim |
| `02Sep26/emails/email-06-nci-ctep-gore-reply.txt` | [`../../../auto-fund`](../../../auto-fund) | Letter 4's thread and the contributor-not-sponsor position |
| `04Sep26/emails/email-01-ucsd-moores-escalation.txt` | [`../../../auto-fund`](../../../auto-fund) | Letter 3's earlier contact |
| `08Sep26/briefs/brief-02-weekly-cadence.md` | [`../../../auto-fund`](../../../auto-fund) | The weekday themes in brief 3 and Figure 8 |
| `final-move-in/sections/sec-14-staffing-and-roles.tex` | [`../../../move-in`](../../../move-in) | Table 15, the sponsor-investigator firewall |
| `science-golden-age/` | [`../../../science-golden-age`](../../../science-golden-age) | The federal policy position letter 1 is written against |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
