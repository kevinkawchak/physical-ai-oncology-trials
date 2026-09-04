# Sub-prompt 2 - 03Sep26, The Private Capital Bridge (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-1B3A5C.svg)](../../03Sep26)
[![Accent](https://img.shields.io/badge/Accent-Harbor%20Navy%20%231B3A5C-1B3A5C.svg)](../../03Sep26/packet)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](../../03Sep26/emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](../../03Sep26/briefs)
[![Forms](https://img.shields.io/badge/Form%20packs-2-6C757D.svg)](../../03Sep26/forms)
[![Capital](https://img.shields.io/badge/Capital%20sets-1-6C757D.svg)](../../03Sep26/investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](../../03Sep26/diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](../../03Sep26/packet)
[![Commits](https://img.shields.io/badge/Commits-24%2B-9AA1A8.svg)](#commit-order)

Day 1 asked the federal side for the $1,606,000 the SBIR route can supply. This
day addresses the $2,104,000 it cannot, and the $5,900,000 private position built
behind the 21 CFR part 54 firewall. It also opens the one commercial relationship
the approval makes newly askable.

## The single decision this day asks for

**Does the chief executive approve one instrument, one raise size, and one first
approach to the drug developer?**

The three candidate instruments are set out side by side at the same raise size
so the choice is a comparison rather than a judgment call, and the developer
letter is written to ask for a conversation rather than for a supply agreement.

## Why the developer letter belongs on day 2 and not day 1

`../../../potential-partners/UC-San-Diego/priority-steps.md` §2 records the
request that was prepared for Revolution Medicines and states that the public
external-research portal indicated submissions were closed. The approval changes
the company's posture, not the portal's calendar, so this letter asks for the
next submission window and a scientific feasibility conversation and asks for
nothing that a closed window would refuse. It is placed after the federal
re-contacts because a developer reasonably asks who is funding the work, and day
1 is the answer to that question.

## What this day produces

| # | Deliverable | Format | Recipient class |
|:--|:--|:--|:--|
| 1 | `emails/email-01-revolution-medicines-external-research.txt` | `.txt` | Developer external research, grants, medical information, business development |
| 2 | `emails/email-02-san-diego-angel-syndicate.txt` | `.txt` | Regional angel and syndicate leads |
| 3 | `emails/email-03-life-science-family-office.txt` | `.txt` | Family office and evergreen holders |
| 4 | `emails/email-04-securities-counsel-engagement.txt` | `.txt` | Securities counsel |
| 5 | `emails/email-05-brokerage-corporate-account.txt` | `.txt` | The broker-dealer, corporate account desk |
| 6 | `briefs/brief-01-instrument-comparison.md` | `.md` | An investor's counsel or analyst |
| 7 | `briefs/brief-02-firewall-and-part-54.md` | `.md` | A reviewer testing the conflict boundary |
| 8 | `briefs/brief-03-use-of-proceeds.md` | `.md` | Any holder asking where the money goes |
| 9 | `forms/form-01-reg-d-506b-form-d.md` | `.md` | SEC EDGAR Form D |
| 10 | `forms/form-02-california-25102f-notice.md` | `.md` | California Department of Financial Protection and Innovation |
| 11 | `investing/capital-02-corporate-reserve-allocation.md` | `.md` | The chief executive and the broker |
| 12 | `diagrams/fig-04` .. `fig-06` | `.md` | The author, when a figure needs correction |
| 13 | `packet/` | `.tex`, `.pdf`, `.zip` | Every recipient above, as the attachment |

## The three figures, and why each platform

| Figure | Platform | Native construct | Why this platform and no other |
|:--|:--|:--|:--|
| 4 | D2 | Container grid, three columns | Three instruments compared on the same eight attributes is a column comparison, and D2's container grid is the only vocabulary that keeps the columns aligned without arrows |
| 5 | PlantUML | State machine with guards | A financing has states and the transitions between them carry conditions; a guard is a first-class PlantUML construct and is not one in the other four |
| 6 | Mermaid | Sequence with lifelines | Signing order across four parties is a sequence, and only Mermaid's sequence form shows both the order and who is idle between messages |

Graphviz and Diagrams are unused on day 2 and appear on days 1, 3, 4 and 5.

## The five tables in the packet

| Table | Subject | Widest column |
|:--|:--|:--|
| 6 | Three instruments at the same raise size, eight attributes each | 3.2 cm |
| 7 | The $5,900,000 position by tranche, with dilution at each | 2.8 cm |
| 8 | Use of proceeds against the $2,104,000 the SBIR route does not buy | 4.4 cm |
| 9 | The 21 CFR part 54 triggers and which are live under each instrument | 4.0 cm |
| 10 | Corporate reserve allocation with instrument, size, and liquidity horizon | 3.0 cm |

## Invariants restated for this day

| # | Invariant | This day's value |
|:--|:--|:--|
| 1 | Accent color | Harbor Navy `#1B3A5C`, with `#3E6F9E` and `#DCE5EE` as its two lighter shades |
| 2 | Caption spacing | `\vspace{-0.60cm}`, 7.44 pt from rule to first caption line |
| 3 | Caption lines | Two, balanced within a small character spread |
| 4 | Table measure | `\textwidth` exactly, every fixed column `>{\raggedright\arraybackslash}p{...}` |
| 5 | Money | The frame in [`../../inputs`](../../inputs), not re-derived |
| 6 | Securities language | Every instrument is described as a candidate under consideration; no offer, solicitation, or commitment is made anywhere |
| 7 | Dialect | American English, La Jolla usage |
| 8 | Punctuation | Single hyphens only |
| 9 | Rasters | None |

Invariant 6 is specific to this day and is load-bearing. A public repository
describing a private placement has to be readable as a plan and not as an offer.
Every file in [`../../03Sep26`](../../03Sep26) carries that sentence, and the
packet carries it on the cover.

## Commit order

Identical in shape to day 1: the day README, the emails README, one commit per
email, then briefs, forms, investing and diagrams, then `main.tex`,
`fundstyle.sty`, `references.bib` and the packet README on separate commits, then
one commit per section, then the sections README, then the compiled PDF and the
Overleaf zip.

## Rule 5 source map

| Used | From | Where it appears in day 2 |
|:--|:--|:--|
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | Tables 6, 7 and 8, and Figures 4 and 5 |
| `final-capital/sections/sec-09-risks-and-limits.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The cost-of-disclosure paragraph in `briefs/brief-03-use-of-proceeds.md` |
| `UC-San-Diego/priority-steps.md` §2 | [`../../../potential-partners`](../../../potential-partners) | Email 01's addresses, its seven requests, and its attachment limit |
| `UC-San-Diego/priority-steps.md` §12 | [`../../../potential-partners`](../../../potential-partners) | The sponsor and IND-holder questions in `briefs/brief-02-firewall-and-part-54.md` |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | The federal versus non-federal separation quoted in `briefs/brief-03-use-of-proceeds.md` |
| `final-capital/references.bib` | [`../../../capitalization-plan`](../../../capitalization-plan) | The 21 CFR part 54 and SBIR ownership entries carried into `references.bib` |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
