# 29Sep26 - Day 2, The External Standard Request (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-34495E.svg)](.)
[![Approval steps](https://img.shields.io/badge/Approval%20steps-1-34495E.svg)](#the-one-approval-step)
[![Emails](https://img.shields.io/badge/Emails-5-6C757D.svg)](emails)
[![Briefs](https://img.shields.io/badge/Briefs-3-6C757D.svg)](briefs)
[![Form packs](https://img.shields.io/badge/Form%20packs-1-6C757D.svg)](forms)
[![Capital](https://img.shields.io/badge/Capital%20set-1-6C757D.svg)](investing)
[![Figures](https://img.shields.io/badge/Figures-3-9AA1A8.svg)](diagrams)
[![Tables](https://img.shields.io/badge/Tables-5-9AA1A8.svg)](packet)
[![Packet](https://img.shields.io/badge/Packet-The%20External%20Standard%20Request-34495E.svg)](packet)
[![Quote](https://img.shields.io/badge/SEC%20words-%22very%20seriously%22-9AA1A8.svg)](#what-the-office-said-and-what-it-did-not)

The SEC San Francisco Regional Office has written to the chief executive that it
is taking his request "very seriously." The request is that ChemicalQDevice's
oncology clinical trial research be listed as an external standard for
substantial oncology clinical trial research, because the company's LLM papers
have been used extensively by OpenAI and Anthropic.

This day answers that update with the record behind the request.

## The one approval step

> **Approve answering the SEC office with the full dated record, routing the
> clinical-standard question to the FDA's oncology AI program, and notifying the
> three companies the request names or concerns.**

## What the office said, and what it did not

| The office said | The office did not say |
|:--|:--|
| That it is taking the request "very seriously" | That it has reached any view on the request |
| | That it has authority to list a clinical research standard |
| | That the company or its papers are endorsed, listed, or approved |

Every file in this directory quotes the office's two words exactly and adds none
to them. A regulator's courtesy that is later described as a finding is the
fastest way to lose the regulator's attention, and a company that writes about
the SEC in a public repository must be more careful than one that does not.

## Why the record matters more than the argument

The request rests on two facts a reader can check. The first is the company's
deposited record: eleven works on this program with persistent identifiers,
dated from June 2025 to September 2026. The second is that the record was built independently of,
and simultaneously with, Revolution Medicines' own Phase 1/2 and Phase 3 trials
of daraxonrasib, which the FDA approved on August 26, 2026 as Rasonque. A
reviewer who checks both facts needs no argument. A reviewer who cannot check
them will not be moved by one.

| Period | Revolution Medicines, public record | ChemicalQDevice, deposited record |
|:--|:--|:--|
| 2022 onward | Phase 1/1b RMC-6236 study in RAS-mutant solid tumors | No work on the molecule |
| Late 2024 to 2026 | RASolute 302, Phase 3, enrolling and then reported | June 2025 identification; August 2025 QSP simulation |
| June to July 2026 | RASolute 302 reported | Phase 1 protocol, Phase 2 protocol, IND, two funding applications |
| August to September 2026 | FDA approval as Rasonque; RASolute 304 enrolling | Ten applications, capitalization plan, La Jolla site package |

## The run order

| Order | Item | Where | Depends on |
|:--|:--|:--|:--|
| 1 | Reply to the SEC office with the record | [`emails/email-01-sec-san-francisco-reply.txt`](emails/email-01-sec-san-francisco-reply.txt) | Nothing |
| 2 | Write to the FDA's Oncology AI Program | [`emails/email-02-fda-oce-oncology-ai.txt`](emails/email-02-fda-oce-oncology-ai.txt) | Letter 1 sent, so letter 2 can say so |
| 3 | Notify Anthropic | [`emails/email-03-anthropic-notice.txt`](emails/email-03-anthropic-notice.txt) | Letter 1 sent |
| 4 | Notify OpenAI | [`emails/email-04-openai-notice.txt`](emails/email-04-openai-notice.txt) | Letter 1 sent |
| 5 | Notify Revolution Medicines | [`emails/email-05-revolution-medicines-notice.txt`](emails/email-05-revolution-medicines-notice.txt) | Letter 1 sent |
| 6 | Adopt the trading and disclosure guardrails | [`investing/capital-02-trading-and-disclosure-guardrails.md`](investing/capital-02-trading-and-disclosure-guardrails.md) | Before letter 5 |
| 7 | Prepare, and hold, EDGAR access | [`forms/form-01-edgar-access-readiness.md`](forms/form-01-edgar-access-readiness.md) | Nothing; filed only before a first filing |

Row 6 comes before letter 5 in practice. A company that writes to a public
company's investor relations office should have its own trading rule in place
before the letter leaves, not after.

## Directory contents

```
29Sep26/
├── README.md              this approval sheet
├── emails/                5 .txt letters, each with at least three verified addresses
├── briefs/                3 .md briefs: the record, the criteria, the OpenAI and Anthropic statement
├── forms/                 1 .md pack: EDGAR access readiness, held
├── investing/             1 .md capital instruction: trading and disclosure guardrails
├── diagrams/              3 .md figure specifications
└── packet/                The External Standard Request: main.tex, fundstyle.sty,
                           references.bib, sections/sec-00 .. sec-06.tex,
                           main.pdf, 29Sep26-packet-LaTeX.zip
```

## Rule 5 source map

| Used | From | Where it appears in this day |
|:--|:--|:--|
| `tripartisan-llm-support.md` | [`../..`](../..) | Brief 3 and letters 3 and 4 |
| `daraxonrasib-llm-story.md` | [`../..`](../..) | The period table above, Table 6 and Figure 5 |
| Root `README.md` release list | [`../../../README.md`](../../../README.md) | Every date in the record |
| `final-capital/sections/sec-06-clinical-evidence.tex` | [`../../capitalization-plan`](../../capitalization-plan) | Table 10 |
| `03Sep26/forms/form-01-reg-d-506b-form-d.md` | [`../../auto-fund`](../../auto-fund) | The reason EDGAR access is prepared |
| `02Sep26/investing/capital-01-treasury-ladder.md` | [`../../auto-fund`](../../auto-fund) | The Treasury-only rule the guardrails keep |

## Positioning, carried into every file in this directory

The SEC office's words are "very seriously," and they are a statement about a
request, not a determination, listing, or endorsement. The statement that the
company's LLM papers have been used extensively by OpenAI and Anthropic is the
company's own, made as the basis of its request; no agreement, sponsorship, or
endorsement exists with either company, and letters 3 and 4 invite each to
correct it. No agreement of any kind exists with Revolution Medicines. The
company has no public securities, no offering is underway, and nothing in
[`investing/`](investing) is investment advice.

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
