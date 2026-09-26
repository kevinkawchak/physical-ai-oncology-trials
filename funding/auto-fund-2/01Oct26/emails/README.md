# 01Oct26 / emails - four letters on the NSF pilot inquiry (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-4B2A7B.svg)](..)
[![Letters](https://img.shields.io/badge/Letters-4-4B2A7B.svg)](.)
[![Format](https://img.shields.io/badge/Format-.txt-6C757D.svg)](.)
[![Addresses](https://img.shields.io/badge/Addresses-3%20per%20letter-6C757D.svg)](#the-four-letters)
[![Paste](https://img.shields.io/badge/iOS%20Mail-one%20paragraph%20per%20line-9AA1A8.svg)](#how-to-send-one)

Four letters. The first answers the NSF inquiry; the second confirms the pilot's
gate with the person who runs it; the third asks about shared AI research
resources; the fourth tells NIH about the NSF pitch before either agency could
wonder about overlap.

## The four letters

| # | File | TO | CC | Purpose |
|:--|:--|:--|:--|:--|
| 1 | [`email-01-nsf-sbir-inquiry-reply.txt`](email-01-nsf-sbir-inquiry-reply.txt) | `sbir@nsf.gov` | `policy@nsf.gov`, `rgov@nsf.gov` | Thank NSF, state the eligibility position exactly, and ask whether the inquiry contemplates a route the company has missed |
| 2 | [`email-02-ncti-eligibility.txt`](email-02-ncti-eligibility.txt) | `Ivan.Garibay@ucf.edu` | `sbir@nsf.gov`, `policy@nsf.gov` | Confirm the Commercialization Readiness Pilot's gate with its principal investigator |
| 3 | [`email-03-nairr-pilot-resources.txt`](email-03-nairr-pilot-resources.txt) | `NAIRR_Pilot@nsf.gov` | `nairr-oc@nsf.gov`, `sbir@nsf.gov` | Ask whether a small company's verification work can use NAIRR Pilot resources, on synthetic data only |
| 4 | [`email-04-nih-sbir-overlap-disclosure.txt`](email-04-nih-sbir-overlap-disclosure.txt) | `sbir@od.nih.gov` | `ncisbir@mail.nih.gov`, `sbir@nsf.gov` | Disclose the NSF pitch to NIH and state how the two scopes are kept apart |

## Why the Policy Office and the Research.gov help desk are copied on letter 1

Eligibility for an NSF program is a policy question, and the Policy Office owns
the answer. Submitting a Project Pitch and, later, a proposal runs through
Research.gov, and the company's organization record there is part of what the
letter asks about. Each copied office owns one part of the letter.

## Why letter 4 exists

A company that pitches the same program to two agencies must be able to show,
from its own letters, that the two scopes do not overlap. SBIR rules prohibit
funding essentially equivalent work twice. Telling NIH now, in writing, with the
NSF SBIR office copied, is cheaper than explaining later.

## How to send one

1. Open the `.txt` file on GitHub, select everything between `=== BODY ===` and `=== END BODY ===`, and copy it.
2. Paste into a new message in iOS Mail. Each paragraph is one line in the file, so the phone wraps it to the screen with no stray breaks.
3. Copy the `TO`, `CC` and `SUBJECT` lines into their fields separately.
4. Attach the files named under `ATTACHMENTS COMPILED FROM THIS DIRECTORY`.
5. Work through `BEFORE SENDING` line by line.

## What no letter here says

| It does not say | Because |
|:--|:--|
| That the company is eligible for the pilot | It is not, until it holds an NSF Phase II |
| That the inquiry was an invitation to submit a proposal | An inquiry to apply is not a proposal invitation |
| That NSF would fund the clinical trial | NSF's SBIR program does not fund clinical trials |
| That any patient data would reach a shared resource | Only synthetic data ever leaves the site |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `inputs/README.md` | [`../../inputs`](../../inputs) | Every address and every NSF quantity |
| `../../28Sep26/emails/email-01-nih-sbir-contingent-personnel.txt` | [`../../28Sep26/emails`](../../28Sep26/emails) | The NIH thread letter 4 continues |
| `tripartisan-llm-support.md` | [`../../..`](../../..) | The verification method in letters 1 and 3 |
| `trial-ind/` | [`../../../../trial-ind`](../../../../trial-ind) | The advisory-only boundary in letter 3 |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
