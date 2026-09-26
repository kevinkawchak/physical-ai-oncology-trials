# 29Sep26 / emails - one answer, one routing letter, three notices (v4.9.0)

[![Repository](https://img.shields.io/badge/Repository-v4.9.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-34495E.svg)](..)
[![Letters](https://img.shields.io/badge/Letters-5-34495E.svg)](.)
[![Format](https://img.shields.io/badge/Format-.txt-6C757D.svg)](.)
[![Addresses](https://img.shields.io/badge/Addresses-3%20per%20letter-6C757D.svg)](#the-five-letters)
[![Paste](https://img.shields.io/badge/iOS%20Mail-one%20paragraph%20per%20line-9AA1A8.svg)](#how-to-send-one)

Five letters. The first answers the SEC office. The second routes the clinical
question to the federal office that owns it. The last three tell the companies
the request names or concerns, before they hear it anywhere else.

## The five letters

| # | File | TO | CC | Purpose |
|:--|:--|:--|:--|:--|
| 1 | [`email-01-sec-san-francisco-reply.txt`](email-01-sec-san-francisco-reply.txt) | `sanfrancisco@sec.gov` | `losangeles@sec.gov`, `smallbusiness@sec.gov` | Thank the office, give the dated record, and state exactly what is and is not asked |
| 2 | [`email-02-fda-oce-oncology-ai.txt`](email-02-fda-oce-oncology-ai.txt) | `OncologyAI@fda.hhs.gov` | `FDAOncology@fda.hhs.gov`, `combination@fda.gov` | Ask which federal venue owns a question about an external standard for AI-developed oncology trial documents |
| 3 | [`email-03-anthropic-notice.txt`](email-03-anthropic-notice.txt) | `press@anthropic.com` | `Legal@anthropic.com`, `sanfrancisco@sec.gov` | Tell Anthropic it is named in the request, and invite a correction |
| 4 | [`email-04-openai-notice.txt`](email-04-openai-notice.txt) | `press@openai.com` | `support@openai.com`, `sanfrancisco@sec.gov` | Tell OpenAI it is named in the request, and invite a correction |
| 5 | [`email-05-revolution-medicines-notice.txt`](email-05-revolution-medicines-notice.txt) | `IR@revmed.com` | `media@revmed.com`, `medinfo@revmed.com` | Tell Revolution Medicines that independent work on its molecule is referenced, with no affiliation claimed |

## Why the SEC office is copied on letters 3 and 4, and not on letter 5

Letters 3 and 4 concern a statement the company made to the SEC office about
OpenAI and Anthropic. Copying the office means the office holds exactly the text
each company was sent, and if either company corrects the statement, the
correction reaches the office at the same moment. That protects the record from
both directions.

Letter 5 is different. Revolution Medicines is a public company, and a copy of a
letter to its investor relations office sent to an SEC regional office could be
read as an attempt to involve the regulator in that company's affairs. The
request is about ChemicalQDevice's papers, not about Revolution Medicines, and
letter 5 says so in its second paragraph.

## Why Los Angeles is copied on letter 1

ChemicalQDevice is in San Diego, which falls within the Los Angeles Regional
Office's region. The update came from San Francisco and the reply goes there,
but the office with geographic responsibility for the company should not learn
of the correspondence secondhand. The Office of the Advocate for Small Business
Capital Formation is copied because the request comes from a small business
with a capital formation question behind it.

## How to send one

1. Open the `.txt` file on GitHub, select everything between `=== BODY ===` and `=== END BODY ===`, and copy it.
2. Paste into a new message in iOS Mail. Each paragraph is one line in the file, so the phone wraps it to the screen with no stray breaks.
3. Copy the `TO`, `CC` and `SUBJECT` lines into their fields separately.
4. Attach the files named under `ATTACHMENTS COMPILED FROM THIS DIRECTORY`.
5. Work through `BEFORE SENDING` line by line.

## The two rules every letter here keeps

| Rule | Why |
|:--|:--|
| The SEC office's words are "very seriously," quoted and never extended | A courtesy described as a finding ends the correspondence |
| The OpenAI and Anthropic statement is the company's own, and each company is invited to correct it | A third party's name in a regulatory request is a representation about that party |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `tripartisan-llm-support.md` | [`../../..`](../../..) | The model roles described in letters 3 and 4 |
| `daraxonrasib-llm-story.md` | [`../../..`](../../..) | The chronology in letters 1 and 5 |
| Root `README.md` release list | [`../../../../README.md`](../../../../README.md) | The dated record in letter 1 |
| `02Sep26/emails/email-06-nci-ctep-gore-reply.txt` | [`../../../auto-fund`](../../../auto-fund) | The commitment-paragraph pattern each letter closes with |
| `inputs/README.md` | [`../../inputs`](../../inputs) | Every address in this directory |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevinkawchak@gmail.com](mailto:kevinkawchak@gmail.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
