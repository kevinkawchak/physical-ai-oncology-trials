# 07Sep26 / emails - four letters, written and held (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](..)
[![Letters](https://img.shields.io/badge/Letters-4-5B3A5E.svg)](.)
[![Status](https://img.shields.io/badge/Status-HOLD%20FOR%20RELEASE-9AA1A8.svg)](#the-hold-line)
[![Format](https://img.shields.io/badge/Format-.txt-6C757D.svg)](.)
[![Sent today](https://img.shields.io/badge/Sent%20today-none-9AA1A8.svg)](.)

Four letters, complete, and none of them sent. Every one carries a
`HOLD FOR RELEASE` line as the first line of its file, naming the next open
session as its earliest send time.

## The hold line

Every file in this directory begins with:

```
HOLD FOR RELEASE: not before the next open federal business session.
```

That line is the first thing in the file and not a footnote, because the failure
mode it prevents is a fast one: a letter that is finished looks sendable, and a
letter sent into a federal holiday lands at the bottom of the following day's
stack behind everything that arrived while the office was closed.

## Why writing on a closed day is not wasted

A letter written under time pressure on an open day is a worse letter. Day 5
opens with five things to do and a market session that closes at a fixed hour;
writing a Pre-Request for Designation inquiry inside that window would produce
the version of it that fits the window rather than the version that is right.

## The four letters

| # | File | To | What it asks |
|:--|:--|:--|:--|
| 1 | [`email-01-congressional-delegation.txt`](email-01-congressional-delegation.txt) | District health staff for the San Diego congressional delegation | An informational meeting about a federal review clock, with the lobbying boundary stated in the letter |
| 2 | [`email-02-california-ibank-inquiry.txt`](email-02-california-ibank-inquiry.txt) | The state small business finance center | Whether any state program fits a pre-revenue clinical research company |
| 3 | [`email-03-san-diego-economic-development.txt`](email-03-san-diego-economic-development.txt) | The regional economic development corporation | Introductions and a view on regional life science programs |
| 4 | [`email-04-fda-combination-products-pre-rfd.txt`](email-04-fda-combination-products-pre-rfd.txt) | The FDA Office of Combination Products | A preliminary classification and lead center determination |

## Letter 4 is the most consequential item in the block

A Pre-Request for Designation determines which FDA center leads the review of a
system that is a drug, a robotic platform, and an advisory software component at
once. Every regulatory assumption downstream of it depends on the answer:
which submissions are expected, in what order, and under whose authority.

It is written on the quietest day of the block deliberately, because it is the
one letter in the whole block that should not be written between two other
things.

## The lobbying boundary in letter 1

Letter 1 is the only one in the five-day block addressed to elected
representatives, and it carries a boundary statement in its own body rather than
only in its checklist. The rule is not a preference: the same activity is lawful
with one source of funds and unlawful with another, and no federal award exists
in any case.

| Paid from | Never paid from |
|:--|:--|
| Company funds or private capital, where lawful and disclosed | Federal award funds, for lobbying of any kind |

## The shared structure

Identical to the other days, with one addition at the top:

`HOLD FOR RELEASE`, then `FROM`, `TO`, `CC`, `SUBJECT`, an introduction, a body,
a closing, an attachment manifest split into what compiles from this repository
and what the author adds by hand, external work to cite but not attach, and a
pre-send checklist whose first item is always the release condition.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `UC-San-Diego/priority-steps.md` §10 | [`../../../potential-partners`](../../../potential-partners) | Letter 4's four addresses, its subject line, and its eight component descriptions |
| `UC-San-Diego/priority-steps.md` §11 | [`../../../potential-partners`](../../../potential-partners) | Letter 4's note on parallel drug and device meeting packages |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | Letter 1's lobbying boundary, in the body and in the checklist |
| `applications/app-01/email-app-01-nih-pioneer-award.txt` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The file structure every letter here reuses |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
