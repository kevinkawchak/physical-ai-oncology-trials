# 03Sep26 / investing - the reserve re-cut against the financing decision (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-2%20of%205-1B3A5C.svg)](..)
[![Instruction sets](https://img.shields.io/badge/Instruction%20sets-1-1B3A5C.svg)](.)
[![New classes](https://img.shields.io/badge/New%20instrument%20classes-2-6C757D.svg)](#what-day-2-adds-that-day-1-excluded)
[![Equity](https://img.shields.io/badge/Operating%20equity-still%20none-9AA1A8.svg)](.)
[![Advice](https://img.shields.io/badge/Investment%20advice-none-9AA1A8.svg)](.)

One instruction set, and it is a **conditional** one. Day 1 cut the reserve to a
nine-month horizon and excluded everything except direct Treasury obligations.
This day says what changes if a financing is taken, and what stays exactly as it
is if one is not.

## Why a conditional instruction and not a new allocation

The reserve's horizon is a function of how long the company has to operate on its
own capital. A financing changes that horizon. Re-cutting the reserve before the
instrument is chosen would be re-cutting it against a guess, and a ladder built
against a guess has to be broken when the guess is wrong, which is the exact cost
a ladder exists to avoid.

So this instruction has two branches, and the chief executive takes one of them
after the day's single approval step, not before.

## What day 2 adds that day 1 excluded

Two instrument classes become available **only on the financed branch**, and both
are still conservative:

| Class | Available on | Why it was excluded on day 1 |
|:--|:--|:--|
| Treasury notes out to 24 months | The financed branch only | A 24-month maturity against a 9-month horizon is a position that has to be sold early |
| Agency discount notes and agency-backed government funds | The financed branch only | The spread over Treasury bills is small and does not pay for the added complexity at a 9-month horizon |

Corporate credit, equity, margin, options and securities lending remain excluded
on **both** branches. A financing does not change what a research company's
reserve is for.

## The one file

| File | What it holds |
|:--|:--|
| [`capital-02-corporate-reserve-allocation.md`](capital-02-corporate-reserve-allocation.md) | The two branches, the allocation on each, the segregation rule for subscription funds, the runway arithmetic, and the six checks before either branch is taken |

## The segregation rule, stated here because it governs everything else

Subscription funds, if any arrive, are held in a **separate account under the
same entity** and are not commingled with the operating reserve. Federal award
funds, if any arrive, are held and drawn under the award's own terms and are
mixed with neither. Three sources of money, three treatments, and the separation
is established before the first dollar rather than reconstructed afterward.
[`../emails/email-05-brokerage-corporate-account.txt`](../emails/email-05-brokerage-corporate-account.txt)
is the letter that puts the second account in place.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `../../02Sep26/investing/capital-01-treasury-ladder.md` | [`../../02Sep26`](../../02Sep26) | The unfinanced branch, which is that instruction unchanged |
| `../briefs/brief-01-instrument-comparison.md` | This day | The financed branch's horizon assumption |
| `../briefs/brief-03-use-of-proceeds.md` | This day | The draw schedule the financed branch is cut against |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | The three-source segregation rule |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
