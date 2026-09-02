# 02Sep26 / investing - the Treasury ladder instruction (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-1%20of%205-0E5C63.svg)](..)
[![Instruction sets](https://img.shields.io/badge/Instruction%20sets-1-0E5C63.svg)](.)
[![Instrument class](https://img.shields.io/badge/Class-U.S.%20Treasury-6C757D.svg)](.)
[![Equity](https://img.shields.io/badge/Equity-none%20authorized-9AA1A8.svg)](#what-this-day-does-not-authorize)
[![Advice](https://img.shields.io/badge/Investment%20advice-none-9AA1A8.svg)](#what-this-is-and-is-not)

One instruction set. It holds the reasoning, the instrument list, the order
types, the limits and the settlement notes. The message that transmits it to the
broker is
[`../emails/email-05-brokerage-treasury-instruction.txt`](../emails/email-05-brokerage-treasury-instruction.txt),
because a brokerage desk should receive a letter and not a repository file.

## What this is, and is not

This is the sole member's direction for the company's own corporate reserve. It
is **not** investment advice, it is **not** a recommendation to any other person,
and it is **not** an offer or solicitation of anything. Nothing in it has been
placed.

## Why a ladder, on this day, and not a sweep

The company's operating horizon for the coming period is nine months, matching
the duration of an SBIR Phase I award should one be made. A single sweep balance
earns a rate but expresses no view about when money is needed. A ladder expresses
exactly that view: each maturity is placed near a point at which the company has
a known cash need, so no position has to be sold before maturity to meet one.

The ladder is also a governance instrument. Each maturity is a decision point at
which the chief executive re-reads the horizon rather than letting an automatic
roll make the decision silently. That is why auto-reinvestment is switched off on
every rung.

## What this day does not authorize

| Not authorized | Why |
|:--|:--|
| Any equity purchase | The reserve funds nine months of operations. An equity sleeve inside an operating reserve converts an operating decision into a market decision |
| Margin, options, securities lending | A research company's reserve should carry no financing risk, no optionality, and no counterparty risk it was not paid to take |
| A prime money market fund | The yield difference over a government fund does not compensate for the gate and fee machinery a prime fund carries |
| Corporate credit of any maturity | Reserved for a later day, and only above a stated cash floor |
| Auto-reinvestment | A maturity is a decision, and an automatic roll removes it |

Day 2 revisits the reserve once the private capital question is decided, and it
is on day 2 that a non-Treasury sleeve is considered for the first time.

## The one file

| File | What it holds |
|:--|:--|
| [`capital-01-treasury-ladder.md`](capital-01-treasury-ladder.md) | Four rungs with maturity targets and CUSIP handling, one exchange-traded sleeve with a limit-order rule, the sweep residual, the settlement and tax notes, and the five checks the chief executive runs before approving |

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `applications/app-05-nih-sbir-seed/` | [`../../../pdac-funding-applications`](../../../pdac-funding-applications) | The nine-month horizon the ladder is cut to |
| `final-capital/sections/sec-04-capital-bridge.tex` | [`../../../capitalization-plan`](../../../capitalization-plan) | The tier structure the reserve sits beneath |
| `final-move-in/sections/sec-15-funding-and-lobbying.tex` | [`../../../move-in`](../../../move-in) | The separation of federal from non-federal funds, which is why award funds are never mixed into the reserve |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
