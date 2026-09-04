# 08Sep26 / investing - execution and settlement (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-5%20of%205-8A4B2A.svg)](..)
[![Instruction sets](https://img.shields.io/badge/Instruction%20sets-1-8A4B2A.svg)](.)
[![Orders](https://img.shields.io/badge/Orders-6-6C757D.svg)](capital-05-execution-and-settlement.md)
[![Checks](https://img.shields.io/badge/Pre--entry%20checks-6-9AA1A8.svg)](.)
[![Advice](https://img.shields.io/badge/Investment%20advice-none-9AA1A8.svg)](.)

One instruction set: what happens between the queue and a confirmed position.

## What this day adds that day 4 did not

Day 4 sized and priced six orders and entered none. This day adds the four things
that only exist once an order is actually transmitted: the entry sequence, the
settlement arithmetic, the record that has to be written on the day of the fill,
and the reconciliation that closes the loop.

| Added here | Why it could not be written on a closed day |
|:--|:--|
| The entry sequence | It depends on the auction calendar as re-read this morning |
| Settlement dates | An auction that has not been announced has no settlement date |
| The fill record | There is nothing to record until there is a fill |
| Reconciliation | The reserve balance is confirmed this morning, not last week |

## The six checks that precede any entry

Restated here because they are the whole of the discipline: the market is open,
confirmed against the calendar; the release approval was given; the financing
branch is unchanged; the auction calendar has been re-read; the limit price was
written this morning; and the reserve balance is current.

Any one of them failing stops the entry. The fourth is the one most likely to be
skipped and the one most likely to matter, because auctions do not settle on a
federal holiday and a calendar read before one will be wrong for the week after
it.

## The one file

| File | What it holds |
|:--|:--|
| [`capital-05-execution-and-settlement.md`](capital-05-execution-and-settlement.md) | The entry sequence, the settlement basis for each line, the fill record fields, the reconciliation, and what is done if a line does not fill |

## What is still not authorized

No margin, no options, no securities lending, no prime money market fund, no
corporate credit, no equity, no auto-reinvestment, and no good-till-canceled
order. Execution day does not relax a single constraint set on day 1; it only
carries them into a broker instruction.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `../../07Sep26/investing/capital-04-queued-orders.md` | [`../../07Sep26`](../../07Sep26) | The six orders and the six pre-entry checks |
| `../../02Sep26/investing/capital-01-treasury-ladder.md` | [`../../02Sep26`](../../02Sep26) | The constraint set and the settlement notes |
| `../../03Sep26/investing/capital-02-corporate-reserve-allocation.md` | [`../../03Sep26`](../../03Sep26) | The branch condition confirmed before entry |
| `../../04Sep26/investing/capital-03-site-startup-reserve.md` | [`../../04Sep26`](../../04Sep26) | The pool that is deliberately not part of this entry |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
