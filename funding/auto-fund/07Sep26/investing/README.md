# 07Sep26 / investing - orders queued against a closed market (v4.8.0)

[![Repository](https://img.shields.io/badge/Repository-v4.8.0-00417A.svg)](../../../../README.md)
[![Day](https://img.shields.io/badge/Day-4%20of%205-5B3A5E.svg)](..)
[![Instruction sets](https://img.shields.io/badge/Instruction%20sets-1-5B3A5E.svg)](.)
[![Market](https://img.shields.io/badge/NYSE%20%2F%20Nasdaq-closed-9AA1A8.svg)](#why-nothing-is-entered-today)
[![Entered](https://img.shields.io/badge/Orders%20entered-none-9AA1A8.svg)](.)
[![Advice](https://img.shields.io/badge/Investment%20advice-none-9AA1A8.svg)](.)

One instruction set. It queues orders that are entered on the next open session
and nowhere today.

## Why nothing is entered today

The New York Stock Exchange and Nasdaq are closed for the federal holiday. An
order entered against a closed session does one of two things, and both are bad:
it rejects, or it rests until the open and fills at a price formed in the first
minutes of a session nobody has seen. The second is worse, because it looks like
a normal fill.

The Treasury auction calendar is also affected. Auctions do not settle on a
federal holiday, and an auction schedule read from a page written before the
holiday will be wrong. The instruction says to re-read the schedule on the next
session rather than to trust a figure carried over.

## What "queued" means precisely

| Term | Meaning here |
|:--|:--|
| Queued | Written down, sized, and priced in this repository. Present in no broker system |
| Entered | Transmitted to the broker. Happens on day 5, after the release approval |
| Limit price | Set from the **last completed session's** close, and re-checked on the morning of entry |
| Time in force | Day. Never good-till-canceled on an instruction written a session in advance |

The last row is the one worth stating twice. A good-till-canceled order written
before a closed session can fill days later at a price the instruction never
contemplated, which converts a queued order into a standing exposure.

## The one file

| File | What it holds |
|:--|:--|
| [`capital-04-queued-orders.md`](capital-04-queued-orders.md) | The order table with instrument, side, type, limit basis, size and time in force; the branch condition each order sits under; the auction re-check; and the six checks before any of it is entered |

## The branch this inherits

Day 2 left the reserve on two branches: unfinanced at a nine-month horizon, or
financed at twenty-four months. **No instrument has been selected**, so every
order queued here sits under branch A and is marked as such. If the financing
decision changes before the next session, the queue is re-cut before it is
entered rather than entered and amended.

## Rule 5 source map

| Used | From | Where it appears here |
|:--|:--|:--|
| `../../02Sep26/investing/capital-01-treasury-ladder.md` | [`../../02Sep26`](../../02Sep26) | The instrument list and the constraint set |
| `../../03Sep26/investing/capital-02-corporate-reserve-allocation.md` | [`../../03Sep26`](../../03Sep26) | The branch condition every queued order sits under |
| `../../04Sep26/investing/capital-03-site-startup-reserve.md` | [`../../04Sep26`](../../04Sep26) | The site start-up pool, which is queued separately and is not laddered |

---

Kevin Kawchak, CEO ChemicalQDevice,
[kevink@chemicalqdevice.com](mailto:kevink@chemicalqdevice.com),
ORCID [0009-0007-5457-8667](https://orcid.org/0009-0007-5457-8667).
