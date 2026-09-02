# Capital Instruction 04: The Queue, Entered Nowhere

**ChemicalQDevice LLC corporate brokerage account.** Direction of the sole member
and chief executive, queued for the next open session. Not investment advice, not
a recommendation to any other person, and not an offer or solicitation. **No
order below has been entered in any system.**

---

## The release condition

Every order in this file is entered **on the next open market session, after the
release approval of this day**, and not before. Nothing here is transmitted to a
broker today.

## The branch condition

No instrument has been selected in the financing decision, so every order below
sits under **branch A**, the unfinanced nine-month horizon from day 1. If an
instrument is selected before the next session, the queue is re-cut against
branch B before it is entered, rather than entered and then amended.

## The queue

| # | Instrument | Side | Type | Limit basis | Size | Time in force |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | U.S. Treasury bill, about 3 months | Buy | Non-competitive at auction | Not applicable | 20 pct of reserve | Auction |
| 2 | U.S. Treasury bill, about 6 months | Buy | Non-competitive at auction | Not applicable | 20 pct of reserve | Auction |
| 3 | U.S. Treasury bill, about 9 months | Buy | Non-competitive at auction | Not applicable | 20 pct of reserve | Auction |
| 4 | U.S. Treasury note or bill, about 12 months | Buy | Limit, secondary market | Last session's close, plus one tick | 20 pct of reserve | Day |
| 5 | Short-duration Treasury ETF, one fund only | Buy | **Limit** | Last session's close, plus one cent | 15 pct of reserve | Day |
| 6 | Government money market sweep | Hold | Sweep | Not applicable | 5 pct residual | Continuous |

## The auction re-check, which is not optional

Treasury auctions do not settle on a federal holiday, and an auction calendar
read before one will be wrong for the following week. On the next open session,
before any of orders 1 to 3 is placed, re-read the auction announcements and
results page and confirm three things: the next auction date for each tenor, its
settlement date, and the broker's own cutoff time relative to that auction.

A non-competitive bid submitted after a broker's internal cutoff is not submitted.

## Line 5, the only order with market price risk

| Item | Value |
|:--|:--|
| Candidate instruments | A short-duration U.S. Treasury exchange-traded fund. **SGOV** (0 to 3 month bills), **BIL** (1 to 3 month bills), and **SHV** (under 1 year) are three widely held examples |
| Selection rule | One fund, not three. Prefer the lowest total expense ratio available in the account with no transaction fee |
| Order type | **Limit, day. Never market, and never good-till-canceled** |
| Limit price | The last completed session's closing price plus one cent, written into the broker instruction on the morning of entry |
| Do not enter | In the first or last fifteen minutes of the session |
| Size | Rounded down to a whole share count |

A limit written before a closed weekend and a holiday is three days stale by the
time the market opens. The rule is that the number is written on the morning of
entry, and the queue records the **basis** rather than a figure.

## Constraints, unchanged from day 1

No margin. No options. No securities lending. No prime money market fund. No
corporate credit. No equity. No auto-reinvestment on any rung. No good-till-
canceled order of any kind on an instruction written in advance of a session.

## The six checks before any of it is entered

1. **Is the market actually open?** Confirm against the exchange calendar, not
   against an assumption about which Monday it is.
2. **Has the release approval been given?** The queue is not self-executing.
3. **Has the financing decision changed the branch?** If an instrument was
   selected, the queue is re-cut before entry.
4. **Is the auction calendar re-read?** See above. This is the check most likely
   to be skipped and most likely to matter.
5. **Is the limit price written this morning?** Not carried from this file.
6. **Is the reserve balance current?** Sizes are shares of the balance on the
   morning of entry.

## What is not queued

The site start-up pool from day 3 is not queued here. It is held in a sweep, a
single short bill and one liquid fund, and its draws release against artifacts
rather than against a market session. Queuing it would imply a schedule it does
not have.

Subscription funds are not queued, because none exist. Federal award funds are
not queued, because none exist, and would not be queued in any case: they are
drawn under the award's own terms.

---

**Sources.** `funding/auto-fund/02Sep26/investing/capital-01-treasury-ladder.md`;
`funding/auto-fund/03Sep26/investing/capital-02-corporate-reserve-allocation.md`;
[TreasuryDirect auction announcements and results](https://www.treasurydirect.gov/auctions/announcements-data-results/);
[NYSE holidays and trading hours](https://www.nyse.com/markets/hours-calendars).
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).
