# Capital Instruction 02: The Reserve on Two Branches

**ChemicalQDevice LLC corporate brokerage account.** Direction of the sole member
and chief executive, conditional on the day's financing decision. Not investment
advice, not a recommendation to any other person, and not an offer or
solicitation. Nothing below has been placed.

---

## The branch point

| Branch | Taken when | Operating horizon |
|:--|:--|:--|
| A, unfinanced | No instrument is selected on this day | 9 months, unchanged |
| B, financed | An instrument is selected and counsel is engaged | 24 months from a first close |

Branch A is the day 1 instruction, unchanged and unamended. Nothing in this file
modifies it. If branch A is taken, the only action is the one already in
[`../emails/email-05-brokerage-corporate-account.txt`](../emails/email-05-brokerage-corporate-account.txt):
leave every rung untouched and let each maturity fall to the sweep.

Everything below describes branch B.

## Branch B: the allocation

Sizes are shares of the operating reserve, not of subscription funds.
Subscription funds are held separately and are drawn against the use-of-proceeds
schedule, not laddered.

| # | Line | Instrument | Share | Order type | Horizon |
|:--|:--|:--|:--|:--|:--|
| 1 | Rung A | U.S. Treasury bill | 15 pct | Non-competitive at auction | About 3 months |
| 2 | Rung B | U.S. Treasury bill | 15 pct | Non-competitive at auction | About 6 months |
| 3 | Rung C | U.S. Treasury bill | 15 pct | Non-competitive at auction | About 9 months |
| 4 | Rung D | U.S. Treasury note | 15 pct | Secondary market | About 12 months |
| 5 | Rung E | U.S. Treasury note | 15 pct | Secondary market | About 18 months |
| 6 | Rung F | U.S. Treasury note | 10 pct | Secondary market | About 24 months |
| 7 | Liquid sleeve | Short-duration Treasury ETF | 10 pct | Limit, day, never market | No maturity |
| 8 | Residual | Government money market sweep | 5 pct | Sweep | Overnight |

Six rungs rather than four, because a 24-month horizon with four rungs leaves
six-month gaps between maturities and a company's cash needs do not arrive on a
six-month grid.

## What stays excluded on both branches

No margin. No options. No securities lending. No prime money market fund. No
corporate credit of any maturity. No equity. No auto-reinvestment on any rung. No
single non-Treasury position other than the one exchange-traded sleeve.

A financing does not change what a research company's reserve is for. It changes
how long the reserve has to last, and that is a maturity question rather than a
credit question.

## The liquid sleeve entry rule, unchanged from day 1

| Item | Value |
|:--|:--|
| Candidate instruments | A short-duration U.S. Treasury exchange-traded fund. **SGOV** (0 to 3 month bills), **BIL** (1 to 3 month bills), and **SHV** (under 1 year) are three widely held examples |
| Selection rule | Choose one, not three. Prefer the lowest total expense ratio available with no transaction fee |
| Order type | **Limit, day. Never market** |
| Limit price | The prior session's closing price plus one cent, written into the broker letter on the morning of entry |
| Do not enter | In the first or last fifteen minutes of the session |

## Subscription funds: held, not laddered

| Rule | Reason |
|:--|:--|
| Separate account under the same entity | Segregation is established before the first dollar, not reconstructed afterward |
| Government money market sweep only | Subscription funds are drawn against a schedule that is not yet fixed; a maturity that has to be broken is worse than a lower yield |
| Drawn against the seven-line use of proceeds | See `../briefs/brief-03-use-of-proceeds.md` |
| Never used to pay a cost that a federal award would properly bear | And never the reverse |

## The runway arithmetic

| Input | Branch A | Branch B |
|:--|:--|:--|
| Operating horizon the reserve is cut to | 9 months | 24 months |
| Longest rung | About 12 months | About 24 months |
| Number of decision points before the horizon ends | 4 | 6 |
| Liquid sleeve share | 15 pct | 10 pct |

The sleeve is smaller on branch B because a financed company has a second source
of liquidity and does not need to hold as much of its reserve at call.

## The six checks before either branch is taken

1. **Has an instrument actually been selected?** If not, branch A stands and
   nothing is entered.
2. **Has counsel been engaged?** A selected instrument with no engaged counsel is
   not a financing; it is an intention.
3. **Is the reserve balance current?** Shares are computed from the balance on
   the morning of entry, not from a prior statement.
4. **Is the limit price current?** See the sleeve rule above.
5. **Have any subscription funds arrived?** If so, they are in the separate
   account before a single reserve order is entered.
6. **Have any federal award funds arrived?** If so, they are segregated and
   drawn under the award's own terms, and are not part of either branch.

---

**Sources.** `funding/auto-fund/02Sep26/investing/capital-01-treasury-ladder.md`;
`../briefs/brief-01-instrument-comparison.md`; `../briefs/brief-03-use-of-proceeds.md`;
`funding/move-in/final-move-in/sections/sec-15-funding-and-lobbying.tex`.
Reference data: [TreasuryDirect auction results](https://www.treasurydirect.gov/auctions/announcements-data-results/)
and the [Treasury par yield curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/).
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).
