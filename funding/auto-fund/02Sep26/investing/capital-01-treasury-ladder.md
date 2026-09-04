# Capital Instruction 01: A Four-Rung Treasury Ladder to a Nine-Month Horizon

**ChemicalQDevice LLC corporate brokerage account.** Direction of the sole member
and chief executive. Not investment advice, not a recommendation to any other
person, and not an offer or solicitation. Nothing below has been placed.

---

## The horizon this is cut to

| Input | Value | Source |
|:--|:--|:--|
| SBIR Phase I duration | 9 months | `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed` |
| SBIR Phase I total cost | $306,000 | Same |
| Phase I award-funded staffing | 1.75 FTE across four roles | `../briefs/brief-02-sbir-phase-i-readiness.md` |
| Operating reserve purpose | Cover operations to the first Phase I milestone without selling a position early | This instruction |

The reserve is the company's own capital. **Award funds are never mixed into it.**
Federal award funds are held and drawn under the award's own terms, and the
separation is the same one set out in the site package's funding stewardship
document.

## The five lines

Sizes are expressed as a share of the reserve rather than in dollars, so the
instruction stays correct whatever the reserve balance is on the day it is
approved.

| # | Line | Instrument | Share | Order type | Maturity target |
|:--|:--|:--|:--|:--|:--|
| 1 | Rung A | U.S. Treasury bill | 20 percent | Non-competitive at auction, or secondary market | About 3 months |
| 2 | Rung B | U.S. Treasury bill | 20 percent | Non-competitive at auction, or secondary market | About 6 months |
| 3 | Rung C | U.S. Treasury bill | 20 percent | Non-competitive at auction, or secondary market | About 9 months |
| 4 | Rung D | U.S. Treasury note or bill | 20 percent | Secondary market | About 12 months |
| 5 | Liquid sleeve | Short-duration U.S. Treasury ETF | 15 percent | **Limit order only** | Not applicable |
| 6 | Residual | Government money market sweep | 5 percent | Sweep | Overnight |

Rungs A to D are **direct obligations with a stated maturity date and a CUSIP on
each rung**, not a fund wrapper. The distinction matters: a fund has a price and a
direct obligation has a maturity, and the ladder's whole purpose is to own
maturities.

## The liquid sleeve, and how to enter it

The sleeve exists so that an unexpected cash need is met without breaking a rung.
It is the only line here that carries a market price, so it is the only line with
an entry rule.

| Item | Value |
|:--|:--|
| Candidate instruments | A short-duration U.S. Treasury exchange-traded fund. Two widely held examples are **SGOV** (0 to 3 month Treasury bills) and **BIL** (1 to 3 month Treasury bills). **SHV** (under 1 year) is a third |
| Selection rule | Choose one, not three. Prefer the lowest total expense ratio available in the account with no transaction fee |
| Order type | **Limit, day. Never market** |
| Limit price | The prior session's closing price, plus one cent. Write the actual number into the broker letter on the morning of entry |
| Do not enter | In the first fifteen minutes or the last fifteen minutes of the session, when the spread is widest |
| Size | 15 percent of the reserve, rounded down to a whole share count |

A limit set on one day and sent on another is a stale limit. The broker letter at
[`../emails/email-05-brokerage-treasury-instruction.txt`](../emails/email-05-brokerage-treasury-instruction.txt)
carries a pre-send check for exactly this.

## Constraints on every line

No margin. No options. No securities lending. No auto-reinvestment on any rung.
No prime money market fund. No corporate credit. No equity. No single
non-Treasury position at any time other than the one exchange-traded sleeve
above.

## Settlement, tax, and record notes

| Item | Note |
|:--|:--|
| Treasury bill settlement | Auction settlement is typically the Thursday following a Monday or Tuesday auction; confirm the settlement date on the auction announcement before assuming a cash date |
| Discount accrual | Treasury bills are issued at a discount; the accretion is interest income for federal purposes and is recognized at maturity or sale |
| State tax | Interest on direct U.S. Treasury obligations is exempt from California personal income tax; the exemption flows through a single-member LLC treated as a disregarded entity. Confirm with the company's tax preparer before relying on it |
| ETF distributions | The Treasury portion of an exchange-traded fund distribution may be state-exempt only if the fund reports it; keep the annual fund tax letter |
| Record | Record the CUSIP, par, price, settlement date and maturity date of every rung in the company's own ledger on the day of the fill, not at quarter end |

## The five checks before this is approved

1. **Is the horizon still nine months?** If the Phase I duration has changed in
   `app-05-nih-sbir-seed`, the ladder is re-cut before it is entered.
2. **Is the reserve balance the number this is sized against?** Shares are
   computed from the balance on the morning of entry, not from a prior statement.
3. **Is the limit price current?** See the sleeve section above.
4. **Is auto-reinvestment off on every rung?** Confirm in the account settings,
   not in the order ticket.
5. **Is any of this money award money?** If any federal award funds have arrived,
   they are segregated before a single order is entered.

## What day 2 revisits

Day 2 decides the private capital instrument. That decision changes the size of
the reserve and can change how long the horizon is, so a non-Treasury sleeve is
considered on day 2 and not before.

---

**Sources.** `funding/pdac-funding-applications/applications/app-05-nih-sbir-seed`;
`funding/capitalization-plan/final-capital/sections/sec-04-capital-bridge.tex`;
`funding/move-in/final-move-in/sections/sec-15-funding-and-lobbying.tex`.
Reference data: [TreasuryDirect auction results](https://www.treasurydirect.gov/auctions/announcements-data-results/)
and the [Treasury par yield curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/).
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).
