# Capital Instruction 01: A Contingent Payroll Earmark Inside the Existing Ladder

**ChemicalQDevice LLC corporate brokerage account.** Direction of the sole member
and chief executive. Not investment advice, not a recommendation to any other
person, and not an offer or solicitation. Nothing below has been placed.

---

## What this instruction does

It designates part of the company's existing Treasury ladder as a payroll
earmark for the first hires, and it buys nothing today. The ladder was set in
`funding/auto-fund/02Sep26/investing/capital-01-treasury-ladder.md`: four rungs
of direct Treasury obligations at about 3, 6, 9 and 12 months, a short Treasury
exchange-traded fund sleeve, and a government money market residual. This
instruction changes the purpose of the shortest rung and nothing else.

## Why an earmark, and why 13 weeks

An award pays salaries, but not on the day it is announced. Between a notice of
award and the first drawdown there is setup time, and a new employee's first
paychecks should not wait on it. Both NIH and NSF allow a recipient to incur
allowable costs up to 90 days before the award's start date, at the recipient's
own risk. Thirteen weeks is 91 days, so an earmark sized to 13 weeks covers
exactly the window the agencies permit and no more.

The earmark is spent only after an award letter is in hand and an offer has been
accepted. Before that, it is simply the shortest rung of the ladder, maturing
and rolling on its normal schedule.

## The sizing rule

Earmark = the annual budgeted cost of each role for which an offer is accepted,
salary and fringe combined, times 13 and divided by 52.

| Role | Budget per year | 13-week earmark |
|:--|:--|:--|
| Director of clinical operations | $54,000 | $13,500 |
| Lead clinical research coordinator | $88,000 | $22,000 |
| Regulatory affairs and quality manager | $52,000 | $13,000 |
| Robotics and physical AI systems engineer, site safety officer | $72,000 | $18,000 |
| LLM verification and model governance lead | $58,000 | $14,500 |
| Data manager and biostatistician | $38,000 | $9,500 |

The largest two-person earmark is the lead coordinator and the systems engineer
together, at **$40,000**. That is the ceiling this instruction plans for, and it
is the number to compare against Rung A's par value.

## The two lines

| # | Line | Instrument | Size | Order type | When |
|:--|:--|:--|:--|:--|:--|
| 1 | Designate Rung A as the payroll earmark | The existing 3-month Treasury bill rung | Its current par value | No order; a ledger entry | Today |
| 2 | Top up Rung A, only if its par is below $40,000 | 13-week U.S. Treasury bill | The shortfall, rounded up to the next $100 of par | Non-competitive bid at the weekly 13-week bill auction | At the next auction after approval |

Line 2 is conditional. If Rung A already holds $40,000 of par or more, line 2 is
not entered, and nothing is bought. If it is entered, the non-competitive bid
takes the auction's high rate, so there is no limit price to set and nothing to
chase in the secondary market.

## Rules that bind the earmark

| Rule | Why |
|:--|:--|
| Award funds are never deposited into the brokerage account | Federal funds are drawn and spent under the award's own terms, and mixing them with the reserve makes both unauditable |
| The earmark pays pre-award costs only inside the 90-day window, and only after an award letter | Costs outside the window are unallowable and would be the company's own expense |
| No position is sold below par to meet payroll | The earmark matures; it is not liquidated |
| No equity, no margin, no options, no corporate credit | The ladder's constraints from the parent instruction are unchanged |
| The earmark is released back to the ladder if no offer is accepted within 13 weeks of an award | An earmark with no hire is idle capital |

## Settlement, tax, and record notes

| Item | Note |
|:--|:--|
| Auction and settlement | 13-week bills are auctioned weekly, normally on Monday, and settle the following Thursday; confirm on the auction announcement before relying on a cash date |
| Discount accrual | The discount is interest income for federal purposes, recognized at maturity or sale |
| State tax | Interest on direct Treasury obligations is exempt from California personal income tax; the exemption flows through a single-member LLC treated as a disregarded entity. Confirm with the company's tax preparer |
| Record | Record the designation, and any top-up's CUSIP, par, price, settlement date and maturity, in the company ledger on the day of the entry |

## The four checks before this is approved

1. The ladder from the parent instruction exists and Rung A's par value is known today.
2. No offer has been made. The earmark is contingent and must stay contingent.
3. The brokerage account is the company's, not the chief executive's personal account.
4. The chief executive has read the rule that award funds never enter this account.

## Sources

- Parent instruction: `funding/auto-fund/02Sep26/investing/capital-01-treasury-ladder.md`
- Role budgets: `funding/move-in/final-move-in/sections/sec-14-staffing-and-roles.tex`
- TreasuryDirect auction data: https://www.treasurydirect.gov/auctions/announcements-data-results/
- NIH pre-award costs: https://grants.nih.gov/policy-and-compliance/policy-topics/small-business
