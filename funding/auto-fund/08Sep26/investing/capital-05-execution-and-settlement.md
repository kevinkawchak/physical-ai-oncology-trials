# Capital Instruction 05: Execution, Settlement, and the Record

**ChemicalQDevice LLC corporate brokerage account.** Direction of the sole member
and chief executive. Not investment advice, not a recommendation to any other
person, and not an offer or solicitation.

---

## The six checks, before anything is transmitted

1. Market open, confirmed against the exchange calendar rather than assumed.
2. Release approval given on the preceding day and unchanged overnight.
3. Financing branch unchanged: no instrument selected, so the nine-month cut
   stands. If one was selected, the queue is re-cut before entry, never amended
   after.
4. Auction calendar re-read this morning: next auction date and settlement date
   per tenor, and the broker's own cutoff relative to each.
5. Limit prices for lines 4 and 5 written this morning from the last completed
   session's close.
6. Reserve balance confirmed this morning, since every size is a share of it.

A no on any of the six stops the entry. Check 4 is the one most likely to be
skipped and the one most likely to matter.

## The entry sequence

Order matters, because an auction cutoff is a hard stop and a limit order is not.

| Order of entry | Line | Why here in the sequence |
|:--|:--|:--|
| First | Lines 1, 2, 3, the auction bids | The cutoff is external and cannot be moved |
| Second | Line 4, the twelve-month secondary purchase | A limit order can rest through the session |
| Third | Line 5, the exchange-traded sleeve | Entered mid-session, never in the first or last fifteen minutes |
| Fourth | Line 6, the sweep residual | Automatic once the others fill |

## Settlement basis, per line

| Line | Settlement basis | What to record |
|:--|:--|:--|
| 1, 2, 3 | Auction settlement date as stated on the auction announcement, typically two business days after the auction | CUSIP, par, discount price, auction date, settlement date, maturity date |
| 4 | Regular way secondary settlement | CUSIP, par, price, trade date, settlement date, maturity date |
| 5 | Standard exchange-traded fund settlement | Ticker, share count, execution price, trade date, settlement date |
| 6 | Overnight | Balance only |

Treasury bills are issued at a discount, and the accretion is interest income for
federal purposes recognized at maturity or sale. Interest on direct United States
Treasury obligations is exempt from California personal income tax and the
exemption flows through a single-member limited liability company treated as a
disregarded entity; confirm with the company's tax preparer before relying on it.

## The fill record, written on the day of the fill

Not at quarter end. A ledger written from statements three months later is a
ledger that reconciles to the statements and to nothing else.

| Field | For every line |
|:--|:--|
| Instrument identifier | CUSIP, or ticker for line 5 |
| Side and quantity | Buy, par or shares |
| Price | Discount price, or execution price |
| Trade date | |
| Settlement date | |
| Maturity date | Lines 1 to 4 only |
| Share of reserve at entry | The percentage this line represented on the morning of entry |
| Auto-reinvestment | Confirmed **off** |

## Reconciliation, at the close of the session

| Check | Passes when |
|:--|:--|
| Every line either filled or is recorded as unfilled with a reason | No line is unaccounted for |
| The six shares sum to 100 percent of the reserve as of this morning | The arithmetic closes |
| No position outside the authorized set exists in the account | A single unauthorized position is a control failure, not a preference failure |
| Auto-reinvestment is off on every rung | Confirmed in the account settings, not in the order ticket |
| No margin balance, no option position, no lending consent | Confirmed on the statement |

## If a line does not fill

| Line | If it does not fill |
|:--|:--|
| 1, 2, 3 | The cutoff was missed or the auction is next week. Carry to the next auction; do not substitute a secondary purchase at a worse basis to feel finished |
| 4 | The limit was not reached. Cancel at the close and re-price tomorrow morning. Do not convert to a market order |
| 5 | The same. **Never** convert a limit to a market order to complete a day |
| 6 | Not applicable |

The instruction to never chase a fill is the single most important line in this
file. A ladder built to avoid being forced should not be completed by forcing it.

## What is not part of this entry

The site start-up pool, which is held in a sweep, one short bill and one liquid
fund, and whose draws release against artifacts rather than against a session.
Subscription funds, because none exist and no offering exists. Federal award
funds, because none exist and would be drawn under the award's own terms in any
case.

---

**Sources.** `funding/auto-fund/07Sep26/investing/capital-04-queued-orders.md`;
`funding/auto-fund/02Sep26/investing/capital-01-treasury-ladder.md`;
[TreasuryDirect auction announcements and results](https://www.treasurydirect.gov/auctions/announcements-data-results/);
[NYSE holidays and trading hours](https://www.nyse.com/markets/hours-calendars).
Repository (v4.8.0):
[physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials).
