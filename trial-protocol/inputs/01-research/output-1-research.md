## output-1-research

# Financial-data research for H. R. 9510 Bill v5.0 (current for 2026)

*Independent research draft. Not legal advice; not endorsed by the FDA, HHS,
the CBO, the GAO, the OMB, the OLRC, or any Member of Congress. Prepared
June 9, 2026 for the 119th Congress, 2d Session.*

This research gives H. R. 9510 **Bill v5.0**, the Financial Data Amendment, its
financial spine. The bill descends from VVUQ-03 (standalone bill, v1.0),
VVUQ-04 (FD&C Act amendment, v2.0), VVUQ-05 (visual amendment, v3.0), and
auto-bill-01 (gray-scale Mermaid amendment, v4.0); v5.0 is built from the Bill
v3.0 LaTeX measure and adds the operative financial core and the financial
appendices. Everything below is what the bill must get exactly right.

## Three reality checks that frame every dollar

1. **"H. R. 9510" is illustrative, not assigned.** The Clerk assigns a number
   only at introduction.
2. **A private project cannot submit a bill directly.** Only a sitting Member
   may introduce a House bill; the financial documents below are what a
   sponsoring office expects to accompany the text.
3. **No formal CBO estimate exists before introduction.** Every dollar figure
   this build generates is therefore illustrative or simulation-derived unless
   it is tied to a cited statute or published notice, and the bill must say so
   wherever a figure appears.

## 1. The scoring framework the bill is judged under

- **CBO cost estimates.** Under section 402 of the Congressional Budget Act of
  1974, CBO prepares a cost estimate for nearly every bill reported by a
  committee, measured against its current-law baseline over the standard
  budget window (now five and ten years). Informal consultation can precede
  introduction; the formal score follows referral and markup.
- **Statutory PAYGO (2 U.S.C. §§ 931-939).** The Statutory Pay-As-You-Go Act
  of 2010 reaches **direct spending and revenue** only. OMB keeps five-year
  and ten-year scorecards; a year-end net debit triggers a sequestration
  order. A bill whose only costs are subject to future appropriations does not
  enter the scorecards; the standard floor language is the "Budgetary Effects"
  clause citing the latest statement of the Chairman of the House Budget
  Committee. Bill v5.0 carries that clause verbatim in SEC. 5.
- **House CutGo (Rule XXI, clause 10, 119th Congress).** A measure may not
  have the net effect of **increasing mandatory spending** within the
  five-year or ten-year windows. Because v5.0 authorizes appropriations
  (discretionary) and creates no entitlement, it is CutGo-compatible by
  design, and the cost appendix must demonstrate that.
- **Earmarks (Rule XXI, clause 9).** The bill must carry a negative
  declaration: no congressional earmarks, limited tax benefits, or limited
  tariff benefits. An authorization of appropriations to an agency for
  generally applicable implementation is not an earmark.

## 2. The vocabulary the bill must use exactly

| Term | Meaning the bill relies on | Source of record |
|:--|:--|:--|
| Budget authority (BA) | Legal authority to incur financial obligations | GAO-05-734SP (budget glossary) |
| Obligation | A binding agreement that will result in outlays | GAO-05-734SP |
| Outlay | The liquidation of an obligation (cash leaves Treasury) | GAO-05-734SP |
| Authorization of appropriations | Authorizes, but does not provide, BA; funds flow only when appropriated | GAO-05-734SP; CBO conventions |
| Direct (mandatory) spending | BA provided outside annual appropriations (entitlements) | 2 U.S.C. § 932 |
| Discretionary spending | BA provided in annual appropriations acts | 2 U.S.C. § 932 |
| PAYGO scorecard | OMB 5- and 10-year ledgers of direct spending and revenue effects | 2 U.S.C. § 933 |
| User fee | A charge for a government service, here the device-review fees | 21 U.S.C. §§ 379i-379j |

The single most common drafting error the bill must avoid: writing "is
appropriated" where "is authorized to be appropriated" is meant. SEC. 5
authorizes; only an appropriations Act provides.

## 3. How FDA device work is paid for (what SEC. 5 rides on)

FDA's device program is funded by **annual discretionary appropriations**
(the Salaries and Expenses account, within the agriculture appropriations
bill) plus **medical device user fees** collected under FD&C Act sections 737
and 738 (21 U.S.C. §§ 379i, 379j), authorized in five-year cycles. The current
cycle, **MDUFA V (Pub. L. 117-180), runs fiscal years 2023 through 2027**, with
fee amounts set each year by Federal Register notice; **MDUFA VI must be
negotiated and enacted before October 1, 2027**, which is the realistic vehicle
for any fee treatment of verification records. The statute fixes the fee
classes as percentages of the standard premarket approval (PMA) fee:

| Submission (FD&C § 738) | Statutory fee, as percent of the standard PMA fee |
|:--|:--|
| Premarket approval application (PMA) | 100 percent (the base fee) |
| Panel-track supplement | 75 percent |
| De novo classification request | 30 percent |
| 180-day supplement | 15 percent |
| Real-time supplement | 7 percent |
| 510(k) premarket notification | 3.4 percent |
| 30-day notice | 1.6 percent |
| 513(g) request for information | 1.35 percent |
| Annual fee for periodic reporting | 3.5 percent |

Two financial design consequences for v5.0: (a) a sponsor change governed by a
predetermined change control plan plus a cleared section 515D verification
record already avoids a new 510(k) or supplement, so the bill's user-fee
treatment must say that **filing a verification record is not itself a
fee-bearing submission**; and (b) any new review workload FDA absorbs is
discretionary and belongs in the authorization of appropriations, not in a new
statutory fee, until the MDUFA VI negotiation prices it.

## 4. Financial-data law already attached to clinical investigations

- **21 CFR part 54 - financial disclosure by clinical investigators.**
  Applicants must certify (Form FDA 3454) or disclose (Form FDA 3455)
  investigator financial arrangements: compensation affected by outcome,
  proprietary interests in the tested product, significant equity interests,
  and significant payments of other sorts. Bill v5.0 aligns the new
  verification-cost record with part 54 so the same disclosure discipline
  reaches verification-service providers.
- **Open Payments (42 U.S.C. § 1320a-7h; 42 CFR part 403, subpart I).**
  Applicable manufacturers must report payments and other transfers of value
  to covered recipients; CMS publishes the data annually. This is the model
  for the bill's public, de-identified verification-cost summaries.
- **Medicare coverage (42 U.S.C. § 1395y(a)(1)(A)).** Coverage requires that
  items and services be reasonable and necessary; a cleared verification
  record is financial evidence supporting that determination for Physical AI
  procedures, which the findings may note without amending the Social
  Security Act.

## 5. Unfunded mandates (what the cost appendix must address)

Under the Unfunded Mandates Reform Act of 1995 (2 U.S.C. § 1501 et seq.), CBO
flags intergovernmental mandates above $50 million and private-sector mandates
above $100 million (both in 1996 dollars, adjusted annually for inflation to
roughly double those nominal levels today). Bill v5.0 regulates sponsors of
Physical AI oncology investigations (a private-sector mandate whose
incremental cost - documentation, attestation, and cost-record keeping layered
on validation work responsible sponsors already perform - is expected to fall
below the threshold) and imposes no intergovernmental mandate; the savings
clause for State law is preserved.

## 6. The financial-data quality scaffolding for Appendix C

The bill's financial-data transparency standard maps to instruments that are
mass-adopted and in force: the **GAO Green Book** (GAO-14-704G, internal
control: control activities and information-and-communication principles),
the **GAO budget glossary** (GAO-05-734SP, the controlled vocabulary), **OMB
Circular A-11** (budget submission and execution discipline), and **21 CFR
part 11** (the tamper-evident record spine the bill already requires). The
standard's four tests: completeness (every cost recorded), traceability
(every figure traceable to a record), periodicity (fiscal-year alignment),
and comparability (the GAO vocabulary used exactly).

## 7. Pre-introduction mechanics, restated for the financial documents

The process facts confirmed in the v4.0 research stand for 2026 (119th
Congress, 2d Session; eHopper introduction; Energy and Commerce referral;
HOLC drafting; Rule XII, clause 7(c) Constitutional Authority Statement). The
financial additions a sponsoring office expects: the **PAYGO/CutGo posture
statement**, the **pre-introduction cost note** (clearly marked as not a CBO
product), and the **earmark negative declaration** - all three are carried
inside Bill v5.0 itself (SEC. 5 and Appendix A) rather than as external
deliverables, because v5.0 builds from the measure alone.

## The money map (ASCII)

```
                      THE FINANCIAL FRAME AROUND H. R. 9510 v5.0

  Congress                          FDA                          Sponsors
  +-----------------------+        +----------------------+     +------------------------+
  | Annual appropriations | --BA-> | Salaries & Expenses  |     | Verification runs      |
  | (discretionary)       |        | device program       |     | (compute + personnel)  |
  | SEC. 5(d) authorizes  |        | rulemaking, records  |     | cost records -> audit  |
  +-----------------------+        | system, audits       |     | trail (515D(k))        |
  +-----------------------+        +----------------------+     +------------------------+
  | MDUFA V user fees     | --$--> | device review        |     | part 54 disclosure     |
  | FY2023-FY2027         |        | (379i/379j classes)  |     | discipline extended    |
  | MDUFA VI by 10/2027   |        +----------------------+     +------------------------+
  +-----------------------+                   |
                                              v
                          Annual de-identified cost summaries (SEC. 5(a))
                          PAYGO: no direct spending; CutGo-compatible
```

## How this research drives the rest of the build

| Research area | Where it lands in Bill v5.0 |
|:--|:--|
| PAYGO, CutGo, earmark rules | SEC. 5(e) budgetary-effects clause; Appendix A scorecard and declaration |
| BA/outlays vocabulary | SEC. 5(d) authorization; Appendix A tables |
| FDA funding and MDUFA V/VI | SEC. 5(c) user-fee treatment; SEC. 4 comparative print (§ 379j); Appendix D timing row |
| Part 54, Open Payments, 1395y | SEC. 5(a)-(b); SEC. 2 findings; Appendix C standard |
| UMRA thresholds | Appendix A mandate analysis |
| Green Book, glossary, A-11 | Appendix C standard map |
| Pre-introduction mechanics | Appendix E implementation narrative |

## Sources

- Statutory Pay-As-You-Go Act of 2010, 2 U.S.C. §§ 931-939
- Rules of the House of Representatives, 119th Congress - Rule XXI, clauses 9 and 10
- Congressional Budget Act of 1974, § 402 (CBO cost estimates)
- GAO, *A Glossary of Terms Used in the Federal Budget Process*, GAO-05-734SP
- GAO, *Standards for Internal Control in the Federal Government* (Green Book), GAO-14-704G
- OMB Circular A-11, *Preparation, Submission, and Execution of the Budget*
- FD&C Act §§ 737-738, 21 U.S.C. §§ 379i-379j (medical device user fees)
- Medical Device User Fee Amendments of 2022 (MDUFA V), Pub. L. 117-180
- FDA, annual medical device user fee rates Federal Register notices
- 21 CFR part 54 (financial disclosure by clinical investigators); Forms FDA 3454/3455
- 42 U.S.C. § 1320a-7h; 42 CFR part 403, subpart I (Open Payments)
- 42 U.S.C. § 1395y(a)(1)(A) (Medicare reasonable and necessary)
- Unfunded Mandates Reform Act of 1995, 2 U.S.C. § 1501 et seq.
- CBO, *Cost Estimates and the Statutory Pay-As-You-Go Act* (process publications)
- `auto-bill-01/01-research/output-1-research.md` (the carried 2026 process facts)
- `cancer-automated/.../VVUQ-05/update-bill/next-steps` (the prior research method)
