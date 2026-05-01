# Hour 00 PSL Scores: 00:00-00:59

Released on 1 May 2026 | CEO Kevin Kawchak | ChemicalQDevice

## PSL Snapshot - Network Aggregate (4 sites)

| # | Robot Type | A (Omniscient) | B (Omnipresent) | C (Omnipotent) | PSL Total |
|---|-----------|----------------|-----------------|----------------|-----------|
| 1 | Surgical Robots | 7.4 | 6.6 | 6.0 | 6.7 |
| 2 | Cobots | 7.8 | 7.0 | 6.8 | 7.2 |
| 3 | RT Positioning | 7.6 | 6.8 | 6.4 | 6.9 |
| 4 | Needle-Placement | 7.2 | 6.5 | 6.2 | 6.6 |
| 5 | Companion | 6.8 | 7.4 | 5.6 | 6.6 |
| 6 | Humanoids | 6.4 | 6.0 | 5.4 | 5.9 |
| 7 | RT Motion-Tracking | 8.0 | 7.2 | 6.8 | 7.3 |
| 8 | Imaging | 7.6 | 7.0 | 6.4 | 7.0 |
| 9 | Steerable Needle | 7.0 | 6.4 | 6.2 | 6.5 |
| 10 | Rehab Exoskeletons | 6.8 | 6.6 | 6.0 | 6.5 |

**Cumulative Site PSL (mean across 4 sites): 67.2 / 100** - Advanced Site
classification.

## Per-Site PSL

| Site | PSL Total | Classification |
|------|-----------|----------------|
| SITE-A | 68.4 | Advanced |
| SITE-B | 66.8 | Advanced |
| SITE-C | 67.6 | Advanced |
| SITE-D | 66.0 | Advanced |

## C-PSL (Continuity-PSL, rolling 24-hour mean)

Hour 00 is the first hour of the continuous trial; rolling window not yet
populated. C-PSL bootstrap value = 67.2 (= cumulative site PSL).

## Dimension Notes

### Dimension A - Omniscient (ICH E6(R3))

- Real-time data streaming from all 4 active procedures.
- Audit trail latency to Paradigm Health: median 2.1 s (target <5 s). Pass.
- Digital twin sync confirmed for PAT-CONT-0002 imaging update.

### Dimension B - Omnipresent (21 CFR Part 50)

- All 15 on-site patients had valid eConsent including the RTCT addendum.
- Pre-procedure safety matrix executed for all 4 active sessions.
- Pediatric assents (2) verified with guardian co-signature.

### Dimension C - Omnipotent (21 CFR Part 312)

- IND-equivalent records updated within 15 min for all signal events.
- No safety reporting events triggered.
- Real-time FDA channel responsive (median ack 13 s).

## RTCT-Specific Adjustments

For continuous trials, the FDA's 28 April 2026 framework introduces three
new attributes inside Dimension A:

1. **Signal latency to FDA** - target <30 s (median this hour: 12.5 s) Pass.
2. **Endpoint validation by Paradigm Health** - target 100% (this hour: 100%).
3. **Continuous re-enrollment readiness** - target binary, achieved at 00:00.

These adjustments contribute +0.4 to Dimension A across all robot types.

## Trend Analysis

Single-hour snapshot. No trend yet. Trend reporting begins hour 01 once a
delta is computed.
