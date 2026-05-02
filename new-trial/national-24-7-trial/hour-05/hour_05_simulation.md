# Hour 05: 05:00-05:59 - Network Awakening

Released on 1 May 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Pre-dawn ramp begins. 3 new arrivals as east-coast patients prepare for early
shifts. STREAM-SCLC patient PAT-CONT-0002 returns for treatment planning
finalization. No AEs.

## Cross-Site Status at 05:00 UTC

| Site | Patients | Active | Standby | Active | Signal |
|------|----------|--------|---------|--------|--------|
| SITE-A | 6 | 0 | 29 | 0 | UP |
| SITE-B | 5 | 0 | 29 | 0 | UP |
| SITE-C | 4 | 0 | 29 | 0 | UP |
| SITE-D | 4 | 0 | 29 | 0 | UP |

## New Arrivals

| Patient ID | Site | Time | Age | Sex | Cancer | Stage | Robot |
|------------|------|------|-----|-----|--------|-------|-------|
| PAT-CONT-0011A | SITE-B | 05:08 | 71 | M | NSCLC | IIIB | RT Pos (3) |
| PAT-CONT-0012A | SITE-A | 05:30 | 47 | F | Soft-tissue sarcoma | II | Cobot bx (2) |
| PAT-CONT-0013A | SITE-C | 05:51 | 60 | F | LS-SCLC (STREAM C2D1) | LS | Humanoid IND |

(Note: -A suffix indicates new continuous-trial patient distinct from
overnight carry-over PAT-CONT-0011/0012/0013.)

## Minute Activity Log

```
05:00 Hour-boundary FDA recon: 2/2 endpoints accepted from hour 04.
05:01-05:07 Quiet.
05:08 PAT-CONT-0011A check-in SITE-B. RT fraction 4 of 30.
05:09-05:10 RTPOS-03 + TRACK-03 wake. Vault 3 prep.
05:11 Patient positioned. Setup error 0.5 mm.
05:12-05:25 RT delivery (2 Gy, 4 fields, NSCLC right lower lobe).
05:26 Beam-off. Patient exits vault.
05:27 RT endpoint streamed: TRAVERSE-CONT-0011A-RT04. FDA ack 05:27+10s.
05:28-05:29 Quiet.
05:30 PAT-CONT-0012A check-in SITE-A. Sarcoma core biopsy.
05:31 COBOT-04 (SITE-A) wakes. Pre-procedure matrix.
05:32-05:42 Biopsy (5 cores). Force trace nominal.
05:43 COBOT-04 retracts. Patient to recovery bay 7 (SITE-A).
05:44 BX endpoint streamed: TRAVERSE-CONT-0012A-BX01. FDA ack 05:44+11s.
05:45-05:50 Quiet.
05:51 PAT-CONT-0013A check-in SITE-C. STREAM C2D1 dosing visit.
05:52 HUMAN-01 (SITE-C) wakes. Pharmacy retrieves blinded kit BTK-d-0089.
05:53 IRT confirms ARM-B (placebo). Drug accountability trio sign.
05:54 Pre-dose vitals: BP 130/76, HR 80, SpO2 97%.
05:55-05:59 Pre-dose ECG (IMAGE-02 SITE-C) and infusion start prep.
05:59 Hour 05 close. Infusion will begin in hour 06.
```

## Departures
None.

## Adverse Events
None.

## IND Administrations
PAT-CONT-0013A pre-dose checks complete; infusion begins hour 06.

## Cross-Site Utilization
- SITE-A: 14 robot-min (COBOT-04)
- SITE-B: 18 robot-min (RTPOS-03 + TRACK-03)
- SITE-C: 9 robot-min (HUMAN-01 + IMAGE-02)
- Network utilization: 0.59%

## Regulatory Compliance
- All standard sections compliant. IRT randomization for STREAM ARM-B
  performed under section 4.6.

## RTCT Signal Stream

| Signal ID | Patient | Endpoint | FDA Ack |
|-----------|---------|----------|---------|
| TRAVERSE-CONT-0011A-RT04 | PAT-CONT-0011A | RT fraction 4 | 10 s |
| TRAVERSE-CONT-0012A-BX01 | PAT-CONT-0012A | Tissue acquired | 11 s |
