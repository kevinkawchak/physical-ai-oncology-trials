# Hour 08: 08:00-08:59 - First Peak Activity Window

Released on 1 May 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

First peak hour. 9 new arrivals; 3 departures; 8 active sessions concurrently
at peak. AE-001 patient remains stable on hydration.

## Cross-Site Status at 08:00 UTC

| Site | Patients | Active | Standby | Active |
|------|----------|--------|---------|--------|
| SITE-A | 11 | 2 | 27 | 2 |
| SITE-B | 8 | 1 | 28 | 1 |
| SITE-C | 6 | 0 | 29 | 0 |
| SITE-D | 6 | 0 | 29 | 0 |

## New Arrivals

| Patient ID | Site | Time | Cancer | Robot |
|------------|------|------|--------|-------|
| PAT-CONT-0024A | SITE-A | 08:02 | NSCLC | RT Pos (3) |
| PAT-CONT-0025A | SITE-C | 08:08 | LS-SCLC restage | Imaging (8) |
| PAT-CONT-0026A | SITE-B | 08:14 | Mantle cell C2D1 | Humanoid IND |
| PAT-CONT-0027A | SITE-A | 08:21 | Soft-tissue sarcoma | Cobot (2) |
| PAT-CONT-0028A | SITE-D | 08:27 | Pediatric osteosarcoma rehab | Rehab (10) |
| PAT-CONT-0029A | SITE-A | 08:35 | Brain GBM | RT Pos (3) |
| PAT-CONT-0030A | SITE-C | 08:42 | LS-SCLC | Cobot (2) |
| PAT-CONT-0031A | SITE-B | 08:48 | Liver HCC | Imaging (8) |
| PAT-CONT-0032A | SITE-A | 08:55 | Pediatric ALL | Companion (5) |

## Minute Activity Log (compressed)

```
08:00 Hour-boundary FDA recon: 5/5 endpoints accepted from hour 07.
08:02-08:24 RTPOS-03 + TRACK-03 (SITE-A): NSCLC RT fraction 18 of 30.
08:08-08:30 IMAGE-01 (SITE-C): SCLC restage CT scan.
08:14-08:55 HUMAN-02 (SITE-B): TRAVERSE C2D1 IND ARM-A 60 mg infusion.
08:21-08:34 COBOT-04 (SITE-A): sarcoma core biopsy.
08:23 PAT-CONT-0023A biopsy (started 07:56) completes. Endpoint 23s.
08:24 RT endpoint streamed: TRAVERSE-CONT-0024A-RT18. ack 11s.
08:27-08:50 REHAB-03 (SITE-D): pediatric gait + strength.
08:30 IMG endpoint: STREAM-CONT-0025A-RESTAGE. ack 12s.
08:35-08:58 RTPOS-01 + TRACK-01 (SITE-A): GBM RT9.
08:42-08:55 COBOT-01 (SITE-C): SCLC core biopsy.
08:48-08:59 IMAGE-03 (SITE-B): HCC pre-ablation imaging.
08:55-08:59 COMPN-01 (SITE-A): pediatric companion engagement.
08:55 IND endpoint: TRAVERSE-CONT-0026A-IND-C2D1 ack 9s.
08:34 BX endpoint: TRAVERSE-CONT-0027A-BX01 ack 11s.
08:50 REHAB endpoint: PED-CONT-0028A-REHAB ack 13s.
08:58 RT endpoint: TRAVERSE-CONT-0029A-RT09 ack 10s.
08:55 BX endpoint: STREAM-CONT-0030A-BX-RESTAGE ack 14s.
```

## Departures (3)

| Patient | Time | Outcome |
|---------|------|---------|
| PAT-CONT-0001 | 08:15 | AM follow-up imaging then discharge |
| PAT-CONT-0008 | 08:30 | Post-ablation 24h obs complete |
| PAT-CONT-0017A | 08:45 | Post-RT 1h obs complete |

## Adverse Events
None new. AE-001 (PAT-CONT-0007) downgraded to grade 0 at 08:45 (resolved).

## IND Administrations
- PAT-CONT-0026A: TRAVERSE C2D1 ARM-A 60 mg, infusion 08:14-08:54.

## Cross-Site Utilization
- SITE-A: 137 robot-min
- SITE-B: 53 robot-min
- SITE-C: 36 robot-min
- SITE-D: 23 robot-min
- Network utilization: 3.58% (peak so far)

## RTCT Signal Stream
8 endpoints streamed; all acked <15 s.

## Regulatory Compliance
- AE resolution recorded at 08:45 per Part 312.32.
