# Hour 00 Patient Records: 00:00-00:59

Released on 1 May 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Patient Census Summary

- New arrivals this hour: 4
- Carry-over patients (overnight): 11 across 4 sites
- Total on-site at 00:59: 15
- Departures: 0
- Adverse events: 0

## New Arrivals - Detailed Records

### PAT-CONT-0001 (SITE-A, TRAVERSE analog)

- Demographics: 64 M, mantle cell lymphoma stage II
- ECOG: 1, vitals stable (BP 132/78, HR 74, SpO2 98%)
- eConsent: Continuous-trial v3.4 with RTCT addendum (signed 28 Apr 2026)
- Procedure: Cobot-guided lymph node biopsy (6 cores), site COBOT-01
- Procedure outcome: Successful, no complications
- Disposition at 00:59: Recovery bay 4, vitals nominal
- RTCT endpoint streamed: TRAVERSE-CONT-0001-BX01 (tissue acquired)

### PAT-CONT-0002 (SITE-C, STREAM-SCLC analog)

- Demographics: 71 F, limited-stage SCLC
- ECOG: 2, vitals: BP 144/86, HR 88, SpO2 94%
- eConsent: Continuous-trial v3.4 with RTCT addendum (signed 30 Apr 2026)
- Procedure: Robotic ultrasound + thoracic CT baseline imaging
- Lung lesion: 22 mm x 19 mm primary, 2 mediastinal nodes (8 mm, 6 mm)
- Disposition at 00:59: Recovery bay 2, awaiting digital twin update
- RTCT endpoint streamed: STREAM-CONT-0002-IMG01

### PAT-CONT-0003 (SITE-A)

- Demographics: 58 M, NSCLC adenocarcinoma stage IIIA
- ECOG: 1, vitals: BP 128/74, HR 70, SpO2 97%
- Treatment: RT fraction 8 of 30, 2 Gy delivered to LUL lesion
- Outcome: No interruptions, gating efficiency 95.1%
- Disposition at 00:59: Discharged-pending, post-RT obs through 01:30
- RTCT endpoint streamed: TRAVERSE-CONT-0003-RT08

### PAT-CONT-0004 (SITE-B)

- Demographics: 49 F, soft-tissue sarcoma stage III
- ECOG: 1, vitals: BP 118/72, HR 76, SpO2 99%
- Procedure: Cobot-guided core biopsy (in progress at hour close)
- Disposition at 00:59: On COBOT-02, biopsy continues into hour 01

## Carry-Over Patients (Overnight Census)

| Patient ID | Site | Cancer Type | Status | Since |
|------------|------|-------------|--------|-------|
| PAT-CONT-0011 | SITE-A | Mediastinal tumor | Post-surgical recovery | 22:30 |
| PAT-CONT-0012 | SITE-A | Pediatric ALL | Companion monitoring | 21:00 |
| PAT-CONT-0013 | SITE-A | HCC | Post-ablation observation | 20:15 |
| PAT-CONT-0014 | SITE-A | NSCLC | Inter-fraction observation | 23:45 |
| PAT-CONT-0015 | SITE-A | Soft-tissue sarcoma | Post-biopsy observation | 23:00 |
| PAT-CONT-0016 | SITE-A | Glioblastoma | Pre-RT prep | 22:50 |
| PAT-CONT-0017 | SITE-B | Mantle cell lymphoma | Inpatient enrollment hold | 19:00 |
| PAT-CONT-0018 | SITE-B | Parotid tumor | Post-needle observation | 22:10 |
| PAT-CONT-0019 | SITE-B | Pediatric osteosarcoma | Rehab continuation | 21:30 |
| PAT-CONT-0020 | SITE-C | SCLC | Pre-imaging fasting | 22:00 |
| PAT-CONT-0021 | SITE-C | NSCLC | Post-RT recovery | 23:20 |

## Vitals Sampling

All 15 patients have continuous vitals telemetry (HR, SpO2, RR sampled every
60 s; BP every 15 min; temperature every 30 min). All readings within normal
ranges or within patient-specific physician-defined tolerance.

## Pediatric Patients

- PAT-CONT-0012 (SITE-A): 7 yr male, ALL, COMPN-03 monitoring all hour.
- PAT-CONT-0019 (SITE-B): 12 yr female, osteosarcoma, REHAB cycle complete
  at 22:00 prior; sleeping all hour, vitals nominal.

Both pediatric patients had legal guardian consent and IRB review per
21 CFR Part 50 Subpart D - Adaption.

## Wait Times

- PAT-CONT-0001: 1 min from check-in to robot positioning
- PAT-CONT-0002: 1 min from check-in to imaging bay
- PAT-CONT-0003: 1 min from check-in to vault
- PAT-CONT-0004: 2 min from check-in to biopsy station

Average wait time hour 00: 1.25 minutes.
