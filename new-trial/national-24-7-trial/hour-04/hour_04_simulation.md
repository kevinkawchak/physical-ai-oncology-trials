# Hour 04: 04:00-04:59 - Maintenance Window and Pre-Dawn Quiet

Released on 1 May 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 04 includes the scheduled COBOT-03 (SITE-A) preventive calibration and a
brief late-overnight imaging session at SITE-D. No IND administrations.
Network utilization remains low.

## Cross-Site Status at 04:00 UTC

| Site | Patients | Active | Standby | Active | Signal |
|------|----------|--------|---------|--------|--------|
| SITE-A | 6 | 0 | 28 | 1 (cal) | UP (139 ms) |
| SITE-B | 5 | 0 | 29 | 0 | UP (161 ms) |
| SITE-C | 4 | 0 | 29 | 0 | UP (199 ms) |
| SITE-D | 3 | 0 | 29 | 0 | UP (121 ms) |

## New Arrivals

| Patient ID | Site | Time | Age | Sex | Cancer | Stage | Robot |
|------------|------|------|-----|-----|--------|-------|-------|
| PAT-CONT-0010 | SITE-D | 04:35 | 9 | F | Pediatric osteosarcoma | II | Imaging (8) |

## Minute Activity Log

```
04:00 Hour-boundary FDA recon: 2/2 endpoints accepted from hour 03.
04:00-04:18 COBOT-03 (SITE-A) preventive calibration. 6-axis sweep, force
            sensor recal, 18 minutes.
04:19 COBOT-03 returns to standby. Calibration log uploaded.
04:20 Calibration endpoint streamed: NETWORK-OBS-COBOT-03-CAL.
04:21-04:34 Network quiet.
04:35 PAT-CONT-0010 check-in at SITE-D pediatric kiosk.
04:36 IMAGE-03 (SITE-D) wakes. Pediatric mode enabled (low-noise, soft).
04:37-04:50 Femur imaging session. 3D ultrasound + MR fusion.
04:51 Image quality 8.9/10. Lesion measured 64 mm x 28 mm.
04:52 PED-CONT-0010-IMG endpoint streamed. FDA ack 04:52+11s.
04:53 PAT-CONT-0010 to pediatric ward bay 3 (SITE-D), guardian present.
04:54-04:59 Network quiet. Hour 04 close.
```

## Departures
None.

## Adverse Events
None.

## IND Administrations
None.

## Cross-Site Utilization
- SITE-A: 18 robot-active minutes (calibration counted as managed activity)
- SITE-D: 16 robot-active minutes (IMAGE-03 + arrival)
- Network utilization: 0.49%

## Regulatory Compliance
- ICH E6(R3) section 4.5.1: Equipment maintenance log entry created
  automatically by COBOT-03 firmware; cross-checked by SSO at 04:19.
- 21 CFR Part 50 Subpart D: Pediatric assent verified for PAT-CONT-0010.
- 21 CFR Part 312.50: General responsibilities of investigator met.

## RTCT Signal Stream

| Signal ID | Patient | Endpoint | FDA Ack |
|-----------|---------|----------|---------|
| NETWORK-OBS-COBOT-03-CAL | (system) | Calibration complete | 14 s |
| PED-CONT-0010-IMG | PAT-CONT-0010 | Femur imaging baseline | 11 s |
