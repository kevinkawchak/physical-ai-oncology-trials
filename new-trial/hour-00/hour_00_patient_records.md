# Hour 00 Patient Records: 00:00-00:59

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Patients On-Site This Hour: 5

## PAT-ODMND-0001

- Demographics: 52 years, Male
- Cancer: NSCLC adenocarcinoma, Stage IIIA, left upper lobe
- ECOG: 1
- Primary robot: RT Motion-Tracking (Robot 7)
- Arrival: 00:12 (patient-chosen overnight slot)
- Location: Check-in (00:12), Waiting (00:13-00:19), Vault 2 (00:20-00:38),
  Discharge (00:42)
- Procedure: RT fraction 12/30 (2 Gy), completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0312), including
  Physical AI disclosure per 21 CFR 50.25
- Digital twin: Lung tumor model updated with fraction 12 dose data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 00:12 | 74 | 132/78 | 97% | 16 | Check-in baseline |
| 00:17 | 72 | 130/76 | 97% | 15 | Waiting area |
| 00:20 | 76 | 134/80 | 97% | 16 | Positioned on couch |
| 00:21 | 75 | 133/79 | 97% | 15 | Calibration |
| 00:22 | 77 | 135/80 | 96% | 16 | Beam-on field 1 |
| 00:23 | 76 | 134/79 | 97% | 15 | Treatment |
| 00:24 | 75 | 133/78 | 97% | 15 | Treatment |
| 00:25 | 76 | 134/79 | 97% | 16 | Treatment |
| 00:26 | 75 | 132/78 | 97% | 15 | Field 1 complete |
| 00:27 | 74 | 131/77 | 97% | 15 | Gantry rotation |
| 00:28 | 76 | 133/79 | 97% | 16 | Beam-on field 2 |
| 00:29 | 78 | 136/81 | 96% | 17 | Deep breath (gated) |
| 00:30 | 76 | 134/79 | 97% | 16 | Treatment |
| 00:31 | 75 | 133/78 | 97% | 15 | Treatment |
| 00:32 | 74 | 132/77 | 97% | 15 | Field 2 complete |
| 00:33 | 74 | 131/77 | 97% | 15 | Gantry rotation |
| 00:34 | 75 | 133/78 | 97% | 16 | Beam-on field 3 |
| 00:35 | 76 | 134/79 | 97% | 16 | Treatment |
| 00:36 | 75 | 132/78 | 97% | 15 | Field 3 complete |
| 00:37 | 73 | 130/76 | 97% | 14 | Post-treatment |
| 00:38 | 72 | 128/75 | 98% | 14 | Exit vault |

### RT-Specific Metrics
- Breathing amplitude: 4.1 mm average
- Marker displacement: 1.8 mm average
- Beam gating efficiency: 94.2%
- Dose delivered: 2.000 Gy (cumulative 24.0 Gy of planned 60.0 Gy)
- Treatment interruptions: 0
- Patient satisfaction: Not formally assessed (overnight abbreviated discharge)

## PAT-ODMND-0002

- Demographics: 67 years, Female
- Cancer: Hepatocellular carcinoma (HCC), Stage II
- ECOG: 1
- Primary robot: Imaging Assistant (Robot 8)
- Arrival: 00:38 (patient-chosen overnight slot)
- Location: Check-in (00:38), Waiting (00:39-00:44), Imaging Bay 2
  (00:45-00:58), Discharge (post-hour)
- Procedure: Pre-ablation liver ultrasound assessment, completed
- Informed consent: Previously completed (consent ID IC-2026-0298)
- Digital twin: HCC tumor model initialized with new imaging data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 00:38 | 68 | 142/86 | 96% | 14 | Check-in baseline |
| 00:43 | 66 | 140/84 | 96% | 14 | Waiting area |
| 00:45 | 70 | 144/87 | 96% | 15 | Positioned, gel applied |
| 00:46 | 69 | 143/86 | 96% | 14 | Probe contact |
| 00:47 | 68 | 142/85 | 96% | 14 | Scanning |
| 00:48 | 67 | 141/84 | 97% | 14 | Scanning |
| 00:49 | 70 | 144/87 | 96% | 15 | Motion artifact (cough) |
| 00:50 | 68 | 142/85 | 96% | 14 | Scanning resumed |
| 00:51 | 67 | 141/84 | 97% | 14 | Scanning |
| 00:52 | 67 | 140/84 | 97% | 14 | Scanning |
| 00:53 | 69 | 143/86 | 96% | 15 | Motion artifact (shift) |
| 00:54 | 68 | 142/85 | 96% | 14 | Scanning resumed |
| 00:55 | 67 | 141/84 | 97% | 14 | Scanning |
| 00:56 | 66 | 140/83 | 97% | 14 | Scan complete |
| 00:57 | 65 | 138/82 | 97% | 13 | Probe retracted |
| 00:58 | 65 | 138/82 | 97% | 13 | Gel removed |

### Imaging-Specific Metrics
- Probe pressure: 1.8 N average
- Image quality score: 8.2/10
- Primary tumor: 34 x 28 mm
- Secondary lesion: 12 x 9 mm
- Scan coverage: 92%
- Motion artifacts: 2 (auto-compensated)

## PAT-ODMND-0003 (Overnight Recovery)

- Demographics: 61 years, Male
- Cancer: Mediastinal (thymus) tumor, Stage II
- ECOG: 1
- Status: Post-surgical recovery (robotic thoracoscopy completed 20:15
  prior day)
- Location: Recovery Bay 3 (full hour)
- Informed consent: Completed (consent ID IC-2026-0287)

### Vital Signs (5-minute intervals, recovery monitoring)

| Time | HR | BP | SpO2 | RR | Temp | Pain | Notes |
|------|----|----|------|----|------|------|-------|
| 00:00 | 82 | 128/74 | 95% | 18 | 36.8 | 3/10 | Sleeping |
| 00:05 | 80 | 126/72 | 96% | 17 | 36.7 | 3/10 | Sleeping |
| 00:10 | 78 | 125/71 | 96% | 16 | 36.7 | 2/10 | Sleeping |
| 00:15 | 79 | 126/72 | 96% | 17 | 36.8 | 2/10 | Sleeping |
| 00:20 | 80 | 127/73 | 96% | 17 | 36.8 | 3/10 | Brief wake |
| 00:25 | 78 | 125/71 | 96% | 16 | 36.7 | 2/10 | Sleeping |
| 00:30 | 77 | 124/70 | 96% | 16 | 36.7 | 2/10 | Sleeping |
| 00:35 | 78 | 125/71 | 96% | 16 | 36.7 | 2/10 | Sleeping |
| 00:40 | 79 | 126/72 | 96% | 17 | 36.8 | 2/10 | Sleeping |
| 00:45 | 77 | 124/70 | 96% | 16 | 36.7 | 2/10 | Sleeping |
| 00:50 | 78 | 125/71 | 96% | 16 | 36.7 | 2/10 | Sleeping |
| 00:55 | 76 | 123/69 | 97% | 15 | 36.6 | 2/10 | Sleeping |

- Surgical drain output: 45 mL this hour (expected range 20-80 mL/hr)
- Blood loss tracking: Cumulative 280 mL (intraoperative 220 mL + 60 mL
  post-op drain)
- Target discharge: 08:00 pending morning assessment

## PAT-ODMND-0004 (Discharged at 00:25)

- Demographics: 44 years, Female
- Cancer: Forearm soft-tissue sarcoma, Grade II
- ECOG: 0
- Status: Post-biopsy observation, discharged at 00:25
- Location: Recovery Bay 7 (00:00-00:25), Discharged

### Vital Signs (5-minute intervals until discharge)

| Time | HR | BP | SpO2 | Pain | Notes |
|------|----|----|------|------|-------|
| 00:00 | 68 | 118/72 | 99% | 1/10 | Biopsy site stable |
| 00:05 | 66 | 116/70 | 99% | 1/10 | No swelling |
| 00:10 | 67 | 117/71 | 99% | 0/10 | Bandage dry |
| 00:15 | 65 | 115/69 | 99% | 0/10 | Cleared for discharge |
| 00:20 | 66 | 116/70 | 99% | 0/10 | Discharge instructions |
| 00:25 | 64 | 114/68 | 99% | 0/10 | Discharged |

- Biopsy sample quality: Grade A (adequate for histopathology)
- Follow-up: Pathology results in 3-5 business days
- Next visit: Scheduled based on pathology findings

## PAT-ODMND-0005 (Pediatric Overnight)

- Demographics: 8 years, Male
- Cancer: Acute lymphoblastic leukemia (ALL)
- ECOG: 1
- Status: Overnight companion monitoring (admitted for morning chemotherapy)
- Location: Pediatric Ward (full hour), COMPN-03 passive monitoring
- Parent/guardian: Mother present in adjacent family area

### Vital Signs (5-minute intervals, pediatric ranges)

| Time | HR | RR | SpO2 | Temp | Notes |
|------|----|----|------|------|-------|
| 00:00 | 78 | 18 | 98% | 36.5 | Sleeping |
| 00:05 | 76 | 17 | 98% | 36.5 | Sleeping |
| 00:10 | 80 | 19 | 98% | 36.6 | Brief stir |
| 00:15 | 77 | 18 | 98% | 36.5 | Sleeping |
| 00:20 | 75 | 17 | 98% | 36.5 | Sleeping |
| 00:25 | 78 | 18 | 98% | 36.5 | Sleeping |
| 00:30 | 76 | 17 | 98% | 36.5 | Sleeping |
| 00:35 | 79 | 18 | 98% | 36.5 | Sleeping |
| 00:40 | 77 | 17 | 98% | 36.5 | Sleeping |
| 00:45 | 76 | 17 | 98% | 36.5 | Sleeping |
| 00:50 | 78 | 18 | 98% | 36.5 | Sleeping |
| 00:55 | 75 | 17 | 98% | 36.5 | Sleeping |

- Anxiety score: N/A (sleeping)
- Morning plan: Companion robot session (COMPN-03) before chemotherapy
  administration per 21 CFR Part 50 Subpart D pediatric protections
