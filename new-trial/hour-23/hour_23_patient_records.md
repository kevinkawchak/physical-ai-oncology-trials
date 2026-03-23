# Hour 23 Patient Records: 23:00-23:59

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Patients On-Site This Hour: 4

## Cumulative Unique Patients (24-Hour Cycle): 175

This is the final patient records file for the 24-hour simulation.
PAT-ODMND-0174 and PAT-ODMND-0175 are the last two arrivals, bringing
the total to 175 unique patients (PAT-ODMND-0001 through PAT-ODMND-0175,
including carried-over P0003, P0004, P0005 from the prior day cycle).

## PAT-ODMND-0174

- Demographics: 52 years, Male
- Cancer: Colorectal adenocarcinoma with hepatic metastases, Stage IV
- ECOG: 1
- Primary robot: Imaging Assistant (Robot 8)
- Arrival: 23:15 (patient-chosen overnight slot, work schedule accommodation)
- Location: Check-in (23:15), Waiting (23:17-23:19), Imaging Bay 2
  (23:20-23:42), Discharge (23:50)
- Procedure: Contrast-enhanced abdominal CT, liver metastasis surveillance
  per RECIST 1.1
- Informed consent: Previously completed (consent ID IC-2026-1482), including
  Physical AI disclosure per 21 CFR 50.25 and ICH E6(R2) Section 4.8
- Digital twin: Liver metastasis model updated with new imaging data,
  volumetric measurements refreshed

### Clinical History
- Diagnosed 14 months ago following colonoscopy with biopsy
- Primary tumor resected (right hemicolectomy, 11 months ago)
- 4 hepatic metastases identified at diagnosis (segments IV, V, VII, VIII)
- Currently on FOLFOX chemotherapy, cycle 8 of 12
- Prior imaging (8 weeks ago): stable disease per RECIST 1.1
- No prior radiation therapy
- Comorbidities: hypertension (controlled), type 2 diabetes (HbA1c 6.8%)

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 23:15 | 72 | 138/82 | 97% | 15 | Check-in baseline |
| 23:17 | 70 | 136/80 | 97% | 14 | Waiting area |
| 23:20 | 74 | 140/84 | 97% | 15 | Positioned, arms up |
| 23:22 | 73 | 139/83 | 97% | 15 | Scout scan |
| 23:25 | 72 | 138/82 | 97% | 14 | Non-contrast scan |
| 23:28 | 76 | 142/85 | 96% | 16 | IV contrast injection |
| 23:30 | 78 | 144/86 | 96% | 16 | Arterial phase (warmth) |
| 23:32 | 75 | 141/84 | 97% | 15 | Portal venous phase |
| 23:36 | 72 | 138/82 | 97% | 14 | Delayed phase |
| 23:40 | 70 | 136/80 | 97% | 14 | Scan complete |
| 23:42 | 69 | 135/79 | 98% | 14 | IV removed, seated |
| 23:50 | 68 | 134/78 | 98% | 13 | Discharge |

### Imaging-Specific Metrics
- Contrast agent: Iohexol 100 mL IV at 3.0 mL/s
- Contrast reaction: None (patient tolerated well, mild warmth reported)
- Scan phases: Non-contrast, arterial, portal venous, delayed
- Motion artifacts: 0
- Image quality: Diagnostic, all 4 hepatic lesions clearly delineated
- Target lesion measurements (RECIST 1.1):
  - Segment VII: 2.8 cm (prior 2.9 cm)
  - Segment V: 1.9 cm (prior 2.0 cm)
  - Segment IV: 1.4 cm (prior 1.4 cm)
  - Segment VIII: 1.1 cm (prior 1.1 cm)
  - Sum of target diameters: 7.2 cm (prior 7.4 cm) - stable disease
- New lesions: None identified
- Robotic assistance: IMAGE-02 positioned detector array, optimized
  bolus-tracking threshold, automated phase timing
- Patient satisfaction: Not formally assessed (overnight abbreviated discharge)

## PAT-ODMND-0175

- Demographics: 66 years, Female
- Cancer: NSCLC adenocarcinoma, Stage IIIB, right hilar mass with mediastinal
  lymph node involvement
- ECOG: 1
- Primary robot: RT Motion-Tracking (Robot 7)
- Arrival: 23:40 (patient-chosen overnight slot, night-shift nurse schedule)
- Location: Check-in (23:40), Waiting (23:42-23:47), Vault 2
  (23:48-ongoing), anticipated completion 00:06 next cycle
- Procedure: RT fraction 18/30 (2 Gy), in progress at end of hour
- Informed consent: Previously completed (consent ID IC-2026-1491), including
  Physical AI robotic motion-tracking disclosure per 21 CFR 50.25
- Digital twin: Lung tumor model to be updated upon fraction completion

### Clinical History
- Diagnosed 6 months ago following persistent cough and hemoptysis
- PET-CT confirmed right hilar mass (3.2 cm) with ipsilateral mediastinal
  lymph node involvement (stations 4R, 7)
- Concurrent chemoradiation: carboplatin/paclitaxel with 60 Gy in 30
  fractions
- Prior 17 fractions delivered at this site (34.0 Gy cumulative)
- No surgical candidacy due to mediastinal involvement
- Comorbidities: COPD (mild, FEV1 72% predicted), osteoporosis

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 23:40 | 78 | 128/76 | 95% | 17 | Check-in baseline |
| 23:42 | 76 | 126/74 | 95% | 16 | Waiting area |
| 23:47 | 77 | 127/75 | 95% | 16 | Called to vault |
| 23:48 | 80 | 130/78 | 95% | 17 | Positioned on couch |
| 23:49 | 79 | 129/77 | 95% | 17 | Calibration, markers placed |
| 23:50 | 81 | 131/78 | 95% | 18 | Beam-on field 1 |
| 23:51 | 80 | 130/77 | 95% | 17 | Treatment |
| 23:52 | 79 | 129/76 | 95% | 17 | Treatment |
| 23:53 | 78 | 128/76 | 96% | 16 | Treatment (relaxed) |
| 23:54 | 77 | 127/75 | 96% | 16 | Field 1 complete |
| 23:55 | 76 | 126/74 | 96% | 16 | Gantry rotation |
| 23:56 | 78 | 128/76 | 95% | 17 | Beam-on field 2 |
| 23:57 | 77 | 127/75 | 95% | 17 | Treatment |
| 23:58 | 77 | 127/75 | 96% | 16 | Treatment |
| 23:59 | 76 | 126/74 | 96% | 16 | Treatment (continues) |

### RT-Specific Metrics (Through 23:59)
- Breathing amplitude: 3.6 mm average
- Marker displacement: 1.5 mm average
- Beam gating efficiency: 93.8%
- Dose delivered through 23:59: 1.4 Gy (of planned 2.0 Gy)
- Cumulative dose through fraction 17: 34.0 Gy (of planned 60.0 Gy)
- Treatment interruptions: 0
- SpO2 remained at 95-96% throughout (baseline for COPD patient)
- Treatment extends past midnight into next cycle

## PAT-ODMND-0154 (Continuing - Post-Surgical Recovery)

- Demographics: 58 years, Male
- Cancer: Esophageal adenocarcinoma, Stage IIB
- ECOG: 2 (post-operative)
- Status: Post-surgical recovery, Recovery Bay 3
- Surgery completed: 20:30 this day (Ivor Lewis esophagectomy, SURG-02)
- Recovery since: Hour 20

### Vital Signs (Overnight Monitoring, 1-hour intervals)

| Time | HR | BP | SpO2 | RR | Temp | Pain | Notes |
|------|----|----|------|----|------|------|-------|
| 23:00 | 82 | 118/72 | 96% | 16 | 37.1 | 3/10 | Resting, IV morphine PCA |
| 23:15 | 80 | 116/70 | 96% | 15 | 37.0 | 3/10 | Sleeping |
| 23:30 | 78 | 114/68 | 97% | 14 | 36.9 | 2/10 | Sleeping |
| 23:45 | 79 | 115/69 | 96% | 15 | 37.0 | 3/10 | Brief awakening, sip water |
| 23:59 | 78 | 114/68 | 97% | 14 | 36.9 | 2/10 | Sleeping |

- Drain output (23:00-23:59): 45 mL serosanguinous (acceptable)
- IV fluids: Lactated Ringer's at 125 mL/hr
- Anticipated discharge: Next day, pending surgical team evaluation

## PAT-ODMND-0173 (Continuing - Pediatric Overnight Monitoring)

- Demographics: 9 years, Female
- Cancer: Ewing sarcoma, left femur
- ECOG: 2
- Status: Overnight companion monitoring with COMPN-02
- Admitted: 22:10 this day for pre-chemotherapy observation
- Companion robot: COMPN-02 (nightlight mode, passive monitoring)

### Vital Signs (Overnight Monitoring)

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 23:00 | 88 | 102/62 | 98% | 18 | Sleeping, parent at bedside |
| 23:15 | 86 | 100/60 | 99% | 17 | Sleeping |
| 23:30 | 90 | 104/64 | 98% | 19 | Brief awakening, COMPN-02 soft tone |
| 23:45 | 85 | 100/60 | 99% | 17 | Sleeping |
| 23:59 | 84 | 98/58 | 99% | 16 | Sleeping |

- COMPN-02 interaction: At 23:30, patient briefly woke and COMPN-02 played
  a soft lullaby for 90 seconds. Patient returned to sleep. No distress.
- Parent confirmed comfortable at bedside. Call button accessible.
- Chemotherapy (vincristine/doxorubicin/cyclophosphamide) scheduled for
  09:00 next day.

## Concurrent Patients at 23:59: 4

| Patient | Status | Location | Robot |
|---------|--------|----------|-------|
| PAT-ODMND-0154 | Post-surgical recovery | Recovery Bay 3 | None (telemetry) |
| PAT-ODMND-0173 | Pediatric overnight | Pediatric Ward | COMPN-02 |
| PAT-ODMND-0174 | Discharged 23:50 | -- | -- |
| PAT-ODMND-0175 | RT in progress | Vault 2 | TRACK-02 |

Note: At 23:59, 3 patients remain on-site (P0154, P0173, P0175). P0174
was discharged at 23:50. These patients carry over into the next 24-hour
cycle, analogous to P0003, P0004, and P0005 carrying into this cycle.
