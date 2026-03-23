# Hour 09 Patient Records: 09:00-09:59

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Patients On-Site This Hour: approximately 28 (peak concurrent)

## PAT-ODMND-0024 (Surgery Completion and Recovery)

- Demographics: Age not specified (prior hour patient), Male
- Cancer: Mediastinal tumor, Stage II
- ECOG: 1
- Primary robot: Surgical Robot (Robot 1), SURG-01
- Location: Surgical Suite 1 (09:00-09:10), Recovery Bay 1 (09:12-09:59)
- Procedure: Robotic thoracoscopic resection, completed at 09:10
- Outcome: R0 resection, negative margins, blood loss 180 mL
- Informed consent: Previously completed (consent ID IC-2026-0524)
- Digital twin: Mediastinal tumor model updated with surgical outcome data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Temp | Pain | Notes |
|------|----|----|------|----|------|------|-------|
| 09:00 | 70 | 112/68 | 99% | 12 | 36.4 | - | Under anesthesia, closure |
| 09:05 | 72 | 114/70 | 99% | 12 | 36.5 | - | Final hemostasis |
| 09:10 | 74 | 110/66 | 98% | 14 | 36.6 | - | Procedure complete |
| 09:12 | 78 | 108/64 | 97% | 16 | 36.5 | 4/10 | Extubated, Recovery Bay 1 |
| 09:15 | 82 | 98/60 | 97% | 18 | 36.4 | 4/10 | BP declining |
| 09:18 | 86 | 92/58 | 96% | 19 | 36.3 | 5/10 | ALERT: hypotension |
| 09:19 | 88 | 94/60 | 96% | 18 | 36.3 | 5/10 | IV bolus initiated |
| 09:22 | 84 | 98/62 | 97% | 17 | 36.4 | 4/10 | BP trending up |
| 09:25 | 80 | 108/68 | 97% | 16 | 36.5 | 4/10 | BP recovered |
| 09:28 | 78 | 118/72 | 98% | 15 | 36.5 | 3/10 | Stabilized |
| 09:35 | 76 | 120/74 | 98% | 15 | 36.6 | 3/10 | Stable |
| 09:40 | 74 | 122/76 | 98% | 14 | 36.6 | 3/10 | Stable |
| 09:45 | 74 | 124/76 | 98% | 14 | 36.6 | 2/10 | Pain improving |
| 09:50 | 72 | 122/74 | 98% | 14 | 36.6 | 2/10 | Stable |
| 09:55 | 72 | 120/74 | 98% | 14 | 36.6 | 2/10 | Stable |

### Adverse Event Record
- AE-009-001: Hypotension Grade 1 at 09:18. BP 92/58 mmHg.
- Intervention: 500 mL NS IV bolus.
- Resolution: BP stabilized within 10 minutes.
- Monitoring increased to every 5 minutes for 2 hours per protocol.
- Reported per ICH E6(R3) Section 2.10 and 21 CFR 312.32.

### Surgical Outcome
- R0 resection confirmed (negative margins)
- Blood loss: 180 mL total
- Procedure duration: approximately 150 minutes
- Chest drain output: 25 mL first hour post-op
- Target discharge: 24-48 hours pending recovery assessment

## PAT-ODMND-0032 (Surgery Ongoing)

- Demographics: Prior hour patient, details in prior hour records
- Cancer: Solid tumor (ongoing resection)
- ECOG: 1
- Primary robot: SURG-02 (Surgical Suite 2)
- Location: Surgical Suite 2 (full hour, surgery started 08:15)
- Status: Main resection phase, stable

### Vital Signs (Under Anesthesia)

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:00 | 70 | 120/74 | 99% | 12 | Stable, resection ongoing |
| 09:15 | 72 | 122/76 | 99% | 12 | Stable |
| 09:30 | 68 | 118/72 | 99% | 12 | Stable |
| 09:45 | 74 | 126/78 | 99% | 12 | Minor hemostasis maneuver |
| 09:59 | 70 | 120/74 | 99% | 12 | Stable, continuing next hour |

- Blood loss this hour: approximately 60 mL (cumulative approximately 140 mL)
- Expected completion: approximately 10:00

## PAT-ODMND-0044

- Demographics: 60 years, Male
- Cancer: Mediastinal tumor, Stage II
- ECOG: 1
- Primary robot: Surgical Robot (Robot 1), SURG-03
- Arrival: 09:00 (patient-chosen morning slot)
- Location: Check-in (09:00), Pre-op (09:02-09:10), Surgical Suite 3
  (09:10-09:59, ongoing)
- Procedure: Robotic thoracoscopic mediastinal tumor resection (ongoing)
- Informed consent: Previously completed (consent ID IC-2026-0612),
  including Physical AI disclosure per 21 CFR 50.25
- Digital twin: Mediastinal tumor model loaded, surgical plan active

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:00 | 76 | 138/84 | 97% | 16 | Check-in baseline |
| 09:05 | 74 | 136/82 | 97% | 15 | Pre-op, safety matrix |
| 09:10 | 68 | 124/76 | 99% | 12 | Anesthesia induction |
| 09:15 | 66 | 118/72 | 99% | 12 | First port, surgery start |
| 09:20 | 68 | 120/74 | 99% | 12 | Tumor identification |
| 09:30 | 70 | 122/76 | 99% | 12 | Dissection in progress |
| 09:45 | 72 | 124/78 | 99% | 12 | Tumor mobilization |
| 09:59 | 70 | 122/76 | 99% | 12 | Ongoing |

## PAT-ODMND-0045

- Demographics: 35 years, Female
- Cancer: Forearm soft-tissue sarcoma, Grade II
- ECOG: 0
- Primary robot: Cobot (Robot 2), COBOT-03
- Arrival: 09:04
- Location: Check-in (09:04), Waiting (09:06-09:11), Biopsy Station 3
  (09:12-09:30), Recovery Bay 4 (09:32-09:52), Discharged (09:55)
- Procedure: Cobot-guided needle biopsy, completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0614)

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:04 | 72 | 118/74 | 99% | 14 | Check-in |
| 09:12 | 76 | 122/76 | 99% | 15 | Positioned for biopsy |
| 09:15 | 78 | 124/78 | 99% | 16 | Local anesthesia applied |
| 09:18 | 74 | 120/76 | 99% | 14 | Core 1 obtained |
| 09:22 | 74 | 120/74 | 99% | 14 | Core 2 and 3 obtained |
| 09:26 | 72 | 118/74 | 99% | 14 | Core 4 obtained |
| 09:30 | 70 | 116/72 | 99% | 13 | Biopsy complete |
| 09:40 | 68 | 114/70 | 99% | 13 | Recovery, bandage dry |
| 09:50 | 66 | 112/68 | 99% | 13 | Cleared for discharge |
| 09:55 | 66 | 112/68 | 99% | 13 | Discharged |

### Biopsy-Specific Metrics
- Force applied: 2.8 N insertion
- Needle trajectory accuracy: 0.3 mm deviation
- Cores obtained: 4 (Grade A quality)
- Bleeding: None
- Follow-up: Pathology results 3-5 business days

## PAT-ODMND-0046

- Demographics: 68 years, Female
- Cancer: Meningioma, Stage I
- ECOG: 0
- Primary robot: RT Positioning (Robot 3), RTPOS-02
- Arrival: 09:07
- Location: Check-in (09:07), Waiting (09:09-09:15), Vault 2 (09:16-09:42),
  Discharged (09:50)
- Procedure: Stereotactic mask fitting and CT simulation
- Informed consent: Previously completed (consent ID IC-2026-0616)
- Digital twin: Meningioma brain model created with CT simulation data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:07 | 64 | 148/88 | 97% | 14 | Check-in (mild HTN baseline) |
| 09:16 | 66 | 146/86 | 97% | 14 | Vault, mask fitting starts |
| 09:20 | 68 | 148/88 | 97% | 15 | Mask molding (mild anxiety) |
| 09:25 | 66 | 144/86 | 97% | 14 | Mask set, CT prep |
| 09:30 | 64 | 142/84 | 98% | 14 | CT scan in progress |
| 09:35 | 64 | 140/82 | 98% | 13 | CT complete |
| 09:42 | 62 | 138/82 | 98% | 13 | Session complete |
| 09:50 | 62 | 138/80 | 98% | 13 | Discharged |

### RT Positioning Metrics
- Mask fit accuracy: 0.4 mm
- CT slices: 124 at 1 mm thickness
- Tumor volume: 8.2 cm3
- Tumor location: Right frontoparietal convexity
- Treatment plan: To be generated (stereotactic radiosurgery)

## PAT-ODMND-0047

- Demographics: 7 years, Female
- Cancer: Pediatric acute myeloid leukemia (AML)
- ECOG: 1
- Primary robot: Social Companion (Robot 5), COMPN-05
- Arrival: 09:10
- Location: Check-in (09:10), Pediatric Waiting (09:12-09:17),
  Play Area 5 (09:18-09:48), Pediatric Ward (09:50-09:59)
- Session: Pre-chemotherapy anxiety management
- Informed consent: Parent/guardian consent completed (consent ID IC-2026-0618)
  per 21 CFR Part 50 Subpart D
- Parent/guardian: Mother present throughout

### Vital Signs (Pediatric Ranges)

| Time | HR | RR | SpO2 | Temp | Anxiety | Notes |
|------|----|----|------|------|---------|-------|
| 09:10 | 110 | 24 | 98% | 36.8 | 7/10 | Check-in, visibly anxious |
| 09:18 | 108 | 22 | 98% | 36.8 | 7/10 | Companion session starts |
| 09:22 | 104 | 22 | 98% | 36.7 | 6/10 | Interactive storytelling |
| 09:28 | 100 | 20 | 99% | 36.7 | 5/10 | Breathing exercises |
| 09:35 | 96 | 20 | 99% | 36.7 | 4/10 | Procedure preview game |
| 09:42 | 94 | 18 | 99% | 36.7 | 3/10 | Engaged, calm |
| 09:48 | 92 | 18 | 99% | 36.7 | 3/10 | Session complete |
| 09:55 | 94 | 18 | 99% | 36.7 | 3/10 | Pediatric ward, awaiting chemo |

## PAT-ODMND-0048

- Demographics: 55 years, Male
- Cancer: NSCLC adenocarcinoma, Stage IIIA, left upper lobe
- ECOG: 1
- Primary robot: RT Motion-Tracking (Robot 7), TRACK-01
- Arrival: 09:14
- Location: Check-in (09:14), Waiting (09:16-09:21), Vault 1 (09:22-09:42),
  Discharged (09:50)
- Procedure: RT fraction 8/30 (2 Gy), completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0620)
- Digital twin: Lung tumor model updated with fraction 8 dose data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:14 | 78 | 134/82 | 96% | 16 | Check-in |
| 09:22 | 80 | 136/84 | 96% | 17 | Positioned, marker placed |
| 09:24 | 78 | 134/82 | 96% | 16 | Calibration complete |
| 09:26 | 80 | 136/84 | 96% | 17 | Beam-on field 1 |
| 09:30 | 78 | 134/82 | 97% | 16 | Field 1 complete |
| 09:32 | 78 | 133/81 | 97% | 16 | Field 2 in progress |
| 09:36 | 76 | 132/80 | 97% | 15 | Field 2 complete |
| 09:38 | 78 | 134/82 | 97% | 16 | Field 3 in progress |
| 09:40 | 76 | 132/80 | 97% | 15 | Field 3 complete |
| 09:42 | 74 | 130/78 | 97% | 15 | Exit vault |
| 09:50 | 72 | 128/76 | 97% | 14 | Discharged |

### RT-Specific Metrics
- Breathing amplitude: 3.8 mm
- Marker displacement: 1.6 mm average
- Beam gating efficiency: 93.8%
- Dose delivered: 2.000 Gy (cumulative 16.0 Gy of 60.0 Gy)
- Treatment interruptions: 0

## PAT-ODMND-0049

- Demographics: 43 years, Female
- Cancer: Parotid tumor, Stage II
- ECOG: 0
- Primary robot: Needle-Placement (Robot 4), NEEDLE-02
- Arrival: 09:18
- Location: Check-in (09:18), Waiting (09:20-09:27), CT Suite 2
  (09:28-09:58), Recovery Bay 6 (09:59, continuing next hour)
- Procedure: CT-guided needle placement for parotid biopsy
- Informed consent: Previously completed (consent ID IC-2026-0622)

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:18 | 70 | 122/76 | 99% | 14 | Check-in |
| 09:28 | 74 | 126/78 | 99% | 15 | Positioned, CT suite |
| 09:32 | 76 | 128/80 | 99% | 15 | Local anesthesia |
| 09:36 | 74 | 126/78 | 99% | 15 | First CT verification |
| 09:40 | 72 | 124/76 | 99% | 14 | Needle advancing |
| 09:45 | 74 | 126/78 | 99% | 15 | Second CT verification |
| 09:50 | 72 | 124/76 | 99% | 14 | Sample obtained |
| 09:55 | 70 | 122/76 | 99% | 14 | Needle withdrawn |
| 09:58 | 68 | 120/74 | 99% | 14 | Procedure complete |

### Needle Placement Metrics
- Placement accuracy: 0.5 mm from planned trajectory
- CT verification scans: 3
- Facial nerve proximity: 4.2 mm (safe margin)
- Sample quality: Adequate tissue for histopathology

## PAT-ODMND-0050

- Demographics: 61 years, Male
- Cancer: Hepatocellular carcinoma (HCC), Stage II
- ECOG: 1
- Primary robot: Imaging Assistant (Robot 8), IMAGE-01
- Arrival: 09:22
- Location: Check-in (09:22), Waiting (09:24-09:29), Imaging Bay 1
  (09:30-09:48), Discharged (09:55)
- Procedure: Robotic ultrasound liver assessment
- Informed consent: Previously completed (consent ID IC-2026-0624)
- Digital twin: HCC liver model updated with imaging data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:22 | 72 | 140/84 | 96% | 15 | Check-in |
| 09:30 | 70 | 138/82 | 97% | 14 | Positioned, gel applied |
| 09:35 | 68 | 136/80 | 97% | 14 | Scanning |
| 09:40 | 68 | 134/80 | 97% | 14 | Scanning |
| 09:45 | 66 | 132/78 | 97% | 13 | Scan complete |
| 09:48 | 66 | 132/78 | 97% | 13 | Session complete |
| 09:55 | 64 | 130/76 | 97% | 13 | Discharged |

### Imaging-Specific Metrics
- Probe pressure: 1.9 N average
- Image quality score: 8.4/10
- Primary tumor: 32 x 26 mm
- Scan coverage: 94%
- Motion artifacts: 1 (auto-compensated)

## PAT-ODMND-0051

- Demographics: 15 years, Male
- Cancer: Pediatric osteosarcoma (knee region)
- ECOG: 1
- Primary robots: Humanoid (Robot 6) HUMAN-01, then Rehab Exoskeleton
  (Robot 10) REHAB-01
- Arrival: 09:25
- Location: Check-in (09:25), Pediatric Waiting (09:27-09:32), Therapy
  Station 1 (09:33-09:48), Rehabilitation Bay 1 (09:50-09:59, continuing)
- Session: Physical therapy preparation then exoskeleton rehab
- Informed consent: Parent/guardian consent (consent ID IC-2026-0626)
  per 21 CFR Part 50 Subpart D
- Parent/guardian: Father present

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:25 | 88 | 116/72 | 98% | 18 | Check-in |
| 09:33 | 92 | 118/74 | 98% | 18 | Humanoid session start |
| 09:38 | 96 | 122/76 | 98% | 20 | ROM assessment |
| 09:42 | 94 | 120/74 | 98% | 19 | Gait analysis |
| 09:48 | 90 | 118/72 | 98% | 18 | Humanoid session complete |
| 09:50 | 94 | 120/74 | 98% | 20 | Exoskeleton fitting |
| 09:55 | 100 | 124/78 | 98% | 22 | Walking with exoskeleton |
| 09:59 | 98 | 122/76 | 98% | 21 | Continuing next hour |

### Humanoid and Rehab Metrics
- Knee flexion ROM: 82 degrees (post-surgical baseline)
- Weight-bearing: 60%
- Gait analysis: Antalgic pattern
- Exoskeleton body weight support: 70%
- Steps completed by 09:59: 42

## PAT-ODMND-0052

- Demographics: 74 years, Female
- Cancer: Colorectal liver metastases, Stage IV
- ECOG: 2
- Primary robot: Steerable Needle (Robot 9), STEER-02
- Arrival: 09:28
- Location: Check-in (09:28), Waiting (09:30-09:37), Ablation Suite 2
  (09:38-09:59, continuing next hour)
- Procedure: CT-guided steerable needle microwave ablation (ongoing)
- Informed consent: Previously completed (consent ID IC-2026-0628)
- Enhanced monitoring active due to ECOG 2 per ICH E6(R3) Section 2.10

### Vital Signs

| Time | HR | BP | SpO2 | RR | Temp | Notes |
|------|----|----|------|----|------|-------|
| 09:28 | 82 | 152/92 | 95% | 18 | 36.6 | Check-in (HTN baseline) |
| 09:38 | 80 | 148/90 | 96% | 16 | 36.6 | Positioned, sedation start |
| 09:42 | 76 | 138/84 | 97% | 14 | 36.6 | Sedated, CT scan |
| 09:45 | 74 | 136/82 | 97% | 14 | 36.7 | Needle insertion |
| 09:48 | 76 | 138/84 | 97% | 14 | 36.8 | Needle at target |
| 09:52 | 74 | 136/82 | 97% | 14 | 37.0 | Ablation started |
| 09:55 | 76 | 138/84 | 96% | 15 | 37.2 | Ablation ongoing |
| 09:59 | 74 | 136/82 | 96% | 14 | 37.4 | Target temp 65 C |

### Steerable Needle Metrics
- Needle insertion accuracy: 0.4 mm from plan
- Target lesion: 2.4 cm metastatic deposit, right hepatic lobe
- Ablation zone target: 3.4 cm (1 cm margin circumferential)
- Status at 09:59: Ablation in progress, continuing next hour

## PAT-ODMND-0053

- Demographics: 50 years, Male
- Cancer: NSCLC squamous cell carcinoma, Stage IIIA, right hilum
- ECOG: 1
- Primary robot: RT Motion-Tracking (Robot 7), TRACK-02
- Arrival: 09:32
- Location: Check-in (09:32), Waiting (09:34-09:39), Vault 3 (09:40-09:58),
  Discharged (09:59)
- Procedure: RT fraction 5/30 (2 Gy), completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0630)
- Digital twin: Lung tumor model updated with fraction 5 dose data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:32 | 80 | 142/86 | 95% | 18 | Check-in |
| 09:40 | 82 | 144/88 | 95% | 18 | Positioned, marker placed |
| 09:42 | 80 | 142/86 | 96% | 17 | Calibration complete |
| 09:44 | 82 | 144/88 | 96% | 18 | Beam-on field 1 |
| 09:48 | 80 | 142/86 | 96% | 17 | Field 2 in progress |
| 09:52 | 78 | 140/84 | 96% | 16 | Field 2 complete |
| 09:54 | 80 | 142/86 | 96% | 17 | Field 3 in progress |
| 09:56 | 78 | 140/84 | 96% | 16 | Field 3 complete |
| 09:58 | 76 | 138/82 | 96% | 16 | Exit vault |

### RT-Specific Metrics
- Breathing amplitude: 4.2 mm
- Marker displacement: 1.9 mm average
- Beam gating efficiency: 94.5%
- Dose delivered: 2.000 Gy (cumulative 10.0 Gy of 60.0 Gy)
- Treatment interruptions: 0

## PAT-ODMND-0054

- Demographics: 46 years, Female
- Cancer: Brain metastases (breast primary), Stage IV
- ECOG: 1
- Primary robot: RT Positioning (Robot 3), RTPOS-03
- Arrival: 09:36
- Location: Check-in (09:36), Waiting (09:38-09:43), Vault 3 (09:44-09:59,
  continuing next hour)
- Procedure: Stereotactic frame fitting and CT simulation (ongoing)
- Informed consent: Previously completed (consent ID IC-2026-0632)
- Digital twin: Brain metastases model being created

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:36 | 74 | 128/78 | 98% | 15 | Check-in |
| 09:44 | 78 | 132/80 | 98% | 16 | Frame fitting starts |
| 09:48 | 76 | 130/78 | 98% | 15 | Frame secured |
| 09:52 | 74 | 128/78 | 98% | 14 | CT scan in progress |
| 09:56 | 74 | 126/76 | 98% | 14 | CT ongoing |
| 09:59 | 72 | 126/76 | 98% | 14 | Continuing next hour |

### RT Positioning Metrics
- Frame placement accuracy: 0.3 mm
- CT simulation: In progress at hour end

## PAT-ODMND-0055

- Demographics: 28 years, Male
- Cancer: Forearm soft-tissue sarcoma, Grade I
- ECOG: 0
- Primary robot: Cobot (Robot 2), COBOT-04
- Arrival: 09:40
- Location: Check-in (09:40), Waiting (09:42-09:47), Biopsy Station 4
  (09:48-09:59, continuing next hour)
- Procedure: Cobot-guided needle biopsy (ongoing)
- Informed consent: Previously completed (consent ID IC-2026-0634)

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:40 | 68 | 120/74 | 99% | 14 | Check-in |
| 09:48 | 72 | 124/76 | 99% | 15 | Positioned, local applied |
| 09:52 | 70 | 122/74 | 99% | 14 | Core 1 obtained |
| 09:56 | 70 | 122/74 | 99% | 14 | Core 2 obtained |
| 09:59 | 68 | 120/74 | 99% | 14 | 2 of 4 cores, continuing |

## PAT-ODMND-0056

- Demographics: 64 years, Female
- Cancer: Hepatocellular carcinoma (HCC), Stage III
- ECOG: 1
- Primary robot: Imaging Assistant (Robot 8), IMAGE-02
- Arrival: 09:44
- Location: Check-in (09:44), Waiting (09:46-09:51), Imaging Bay 2
  (09:52-09:59, continuing next hour)
- Procedure: Robotic ultrasound liver assessment (ongoing)
- Informed consent: Previously completed (consent ID IC-2026-0636)

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:44 | 74 | 146/88 | 96% | 16 | Check-in |
| 09:52 | 72 | 142/86 | 96% | 15 | Positioned, scanning starts |
| 09:56 | 70 | 140/84 | 97% | 14 | Scanning in progress |
| 09:59 | 70 | 140/84 | 97% | 14 | Continuing next hour |

## PAT-ODMND-0057

- Demographics: 69 years, Male
- Cancer: Femur osteosarcoma (post-surgical)
- ECOG: 2
- Primary robot: Rehab Exoskeleton (Robot 10), REHAB-02
- Arrival: 09:48
- Location: Check-in (09:48), Waiting (09:50-09:55), Rehabilitation Bay 2
  (09:56-09:59, continuing next hour)
- Session: Post-surgical rehabilitation exoskeleton (ongoing)
- Informed consent: Previously completed (consent ID IC-2026-0638)
- Enhanced monitoring due to ECOG 2 per ICH E6(R3) Section 2.10

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 09:48 | 78 | 148/90 | 96% | 16 | Check-in |
| 09:56 | 82 | 150/92 | 96% | 18 | Exoskeleton fitting |
| 09:59 | 84 | 152/92 | 96% | 18 | Initial calibration |

## PAT-ODMND-0058

- Demographics: 12 years, Female
- Cancer: Pediatric acute lymphoblastic leukemia (ALL)
- ECOG: 1
- Primary robot: Social Companion (Robot 5), COMPN-01
- Arrival: 09:52
- Location: Check-in (09:52), Pediatric Waiting (09:53-09:54),
  Play Area 1 (09:55-09:59, continuing next hour)
- Session: Pre-treatment anxiety management (ongoing)
- Informed consent: Parent/guardian consent (consent ID IC-2026-0640)
  per 21 CFR Part 50 Subpart D
- Parent/guardian: Father present

### Vital Signs (Pediatric Ranges)

| Time | HR | RR | SpO2 | Temp | Anxiety | Notes |
|------|----|----|------|------|---------|-------|
| 09:52 | 96 | 20 | 98% | 36.6 | 6/10 | Check-in |
| 09:55 | 94 | 20 | 98% | 36.6 | 6/10 | Companion session starts |
| 09:59 | 92 | 18 | 99% | 36.6 | 5/10 | Interactive activities |
