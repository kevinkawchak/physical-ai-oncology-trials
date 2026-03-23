# Hour 11 Patient Records: 11:00-11:59

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Patients On-Site This Hour: approximately 28

## PAT-ODMND-0073

- Demographics: 64 years, Male
- Cancer: NSCLC adenocarcinoma, Stage IIIA, right upper lobe
- ECOG: 1
- Primary robot: RT Motion-Tracking (TRACK-02)
- Arrival: 11:00 (patient-chosen daytime slot)
- Location: Check-in (11:00), Waiting (11:01-11:07), Vault 2 (11:08-11:26),
  Discharge (11:30)
- Procedure: RT fraction delivery (2.0 Gy), completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0587), including
  Physical AI disclosure per 21 CFR 50.25
- Digital twin: Lung tumor model updated with fraction dose data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:00 | 78 | 138/82 | 96% | 17 | Check-in baseline |
| 11:05 | 76 | 136/80 | 96% | 16 | Waiting area |
| 11:08 | 80 | 140/84 | 96% | 17 | Positioned on couch |
| 11:09 | 79 | 139/83 | 96% | 16 | Calibration |
| 11:10 | 81 | 141/84 | 96% | 17 | Beam-on field 1 |
| 11:12 | 80 | 140/83 | 96% | 17 | Treatment |
| 11:14 | 79 | 139/82 | 97% | 16 | Field 1 complete |
| 11:16 | 78 | 138/82 | 97% | 16 | Beam-on field 2 |
| 11:18 | 80 | 140/83 | 96% | 17 | Treatment |
| 11:20 | 79 | 139/82 | 97% | 16 | Field 2 complete |
| 11:22 | 78 | 138/81 | 97% | 16 | Beam-on field 3 |
| 11:24 | 77 | 137/81 | 97% | 16 | Field 3 complete |
| 11:25 | 76 | 136/80 | 97% | 15 | Post-treatment |
| 11:26 | 75 | 134/79 | 97% | 15 | Exit vault |

### RT-Specific Metrics
- Breathing amplitude: 3.9 mm average
- Marker displacement: 1.7 mm average
- Beam gating efficiency: 93.8%
- Dose delivered: 2.000 Gy
- Treatment interruptions: 0
- Patient satisfaction: 8/10

## PAT-ODMND-0074

- Demographics: 45 years, Female
- Cancer: Forearm sarcoma, Stage II, left forearm
- ECOG: 0
- Primary robot: Cobot (COBOT-03)
- Arrival: 11:05 (patient-chosen daytime slot)
- Location: Check-in (11:05), Waiting (11:06-11:12), Biopsy Station 3
  (11:13-11:28), Discharge (11:32)
- Procedure: Core needle biopsy, completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0591)
- Digital twin: Sarcoma model initialized with biopsy location data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:05 | 72 | 118/72 | 99% | 14 | Check-in baseline |
| 11:10 | 74 | 120/74 | 99% | 14 | Waiting area |
| 11:13 | 76 | 122/75 | 99% | 15 | Positioned, local anesthetic applied |
| 11:15 | 78 | 124/76 | 99% | 15 | First core obtained |
| 11:18 | 77 | 123/75 | 99% | 15 | Second core obtained |
| 11:20 | 76 | 122/74 | 99% | 14 | Third core obtained |
| 11:22 | 75 | 121/74 | 99% | 14 | Fourth core obtained |
| 11:25 | 74 | 120/73 | 99% | 14 | Hemostasis, dressing applied |
| 11:28 | 73 | 119/72 | 99% | 14 | Procedure complete |

### Biopsy-Specific Metrics
- Force applied: 2.4 N average
- Tissue cores obtained: 4
- Core length: 12-15 mm (adequate)
- Bleeding: Minimal, controlled with direct pressure
- Dressing applied: Sterile compression bandage

## PAT-ODMND-0075

- Demographics: 69 years, Male
- Cancer: Meningioma, Grade I, right frontal convexity
- ECOG: 0
- Primary robot: RT Positioning (RTPOS-03)
- Arrival: 11:08 (patient-chosen daytime slot)
- Location: Check-in (11:08), Waiting (11:09-11:15), Vault 3 (11:16-11:38),
  Discharge (11:42)
- Procedure: Stereotactic RT positioning and treatment, completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0594)
- Digital twin: Cranial tumor model updated with fraction data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:08 | 66 | 144/86 | 97% | 14 | Check-in baseline |
| 11:12 | 64 | 142/84 | 97% | 14 | Waiting area |
| 11:16 | 68 | 146/87 | 97% | 15 | Mask fitting begins |
| 11:21 | 66 | 144/86 | 97% | 14 | Mask registered, positioning |
| 11:26 | 67 | 145/86 | 97% | 14 | Treatment beam-on |
| 11:30 | 66 | 144/85 | 97% | 14 | Treatment midpoint |
| 11:34 | 65 | 143/85 | 97% | 14 | Treatment near complete |
| 11:36 | 64 | 142/84 | 98% | 13 | Treatment complete |
| 11:38 | 64 | 141/84 | 98% | 13 | Mask removed, exit |

### RT-Positioning Metrics
- Mask registration accuracy: 0.4 mm
- 6-DOF couch alignment: Within 0.3 mm / 0.2 degrees all axes
- Dose delivered: 1.8 Gy (stereotactic fractionation)
- Treatment time: 10 minutes beam-on
- Intrafraction motion: less than 0.5 mm (within tolerance)

## PAT-ODMND-0076

- Demographics: 8 years, Female
- Cancer: Pediatric ALL (acute lymphoblastic leukemia)
- ECOG: 1
- Primary robot: Social Companion (COMPN-04)
- Arrival: 11:12 (parent-scheduled appointment)
- Location: Check-in (11:12 with parent), Pediatric Waiting (11:13-11:14),
  Pediatric Play Area 4 (11:15-11:55)
- Procedure: Companion-supported anxiety management and therapeutic play
- Informed consent: Parental consent (consent ID IC-2026-0597), child assent
  obtained per 21 CFR 50.25 with age-appropriate explanation
- Digital twin: Not applicable (companion session)

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:12 | 98 | 100/62 | 99% | 20 | Check-in, anxious |
| 11:15 | 96 | 98/61 | 99% | 19 | Companion interaction begins |
| 11:20 | 92 | 96/60 | 99% | 18 | Engaged in play |
| 11:25 | 90 | 95/59 | 99% | 18 | Anxiety reducing |
| 11:30 | 88 | 94/58 | 99% | 17 | Calm, engaged |
| 11:35 | 86 | 93/58 | 99% | 17 | Anxiety score 3/10 |
| 11:40 | 88 | 94/58 | 99% | 17 | Active play |
| 11:45 | 87 | 93/58 | 99% | 17 | Continued engagement |
| 11:50 | 86 | 93/57 | 99% | 17 | Preparing for transition |
| 11:55 | 88 | 94/58 | 99% | 18 | Session concluding |

### Companion-Specific Metrics
- Initial anxiety score: 6/10
- Final anxiety score: 3/10 (at 11:35)
- Interaction mode: Therapeutic play (art, storytelling)
- Parent present: Yes, throughout session
- Duration: 40 minutes

## PAT-ODMND-0077

- Demographics: 52 years, Male
- Cancer: Parotid tumor, Stage II, left parotid gland
- ECOG: 0
- Primary robot: Needle-Placement (NEEDLE-02)
- Arrival: 11:16 (patient-chosen daytime slot)
- Location: Check-in (11:16), Waiting (11:17-11:23), CT Suite 2
  (11:24-11:42), Discharge (11:48)
- Procedure: CT-guided fine needle aspiration, completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0600)
- Digital twin: Parotid tumor model initialized with biopsy coordinates

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:16 | 70 | 128/78 | 98% | 15 | Check-in baseline |
| 11:20 | 68 | 126/76 | 98% | 14 | Waiting area |
| 11:24 | 72 | 130/79 | 98% | 15 | Positioned in CT suite |
| 11:26 | 74 | 132/80 | 98% | 16 | Local anesthetic infiltrated |
| 11:28 | 73 | 131/79 | 98% | 15 | Planning CT acquired |
| 11:30 | 74 | 132/80 | 98% | 16 | Needle insertion |
| 11:34 | 76 | 134/81 | 98% | 16 | Aspiration in progress |
| 11:38 | 74 | 132/80 | 98% | 15 | Needle withdrawal |
| 11:40 | 72 | 130/78 | 98% | 15 | Confirmation CT |
| 11:42 | 70 | 128/77 | 98% | 14 | Procedure complete |

### Needle-Placement Metrics
- Trajectory accuracy: 1.1 mm from planned path
- Needle gauge: 22G
- Passes: 2
- Sample adequacy: Confirmed by rapid on-site evaluation
- Complications: None
- Post-procedure CT: No pneumothorax, no hemorrhage

## PAT-ODMND-0078

- Demographics: 60 years, Female
- Cancer: HCC (hepatocellular carcinoma), Stage II
- ECOG: 1
- Primary robot: Imaging Assistant (IMAGE-04)
- Arrival: 11:20 (patient-chosen daytime slot)
- Location: Check-in (11:20), Waiting (11:21-11:27), Imaging Bay 4
  (11:28-11:41), Discharge (11:46)
- Procedure: Robotic ultrasound liver assessment, completed successfully
- Informed consent: Previously completed (consent ID IC-2026-0603)
- Digital twin: HCC tumor model updated with imaging data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:20 | 72 | 140/84 | 96% | 15 | Check-in baseline |
| 11:25 | 70 | 138/82 | 96% | 14 | Waiting area |
| 11:28 | 74 | 142/85 | 96% | 15 | Positioned, gel applied |
| 11:30 | 73 | 141/84 | 96% | 15 | Probe contact, scanning |
| 11:33 | 72 | 140/83 | 97% | 14 | Scanning |
| 11:36 | 71 | 139/83 | 97% | 14 | Motion artifact (cough) |
| 11:38 | 72 | 140/83 | 97% | 14 | Scanning resumed |
| 11:40 | 70 | 138/82 | 97% | 14 | Scan complete |
| 11:41 | 69 | 137/81 | 97% | 13 | Gel removed |

### Imaging-Specific Metrics
- Probe pressure: 1.9 N average
- Image quality score: 8.4/10
- Primary tumor: 31 x 25 mm
- Scan coverage: 94%
- Motion artifacts: 1 (auto-compensated)

## PAT-ODMND-0079

- Demographics: 57 years, Male
- Cancer: Mediastinal tumor, Stage III
- ECOG: 1
- Primary robot: Surgical (SURG-02)
- Arrival: 11:24 (scheduled surgical slot)
- Location: Check-in (11:24), Pre-op (11:25-11:29), Surgical Suite 2
  (11:30-ongoing)
- Procedure: Mediastinal tumor debulking with pre-operative pembrolizumab
  per IND protocol
- Informed consent: Previously completed (consent ID IC-2026-0606),
  including IND drug disclosure and Physical AI surgical system disclosure
  per 21 CFR 50.25 and 21 CFR 312.61
- Digital twin: Mediastinal tumor model loaded, intraoperative updates active

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:24 | 74 | 134/80 | 97% | 16 | Check-in baseline |
| 11:28 | 72 | 132/78 | 97% | 15 | Pre-op holding, pembrolizumab complete |
| 11:30 | 68 | 128/76 | 99% | Vent | General anesthesia induction |
| 11:35 | 66 | 118/72 | 99% | Vent | Surgery commenced |
| 11:40 | 68 | 120/74 | 99% | Vent | Tumor exposure |
| 11:45 | 70 | 122/75 | 99% | Vent | Debulking in progress |
| 11:50 | 69 | 121/74 | 99% | Vent | Debulking continues |
| 11:55 | 68 | 120/73 | 99% | Vent | Stable, ongoing |
| 11:59 | 67 | 119/73 | 99% | Vent | Blood loss 85 mL |

### Surgery-Specific Metrics (Partial - In Progress)
- Anesthesia: General, stable
- Blood loss at 11:59: 85 mL
- Robotic arms active: 3 (camera, dissector, stapler)
- AI tissue classification: Active, real-time margin assessment
- Estimated remaining time: 45-60 minutes

## PAT-ODMND-0080

- Demographics: 11 years, Female
- Cancer: Pediatric osteosarcoma, left distal femur
- ECOG: 1
- Primary robots: Humanoid (HUMAN-03), Rehabilitation (REHAB-01)
- Arrival: 11:28 (parent-scheduled appointment)
- Location: Check-in (11:28 with parent), Pediatric Waiting (11:29-11:35),
  Humanoid Therapy Room 3 (11:36-11:56), Rehabilitation Bay 1 (11:56-ongoing)
- Procedure: Humanoid-guided mobility assessment followed by exoskeleton
  gait training
- Informed consent: Parental consent (consent ID IC-2026-0609), child assent
  obtained per 21 CFR 50.25
- Digital twin: Musculoskeletal model initialized for gait analysis

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:28 | 88 | 108/66 | 99% | 18 | Check-in baseline |
| 11:33 | 86 | 106/64 | 99% | 17 | Pediatric waiting |
| 11:36 | 90 | 110/68 | 99% | 18 | Humanoid therapy begins |
| 11:40 | 94 | 112/70 | 99% | 19 | Walking assessment |
| 11:44 | 96 | 114/71 | 99% | 20 | Active mobility testing |
| 11:48 | 92 | 110/68 | 99% | 18 | Rest period |
| 11:52 | 90 | 108/66 | 99% | 18 | Assessment concluding |
| 11:56 | 94 | 112/69 | 99% | 19 | Transition to REHAB-01 |
| 11:59 | 96 | 114/70 | 99% | 20 | Exoskeleton gait training |

### Rehabilitation-Specific Metrics
- Humanoid assessment duration: 20 minutes
- Walking pattern: Antalgic gait, left lower extremity guarding
- Joint range of motion: Left knee 5-110 degrees (limited by pain)
- Pain score during assessment: 4/10 (left leg)
- Exoskeleton session: Initial evaluation, 40% body weight offloaded
- Gait speed: 0.4 m/s initial measurement

## PAT-ODMND-0081

- Demographics: 76 years, Male
- Cancer: Liver metastases (colorectal primary), Stage IV
- ECOG: 2
- Primary robot: Steerable Needle (STEER-02)
- Arrival: 11:32 (scheduled interventional slot)
- Location: Check-in (11:32), Waiting (11:33-11:39), Ablation Suite 2
  (11:40-11:58), Recovery Bay 6 (11:58-ongoing)
- Procedure: Steerable needle ablation of liver metastases. ADVERSE EVENT
  at 11:45 (Grade 2 procedural pain, managed with lidocaine bolus).
  Procedure completed successfully.
- Informed consent: Previously completed (consent ID IC-2026-0612),
  including Physical AI steerable needle disclosure per 21 CFR 50.25
- Digital twin: Hepatic tumor model updated with ablation zone data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:32 | 82 | 152/88 | 95% | 18 | Check-in baseline |
| 11:37 | 80 | 150/86 | 95% | 17 | Waiting area |
| 11:40 | 84 | 154/89 | 95% | 18 | Positioned in ablation suite |
| 11:42 | 82 | 152/88 | 95% | 17 | Local anesthetic, sedation |
| 11:44 | 80 | 148/86 | 96% | 16 | Needle advancing |
| 11:45 | 96 | 168/98 | 95% | 22 | PAIN 7/10 - procedure paused |
| 11:46 | 94 | 164/96 | 95% | 20 | Lidocaine bolus administered |
| 11:47 | 90 | 158/92 | 96% | 19 | Pain improving |
| 11:48 | 86 | 152/88 | 96% | 17 | Pain 3/10, patient comfortable |
| 11:50 | 84 | 150/87 | 96% | 17 | Procedure resumed |
| 11:52 | 82 | 148/86 | 96% | 16 | Needle at target |
| 11:54 | 80 | 146/84 | 96% | 16 | Ablation in progress |
| 11:56 | 80 | 146/84 | 96% | 16 | Ablation completing |
| 11:58 | 78 | 144/82 | 96% | 15 | Procedure complete, to recovery |

### Steerable Needle Metrics
- Needle tip accuracy: 1.3 mm from planned target
- Ablation zone: 95% of planned volume covered
- Needle steering corrections: 4 (within nominal range)
- Adverse event: Grade 2 pain at 11:45 (see AE-011-001)
- Resolution: Lidocaine bolus, pain reduced 7/10 to 3/10
- Post-procedure imaging: Ablation zone confirmed on CT

## PAT-ODMND-0082

- Demographics: 48 years, Female
- Cancer: NSCLC squamous cell carcinoma, Stage IIIB
- ECOG: 1
- Primary robot: RT Motion-Tracking (TRACK-03)
- Arrival: 11:36 (patient-chosen daytime slot)
- Location: Check-in (11:36), Waiting (11:37-11:43), Vault 1 (11:44-ongoing)
- Procedure: RT fraction delivery, in progress at hour end
- Informed consent: Previously completed (consent ID IC-2026-0615)
- Digital twin: Lung tumor model active, updating with fraction data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:36 | 76 | 130/78 | 96% | 16 | Check-in baseline |
| 11:40 | 74 | 128/76 | 96% | 15 | Waiting area |
| 11:44 | 78 | 132/80 | 96% | 16 | Positioned on couch |
| 11:46 | 77 | 131/79 | 96% | 16 | Calibration, markers placed |
| 11:48 | 79 | 133/80 | 96% | 17 | Beam-on field 1 |
| 11:52 | 78 | 132/79 | 97% | 16 | Treatment |
| 11:56 | 77 | 131/78 | 97% | 16 | Treatment |
| 11:59 | 76 | 130/78 | 97% | 16 | In progress, 1.6 Gy delivered |

### RT-Specific Metrics (Partial)
- Breathing amplitude: 4.3 mm
- Beam gating efficiency: 94.1% (partial)
- Dose delivered at 11:59: 1.6 Gy of planned 2.0 Gy
- Treatment interruptions: 0

## PAT-ODMND-0083

- Demographics: 36 years, Male
- Cancer: Forearm sarcoma, Stage I, right forearm
- ECOG: 0
- Primary robot: Cobot (COBOT-04)
- Arrival: 11:40 (patient-chosen daytime slot)
- Location: Check-in (11:40), Waiting (11:41-11:47), Biopsy Station 4
  (11:48-ongoing)
- Procedure: Core needle biopsy, in progress at hour end
- Informed consent: Previously completed (consent ID IC-2026-0618)
- Digital twin: Sarcoma model pending biopsy data

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:40 | 68 | 122/74 | 99% | 14 | Check-in baseline |
| 11:44 | 66 | 120/72 | 99% | 13 | Waiting area |
| 11:48 | 70 | 124/75 | 99% | 15 | Positioned, local anesthetic |
| 11:52 | 72 | 126/76 | 99% | 15 | First core obtained |
| 11:56 | 71 | 125/75 | 99% | 15 | Second core obtained |
| 11:59 | 70 | 124/75 | 99% | 14 | In progress |

### Biopsy-Specific Metrics (Partial)
- Force applied: 2.2 N average
- Tissue cores obtained by 11:59: 2 of planned 4

## PAT-ODMND-0084

- Demographics: 72 years, Female
- Cancer: Brain metastases (breast primary), Stage IV
- ECOG: 1
- Primary robot: RT Positioning (RTPOS-01)
- Arrival: 11:44 (scheduled RT slot)
- Location: Check-in (11:44), Waiting (11:45-11:51), Vault 1 (11:52-ongoing)
- Procedure: Stereotactic RT positioning for brain metastases, in progress
- Informed consent: Previously completed (consent ID IC-2026-0621)
- Digital twin: Brain metastases model loaded for treatment planning

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:44 | 74 | 148/88 | 97% | 15 | Check-in baseline |
| 11:48 | 72 | 146/86 | 97% | 14 | Waiting area |
| 11:52 | 76 | 150/89 | 97% | 16 | Mask fitting begins |
| 11:55 | 78 | 152/90 | 97% | 16 | Mask adjustment |
| 11:59 | 76 | 150/88 | 97% | 15 | Mask fitting in progress |

### RT-Positioning Metrics (Partial)
- Mask fitting: In progress at hour end
- Registration: Pending

## PAT-ODMND-0085

- Demographics: 65 years, Male
- Cancer: Femur osteosarcoma, right femur
- ECOG: 2
- Primary robot: Rehabilitation (REHAB-02)
- Arrival: 11:48 (scheduled rehabilitation slot)
- Location: Check-in (11:48), Waiting (11:49-11:55), Rehabilitation Bay 2
  (11:56-ongoing)
- Procedure: Lower extremity exoskeleton-assisted rehabilitation
- Informed consent: Previously completed (consent ID IC-2026-0624)
- Digital twin: Musculoskeletal model initialized for gait analysis

### Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:48 | 80 | 142/84 | 96% | 16 | Check-in baseline |
| 11:52 | 78 | 140/82 | 96% | 15 | Waiting area |
| 11:56 | 82 | 144/85 | 96% | 17 | Exoskeleton fitting |
| 11:59 | 84 | 146/86 | 96% | 17 | Initial gait assessment |

### Rehabilitation-Specific Metrics (Partial)
- Gait pattern: Antalgic, right lower extremity
- Weight support: 45% body weight offloaded
- Session type: Initial evaluation

## PAT-ODMND-0044 (Surgery Completion and Recovery)

- Demographics: Previously documented
- Status this hour: Surgery completed at 11:00 (180 minutes total)
- Outcome: Successful tumor resection, estimated blood loss 210 mL
- Location: Surgical Suite 1 (until 11:00), Recovery Bay 4 (11:05-ongoing)
- Recovery monitoring: Standard post-surgical vital sign protocol

### Recovery Vital Signs

| Time | HR | BP | SpO2 | RR | Notes |
|------|----|----|------|----|-------|
| 11:00 | 72 | 124/76 | 98% | Vent | Surgery complete |
| 11:05 | 74 | 126/78 | 98% | 14 | Extubated, to recovery bay |
| 11:15 | 76 | 128/78 | 97% | 15 | Awake, oriented |
| 11:30 | 74 | 126/76 | 98% | 14 | Stable, pain 4/10 |
| 11:45 | 72 | 124/74 | 98% | 14 | Stable, pain 3/10 |
| 11:59 | 70 | 122/72 | 98% | 13 | Recovery progressing |

## PAT-ODMND-0065 (Ongoing Surgery)

- Demographics: Previously documented
- Status this hour: Surgery ongoing (started approximately 10:40)
- Robot: SURG-01 (Surgical Suite 1)
- Blood loss at 11:59: 165 mL
- Procedure status: Within expected parameters, estimated completion Hour 12
