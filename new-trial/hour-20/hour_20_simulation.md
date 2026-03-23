# Hour 20: 20:00-20:59 - Wind-Down Period Operations

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 20 enters the evening wind-down period with 5 new arrivals
(PAT-ODMND-0162 through PAT-ODMND-0166) as most daytime and evening patients
complete their procedures and discharge. PAT-ODMND-0154 remains in surgical
suite 1 with an ongoing robotic-assisted procedure that began at approximately
19:20. Approximately 12 patients are concurrently on-site at peak, declining
through the hour as evening discharges proceed. Robot utilization drops to
approximately 30% consistent with the wind-down profile. No adverse events
are recorded. Site PSL remains at 65.6 with no scoring changes.

## Site Status at 20:00

- Total patients on-site: ~12 (declining through hour)
- Active procedures: 1 ongoing (P0154 surgery), 5 new procedures initiating
- Robots in active mode: ~9 (declining to ~7 by hour end)
- Robots in standby mode: ~20 (increasing through hour)
- Robots in maintenance: 0
- Queue length: 0 across all stations
- Site safety officer on duty: SSO-E2 (evening shift)

## Regulatory Framework

This trial operates under FDA 21 CFR Part 11 (electronic records and
signatures), 21 CFR Part 820 (quality system regulation for medical devices),
and 21 CFR Part 812 (investigational device exemptions). Robot-assisted
procedures comply with IEC 62304 (medical device software lifecycle) and
ISO 13482 (personal care robots). Radiation therapy systems adhere to
IEC 60601-2-1 (medical electrical equipment for radiation therapy) and
AAPM TG-142 (quality assurance of medical accelerators). Surgical robotics
follow IEC 80601-2-77 (robotically assisted surgical equipment). Imaging
procedures conform to ACR accreditation standards and 21 CFR Part 1020
(radiological health performance standards).

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage | ECOG | Robot Assigned |
|------------|------|-----|-----|-------------|-------|------|----------------|
| PAT-ODMND-0162 | 20:05 | 53 | M | NSCLC squamous | IIIB | 1 | TRACK-02 |
| PAT-ODMND-0163 | 20:15 | 47 | F | Forearm sarcoma | II | 0 | COBOT-03 |
| PAT-ODMND-0164 | 20:25 | 70 | M | Glioblastoma | IV | 2 | RTPOS-02 |
| PAT-ODMND-0165 | 20:35 | 62 | F | Hepatocellular carcinoma | II | 1 | IMAGE-03 |
| PAT-ODMND-0166 | 20:48 | 68 | M | Liver metastases | IV | 2 | IMAGE-04 |

Patient PAT-ODMND-0162 is a 53-year-old male with Stage IIIB NSCLC squamous
cell carcinoma presenting for a radiotherapy session with real-time motion
tracking. He selected the 20:00 window to accommodate work obligations during
standard business hours. ECOG performance status 1. Treatment plan:
Hypofractionated radiotherapy (8 Gy x 5 fractions) targeting a 4.2 cm right
upper lobe mass with mediastinal involvement. TRACK-02 provides respiratory-
gated beam delivery with real-time tumor tracking via implanted fiducial
markers per AAPM TG-76 respiratory management guidelines.

Patient PAT-ODMND-0163 is a 47-year-old female with Grade II forearm sarcoma
presenting for robotic core needle biopsy of a 2.8 cm mass in the right
forearm dorsal compartment. She selected the evening slot due to childcare
responsibilities during daytime hours. ECOG performance status 0. COBOT-03
will perform ultrasound-guided core needle biopsy with AI-assisted soft-tissue
segmentation per 21 CFR 820.30 design controls.

Patient PAT-ODMND-0164 is a 70-year-old male with Stage IV glioblastoma
multiforme presenting for radiotherapy positioning and CT simulation. ECOG
performance status 2. Treatment plan: Standard fractionation (60 Gy in 30
fractions) with concurrent temozolomide per Stupp protocol. RTPOS-02 will
perform 6-DOF couch alignment, thermoplastic mask fitting, and CT simulation
with 1.25 mm slice acquisition for treatment planning. Evening slot selected
to minimize wait times and reduce patient fatigue per ECOG 2 accommodations.

Patient PAT-ODMND-0165 is a 62-year-old female with Stage II hepatocellular
carcinoma (HCC) presenting for diagnostic imaging assessment. Single 3.5 cm
lesion in segment VI. IMAGE-03 will perform contrast-enhanced CT with AI-
assisted volumetric analysis and LI-RADS classification per ACR diagnostic
criteria. Selected the evening slot for caregiver coordination.

Patient PAT-ODMND-0166 is a 68-year-old male with Stage IV colorectal cancer
with liver metastases presenting for imaging assessment. Multiple bilobar
lesions (largest 4.1 cm segment VIII). ECOG performance status 2. IMAGE-04
will perform triphasic CT with AI-assisted lesion segmentation and RECIST 1.1
measurement for treatment response evaluation. Evening slot selected to
accommodate transportation arrangements.

## Active Procedures This Hour

### Ongoing Surgery - PAT-ODMND-0154 (started ~19:20, continuing)
- Patient: PAT-ODMND-0154
- Robot: SURG-01 (Surgical Suite 1)
- Procedure: Robotic-assisted procedure (ongoing from Hour 19)
- Status: Active throughout Hour 20, surgeon and anesthesia team in attendance
- Telemetry: All vital signs within acceptable parameters. Robot functioning
  nominally. No complications reported this hour.
- Estimated completion: ~21:00-21:30

### RT Motion Tracking Session (20:12-20:42)
- Patient: PAT-ODMND-0162
- Robot: TRACK-02 (Radiotherapy Vault 2)
- Vault: Radiotherapy Vault 2
- Procedure: Hypofractionated RT with respiratory-gated beam delivery
- Duration: 30 minutes (setup 5 min, imaging 3 min, treatment 18 min,
  verification 2 min, exit 2 min)
- Fiducial tracking accuracy: 0.6 mm (spec: less than 1.5 mm per AAPM TG-142)
- Respiratory gating efficiency: 94.2% (duty cycle within 30-50% window)
- Beam-on time: 12.4 minutes of 18-minute treatment window
- Dose delivered: 8.00 Gy to PTV (prescription: 8.00 Gy, fraction 3 of 5)
- OAR doses: Spinal cord 0.8 Gy, esophagus 1.2 Gy, heart 0.4 Gy (all within
  constraints)
- Treatment interruptions: 0
- Outcome: Successful fraction delivery. Cumulative dose 24.0 Gy of 40.0 Gy.

Minute-by-minute summary (active procedure):
- 20:12 - Patient positioned supine, arms above head in wing board
- 20:13 - TRACK-02 initiates couch alignment, fiducial detection active
- 20:14 - CBCT acquired for position verification, 3 fiducials identified
- 20:15 - Auto-registration complete, 0.3 mm shift applied (lat/lng/vert)
- 20:16 - Respiratory baseline established, gating window set (30% phase)
- 20:17 - Treatment plan loaded, MLC leaf positions verified
- 20:18 - Beam 1 initiated (6 MV FFF, Arc 1 of 2)
- 20:19 - Arc 1 delivery in progress, fiducial tracking continuous
- 20:20 - Arc 1 delivery, respiratory gating pauses: 2 (normal)
- 20:21 - Arc 1 delivery continues, dose rate 1400 MU/min
- 20:22 - Arc 1 delivery continues
- 20:23 - Arc 1 delivery continues, patient motion within tolerance
- 20:24 - Arc 1 complete, gantry repositioning for Arc 2
- 20:25 - Arc 2 initiated
- 20:26 - Arc 2 delivery in progress
- 20:27 - Arc 2 delivery continues
- 20:28 - Arc 2 delivery continues, respiratory gating nominal
- 20:29 - Arc 2 delivery continues
- 20:30 - Arc 2 delivery continues
- 20:31 - Arc 2 delivery continues
- 20:32 - Arc 2 delivery continues
- 20:33 - Arc 2 delivery complete
- 20:34 - CBCT verification scan initiated
- 20:35 - Verification scan complete, fiducial positions confirmed stable
- 20:36 - Dose verification: 8.00 Gy delivered (100.0% of prescription)
- 20:37 - Treatment complete, beam off, couch retracted
- 20:38 - Patient assisted off couch
- 20:39 - Post-treatment vitals obtained (stable)
- 20:40 - Patient escorted to recovery area
- 20:41 - TRACK-02 cleaning cycle initiated
- 20:42 - Session documented in treatment management system

### Robotic Biopsy Session (20:22-20:40)
- Patient: PAT-ODMND-0163
- Robot: COBOT-03 (Biopsy Station 3)
- Procedure: US-guided core needle biopsy of forearm sarcoma
- Duration: 18 minutes (setup 3 min, localization 3 min, biopsy 8 min,
  hemostasis 3 min, exit 1 min)
- Needle trajectory accuracy: 0.4 mm from planned path (spec: less than 1 mm)
- Core samples obtained: 4 (Grade A quality)
- Needle insertion force: Average 1.9 N (range 1.7-2.2 N)
- Ultrasound visualization: 100% needle tip visibility across all passes
- AI soft-tissue segmentation model v3.2: inference latency 7 ms
- Complications: None
- Outcome: Adequate tissue for histological and molecular analysis. Samples
  sent to pathology. Patient to recovery observation.

### RT Positioning Session (20:32-20:55)
- Patient: PAT-ODMND-0164
- Robot: RTPOS-02 (Radiotherapy Vault 2, after TRACK-02 session completes)
- Note: Vault 2 cleaned and transitioned from TRACK-02 to RTPOS-02 use
- Procedure: GBM RT positioning, mask fitting, CT simulation
- Duration: 23 minutes (setup 3 min, mask molding 7 min, CT sim 9 min,
  verification 2 min, exit 2 min)
- 6-DOF couch positioning accuracy: 0.5 mm deviation (spec: less than 1 mm)
- Mask conformity: 96.8% surface fit
- CT simulation: 1.25 mm slice thickness, full brain coverage with 2 cm
  margin inferiorly to C2 vertebral body
- AI lesion detection: Primary GBM (right temporal, 5.2 cm) and perilesional
  edema correctly delineated
- Treatment interruptions: 0
- Patient tolerance: Adequate (ECOG 2, mild fatigue, completed without breaks)
- Outcome: Successful completion. Mask and CT data transmitted to treatment
  planning system for IMRT dose optimization.

### Imaging Session 1 (20:42-20:55)
- Patient: PAT-ODMND-0165
- Robot: IMAGE-03 (Imaging Bay 3)
- Procedure: Contrast-enhanced CT with AI volumetric analysis for HCC
- Duration: 13 minutes (setup 2 min, scout 1 min, arterial phase 2 min,
  portal venous phase 2 min, delayed phase 2 min, AI analysis 2 min,
  exit 2 min)
- Contrast: Iohexol 100 mL IV, power injector 4 mL/s
- IMAGE-03 positioning accuracy: 1.2 mm (spec: less than 2 mm)
- AI volumetric analysis: Lesion volume 18.4 cm3, LI-RADS 5 (definite HCC)
- Arterial phase hyperenhancement: Present
- Portal venous washout: Present
- Enhancing capsule: Present
- Complications: None
- Outcome: Diagnostic quality images acquired. AI report generated and queued
  for radiologist review per ACR accreditation standards.

### Imaging Session 2 (20:55-21:10, extends into Hour 21)
- Patient: PAT-ODMND-0166
- Robot: IMAGE-04 (Imaging Bay 4)
- Procedure: Triphasic CT with AI lesion segmentation for liver metastases
- Duration: ~15 minutes (setup 2 min, scout 1 min, arterial 2 min, portal
  venous 2 min, delayed 2 min, AI segmentation 4 min, exit 2 min)
- Status at 20:59: Scan in progress (arterial phase acquired, portal venous
  phase in progress)
- Contrast: Iohexol 120 mL IV, power injector 3.5 mL/s
- To be completed in Hour 21.

## Discharges This Hour

Multiple evening patients discharged during this hour as procedures complete
and recovery observation periods conclude. Specific discharge details recorded
in patient records. The wind-down pattern shows controlled reduction from
approximately 12 patients at hour start toward approximately 8 by hour end.

## Safety and Quality Metrics

- Adverse events: 0
- Near-miss events: 0
- Protocol deviations: 0
- Robot error codes: 0
- Emergency stops: 0
- Radiation safety incidents: 0
- Contrast reactions: 0
- Fall events: 0
- Medication errors: 0

## End-of-Hour Status

- Patients on-site at 20:59: ~10 (ongoing reduction)
- Active procedures: 2 (P0154 surgery ongoing, P0166 imaging in progress)
- Completed procedures this hour: 4 (P0162 RT, P0163 biopsy, P0164
  positioning, P0165 imaging)
- Robot utilization: ~30%
- Site PSL: 65.6 (no change)
- Next hour outlook: Continued wind-down, P0154 surgery expected to complete,
  P0166 imaging will complete early in Hour 21
