# Hour 21: 21:00-21:59 - Wind-Down Operations

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 21 represents the late-evening wind-down phase of the 24-hour on-demand
simulation cycle. Four new patients arrive during this hour, bringing the
concurrent on-site count to approximately 8. Two of the new arrivals
(PAT-ODMND-0167 and PAT-ODMND-0168) selected late evening slots due to daytime
work and family obligations, demonstrating the on-demand model's ability to
serve patients outside traditional clinic hours. PAT-ODMND-0154's surgical
procedure completes successfully at 21:10. Night shift handoff preparation
begins at 21:30 in anticipation of the midnight staffing transition. No
adverse events are recorded.

Per ICH E6(R3) Section 4.2 (DOI: 10.5281/zenodo.18973368), all procedures
this hour maintain full audit trail documentation. Per 21 CFR Part 812.150,
device accountability records are updated for all robot instances activated
during this hour.

## Site Status at 21:00

- Total patients on-site: 8 (approximate concurrent occupancy)
- Active procedures: 1 (P0154 surgery in progress, final phase)
- Robots in active mode: 6
- Robots in standby mode: 23
- Robots in maintenance: 0
- Queue length: 0 across all stations
- Site safety officer on duty: SSO-E2 (evening shift)

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage | ECOG | Robot Assigned |
|------------|------|-----|-----|-------------|-------|------|----------------|
| PAT-ODMND-0167 | 21:08 | 45 | M | NSCLC adenocarcinoma | IIB | 1 | TRACK-03 |
| PAT-ODMND-0168 | 21:22 | 59 | F | Parotid tumor | I | 0 | NEEDLE-01 |
| PAT-ODMND-0169 | 21:35 | 14 | M | Pediatric osteosarcoma | - | 1 | HUMAN-01 |
| PAT-ODMND-0170 | 21:48 | 71 | M | Femur osteosarcoma | - | 2 | REHAB-02 |

Patient PAT-ODMND-0167 is a 45-year-old male with Stage IIB non-small cell
lung cancer (adenocarcinoma subtype) presenting for RT motion-tracking
calibration as part of his radiation therapy treatment plan. He selected the
21:08 slot due to daytime employment obligations. ECOG performance status 1.
Per 21 CFR Part 50.25(a)(1), informed consent includes description of the
RT motion-tracking procedure and its role in respiratory-gated beam delivery.

Patient PAT-ODMND-0168 is a 59-year-old female with Stage I parotid gland
tumor presenting for CT-guided needle placement biopsy. She chose the late
evening appointment to accommodate her caregiving responsibilities during
daytime hours. ECOG performance status 0. Per ICH E6(R3) Section 2.10,
the needle-placement system records all trajectory parameters for adverse
event monitoring.

Patient PAT-ODMND-0169 is a 14-year-old male with pediatric osteosarcoma
presenting for humanoid-assisted therapy and psychosocial support. Parental
consent obtained per 21 CFR Part 50.55 (requirements for permission by
parents and assent by children). ECOG performance status 1. The humanoid
therapy session targets pain management education and treatment anxiety
reduction.

Patient PAT-ODMND-0170 is a 71-year-old male with femur osteosarcoma
presenting for exoskeleton-assisted rehabilitation. ECOG performance status 2,
reflecting functional limitations from the femoral tumor. Per 21 CFR
Part 812.62(a), the rehabilitation exoskeleton's assistive parameters are
configured based on his weight-bearing restrictions and pain thresholds.

## Active Procedures This Hour

### P0154 Surgery Completion (20:20-21:10)

- Patient: PAT-ODMND-0154
- Robot: SURG-02 (Surgical Robot, Instance 2)
- Suite: Surgical Suite 2
- Procedure: Tumor resection (procedure started at 20:20, Hour 20)
- Total duration: 110 minutes
- Outcome: Successful, R0 resection (negative margins confirmed)
- Blood loss: 190 mL (within acceptable range)
- Specimen sent to pathology for margin verification

Minute-by-minute summary (Hour 21 portion):
- 21:00 - Final dissection plane completed, hemostasis check in progress
- 21:02 - Vascular pedicle secured, no active bleeding identified
- 21:04 - Specimen removed, placed in pathology container
- 21:06 - Wound irrigation with normal saline, drain placement
- 21:08 - Layer closure initiated, SURG-02 precision suturing mode
- 21:10 - Closure complete, dressing applied, patient extubated
- 21:12 - Patient transferred to Recovery Bay 2, vitals stable

### RT Motion-Tracking Calibration (21:15-21:45)

- Patient: PAT-ODMND-0167
- Robot: TRACK-03 (RT Motion-Tracking, Instance 3)
- Location: Radiotherapy Vault 3
- Procedure: Respiratory-gated motion tracking calibration for NSCLC
- Duration: 30 minutes (setup 5 min, calibration 20 min, verification 5 min)
- Breathing cycle analysis: 14 cycles/min, amplitude 8-12 mm
- Tracking accuracy: 0.4 mm RMS error (specification: less than 1.0 mm)
- Gating window established: 30% duty cycle at end-expiration
- Digital twin sync: Respiratory model initialized with 98.2% correlation
- Outcome: Successful. Tracking model validated for treatment delivery.

Minute-by-minute summary:
- 21:15 - Patient positioned supine on RT couch, external markers placed
- 21:17 - TRACK-03 optical tracking system initialized, 4 markers acquired
- 21:18 - Baseline free-breathing scan acquired (4DCT surrogate)
- 21:20 - Respiratory coaching initiated, amplitude training
- 21:23 - Phase-sorted data acquisition, 10 respiratory bins generated
- 21:27 - Internal-external correlation model fitted, R-squared 0.96
- 21:30 - Beam gating simulation initiated, virtual MLC tracking
- 21:33 - Gating latency measured: 85 ms (specification: less than 200 ms)
- 21:35 - Reproducibility check: 5 consecutive cycles within tolerance
- 21:38 - Digital twin respiratory model uploaded to treatment planning
- 21:40 - Patient comfort check, repositioning verification
- 21:42 - Final verification scan, model correlation confirmed at 98.2%
- 21:45 - Calibration complete, patient escorted from vault

### CT-Guided Needle Placement (21:30-22:05)

- Patient: PAT-ODMND-0168
- Robot: NEEDLE-01 (Needle-Placement System, Instance 1)
- Location: CT Suite 1
- Procedure: CT-guided fine needle aspiration of parotid mass
- Duration: 35 minutes (extends into Hour 22)
- This hour: Setup and initial needle insertion (21:30-21:59)
- CT guidance: Real-time fluoroscopic mode, 0.5 mm slice thickness
- Needle trajectory: Planned to avoid facial nerve branches
- Per 21 CFR Part 812.150(a)(1), needle cartridge serial number logged

Minute-by-minute summary (this hour portion):
- 21:30 - Patient positioned, CT scout acquired, tumor localized
- 21:32 - Skin entry point marked, local anesthesia administered
- 21:35 - NEEDLE-01 trajectory planning complete, facial nerve mapped
- 21:37 - First pass initiated, real-time CT guidance active
- 21:40 - Needle tip confirmed within target lesion, 22 mm depth
- 21:43 - First aspirate collected, syringe labeled and set aside
- 21:46 - Needle repositioned for second pass
- 21:49 - Second aspirate collected, adequate cellularity confirmed on rapid stain
- 21:52 - Third pass initiated for additional tissue sampling
- 21:55 - Third aspirate collected, samples sent to cytology
- 21:58 - Hemostasis check, no active bleeding at puncture site

### Humanoid Therapy Session (21:40-22:10)

- Patient: PAT-ODMND-0169
- Robot: HUMAN-01 (Humanoid, Instance 1)
- Location: Therapy Room 1
- Procedure: Psychosocial therapy and pain management education
- Duration: 30 minutes (extends into Hour 22)
- This hour: Initial engagement and therapy activities (21:40-21:59)
- Session goals: Anxiety reduction, treatment explanation, coping strategies
- Parental observer: Father present in observation area

Minute-by-minute summary (this hour portion):
- 21:40 - HUMAN-01 greets patient, age-appropriate introduction
- 21:42 - Anxiety assessment: patient reports 6/10 treatment anxiety
- 21:44 - Interactive disease education using holographic bone model
- 21:47 - Patient asks questions about surgery, HUMAN-01 provides responses
- 21:50 - Guided relaxation exercise initiated (breathing technique)
- 21:53 - Post-exercise anxiety reassessment: 4/10 (improvement noted)
- 21:55 - Pain management discussion: pharmacologic and non-pharmacologic
- 21:58 - Interactive distraction game initiated (continues into Hour 22)

### Rehabilitation Session (21:55-22:25)

- Patient: PAT-ODMND-0170
- Robot: REHAB-02 (Rehabilitation Exoskeleton, Instance 2)
- Location: Rehabilitation Bay 2
- Procedure: Lower extremity assisted ambulation, femur protection protocol
- Duration: 30 minutes (extends into Hour 22)
- This hour: Setup and initial calibration (21:55-21:59)
- Weight-bearing restriction: 50% on affected limb
- Per 21 CFR Part 890.3480, the exoskeleton is configured as a powered
  exercise device with safety force limits

Minute-by-minute summary (this hour portion):
- 21:55 - Patient seated, REHAB-02 lower extremity attachment initiated
- 21:57 - Joint angle sensors calibrated, range of motion baseline recorded
- 21:59 - Gait pattern selection: assisted partial weight-bearing mode

## Night Shift Handoff Preparation

Beginning at 21:30, the evening shift safety officer (SSO-E2) initiates
handoff documentation for the overnight transition. Per ICH E6(R3)
Section 4.2.5, all active patient statuses, pending procedures, and robot
operational states are documented in the shift handoff report. The handoff
includes:

- Active patient census and procedure statuses
- Robot maintenance schedule for overnight hours
- Pending laboratory results and pathology reports
- Any unresolved clinical queries from the evening shift
- Emergency contact protocols for on-call physician coverage

## Patient Departures This Hour

| Patient ID | Time | Outcome | Notes |
|------------|------|---------|-------|
| PAT-ODMND-0154 | 21:10 | Surgery complete, to recovery | R0 resection, blood loss 190 mL, Recovery Bay 2 |

## Late Evening Scheduling Note

Patients PAT-ODMND-0167 and PAT-ODMND-0168 selected late evening appointment
slots due to daytime work and caregiving obligations, respectively. The
on-demand scheduling model per 21 CFR Part 50.25(a)(2) ensures that patients
are informed that participation is voluntary and that appointment flexibility
does not compromise procedural quality. The availability of full robotic
capability during evening hours supports equitable access to clinical trial
participation regardless of patients' daytime schedules.

## Regulatory Compliance Notes

- All procedures performed under IRB-approved protocol (IRB-2026-ODMND-001)
- Per ICH E6(R3) Section 1.1, investigational device accountability maintained
  for all robot instances activated this hour
- Per 21 CFR Part 11.10(e), electronic records for all procedures include
  timestamps, operator identification, and device serial numbers
- Per 21 CFR Part 812.140(a)(3), records of device usage maintained for
  TRACK-03, NEEDLE-01, HUMAN-01, and REHAB-02
- Pediatric consent for PAT-ODMND-0169 documented per 21 CFR Part 50.55
  with parental permission and minor assent
- No adverse events reported per ICH E6(R3) Section 2.10
