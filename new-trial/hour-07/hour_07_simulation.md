# Hour 07: 07:00-07:59 - Morning Ramp-Up and First Surgery

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 07 marks the transition to full daytime operations with the morning
ramp-up period. Eight new patients arrive (PAT-ODMND-0024 through
PAT-ODMND-0031), the day shift safety officer (SSO-D1) replaces the night
shift officer (SSO-N1), and the site's first surgery of the day initiates
at 07:40 - a 90-minute mediastinal tumor resection for PAT-ODMND-0024.
One patient (PAT-ODMND-0003) is discharged after overnight post-surgical
recovery. An ongoing ablation procedure for PAT-ODMND-0022 (started in
Hour 06) completes at 07:25. One Grade 1 adverse event is recorded:
PAT-ODMND-0029 experiences nausea during imaging prep, managed per IND
protocol with ondansetron. Robot utilization rises to approximately 35%
as multiple procedures run concurrently. The cumulative site PSL score
increases to 64.1 with a +0.1 increment to Surgical Dim C following the
first surgical procedure initiation.

Regulatory framework: ICH E6(R3) Section 4 (investigator responsibilities),
21 CFR Part 50 (informed consent), 21 CFR Part 312 (IND requirements).
USL benchmarking per Kawchak, 2026 (DOI: 10.5281/zenodo.18778220).
Patient journey mapping per DOI: 10.5281/zenodo.19119939.

## Site Status at 07:00

- Total patients on-site at start: 7 (carried over from Hour 06)
- Day shift change: SSO-D1 assumes duty from SSO-N1
- Shift handoff briefing: 07:00-07:05 (5-minute overlap per ICH E6(R3))
- Active procedures at start: 1 (P0022 ablation, continuing from Hour 06)
- Robots in active mode: 2 (STEER-01 for P0022, COMPN-03 for P0005)
- Robots in standby mode: 27
- Queue length: 0 (first morning arrivals pending)
- Site safety officer on duty: SSO-D1 (day shift, effective 07:00)

## Shift Handoff (07:00-07:05)

SSO-N1 provides verbal and electronic handoff to SSO-D1 per protocol:
- 7 patients on-site: P0003 (post-surgical recovery, stable), P0005
  (pediatric ALL, overnight companion monitoring), P0022 (ablation in
  progress, STEER-01, expected completion 07:25), and 4 others in
  various recovery or monitoring states
- No active safety concerns from overnight period
- COBOT-03 preventive calibration completed at 04:00 (passed all checks)
- Facility environmental systems nominal
- SSO-D1 acknowledges receipt at 07:05. SSO-N1 departs.

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage/Grade | ECOG | Robot Assignment |
|-----------|------|-----|-----|-------------|-------------|------|-----------------|
| PAT-ODMND-0024 | 07:05 | 59 | M | Mediastinal tumor | Stage II | 1 | SURG-01 |
| PAT-ODMND-0025 | 07:08 | 14 | F | Pediatric ALL | - | 1 | COMPN-02 |
| PAT-ODMND-0026 | 07:15 | 66 | M | Glioblastoma | Stage IV | 1 | RTPOS-03 |
| PAT-ODMND-0027 | 07:20 | 41 | F | Forearm sarcoma | Grade III | 0 | COBOT-04 |
| PAT-ODMND-0028 | 07:28 | 53 | M | NSCLC adenocarcinoma | Stage IIIA | 1 | TRACK-01 |
| PAT-ODMND-0029 | 07:35 | 77 | F | Liver mets colorectal | Stage IV | 2 | IMAGE-02, STEER-02 |
| PAT-ODMND-0030 | 07:42 | 9 | M | Pediatric osteosarcoma | - | 1 | HUMAN-02 |
| PAT-ODMND-0031 | 07:50 | 62 | F | Parotid tumor | Stage I | 0 | NEEDLE-02 |

## Continuing Patients

| Patient ID | Status at 07:00 | Key Events This Hour |
|-----------|----------------|---------------------|
| PAT-ODMND-0003 | Post-surgical recovery | Discharged at 07:15 |
| PAT-ODMND-0005 | Overnight companion monitoring | Companion session 07:00 with COMPN-03, pre-chemo |
| PAT-ODMND-0022 | Ablation in progress (STEER-01) | Procedure completes 07:25 |

## Active Procedures This Hour

### P0005 Companion Session (07:00-07:30)
- Patient: PAT-ODMND-0005 (8M, Pediatric ALL)
- Robot: COMPN-03 (Social Companion, Instance 3)
- Location: Pediatric Play Room 3
- Context: Morning companion session before scheduled chemotherapy. COMPN-03
  transitions from passive overnight monitoring to active companion mode.
  Session includes guided breathing exercises, therapy card games, and
  treatment preparation discussion using age-appropriate language.
- Investigational drug context: Patient scheduled for vincristine/prednisone
  (standard ALL protocol) following this session per 21 CFR Part 312 IND
  protocol requirements. Companion session aims to reduce pre-treatment
  anxiety.
- Outcome: Session completed. Patient reports reduced anxiety (self-reported
  4/10 down from 7/10). Transition to chemotherapy infusion area at 07:35.

### P0022 Ablation Completion (07:00-07:25)
- Patient: PAT-ODMND-0022 (continuing from Hour 06)
- Robot: STEER-01 (Steerable Needle, Instance 1)
- Location: Ablation Suite 1
- Procedure: Liver ablation, final 25 minutes of procedure started in Hour 06
- Key events:
  - 07:00-07:15: Active ablation zone monitoring, real-time temperature
    feedback from steerable needle tip thermocouples
  - 07:15-07:20: Ablation energy delivery complete. Needle withdrawal sequence
  - 07:20-07:25: Post-ablation imaging confirmation. Ablation zone 3.2 cm
    (target 3.0 cm, within acceptable margin)
- Outcome: Successful completion. Patient transferred to recovery.

### P0003 Discharge (07:15)
- Patient: PAT-ODMND-0003 (61M, Mediastinal tumor)
- Post-surgical recovery complete after overnight stay (admitted 22:30
  prior day)
- Discharge criteria met: Vitals stable for 4 consecutive hours, pain
  controlled (VAS 2/10), ambulating independently, drain output < 50 mL/hr
- Discharge instructions provided per ICH E6(R3) documentation requirements
- Follow-up appointment: 5 days

### P0024 Pre-Op and Surgery Initiation (07:05-end of hour)
- Patient: PAT-ODMND-0024 (59M, Mediastinal tumor, Stage II, ECOG 1)
- Robot: SURG-01 (Surgical Robot, Instance 1, Surgical Suite 1)
- Pre-operative sequence:
  - 07:05: Arrival, check-in, identity verification per 21 CFR Part 50
  - 07:08: Informed consent confirmed (consent ID IC-2026-0487), including
    Physical AI robotic surgery disclosure and IND protocol for pre-op
    antibiotics per 21 CFR Part 312
  - 07:10-07:20: Pre-op assessment. Vitals, labs reviewed, NPO verified.
    Pre-op antibiotics administered (cefazolin 2g IV per IND protocol).
  - 07:20-07:35: Anesthesia preparation. General anesthesia induced.
    Endotracheal intubation. Double-lumen tube for single-lung ventilation.
  - 07:35-07:40: Sterile draping. SURG-01 positioned. Three chest port sites
    marked (right lateral approach).
  - 07:40: Surgery begins. SURG-01 activates for 3-port mediastinal tumor
    resection.
- Surgical plan: 90-minute estimated duration. Three chest ports (camera port
  + two instrument ports). AI-assisted tumor margin identification. Real-time
  force feedback monitoring.
- Minute-by-minute (surgery start through end of hour):
  - 07:40 - First port placed (12 mm camera port, right 5th intercostal)
  - 07:42 - Second port placed (8 mm instrument port, right 3rd intercostal)
  - 07:44 - Third port placed (8 mm instrument port, right 7th intercostal)
  - 07:45 - Camera inserted. Mediastinal anatomy visualized. AI tumor margin
    overlay activated.
  - 07:48 - Dissection begins. Peritumoral tissue identification.
  - 07:50 - Tumor capsule identified. Force feedback: 2.1 N (safe range).
  - 07:52 - Superior pole dissection. Vessel identification via AI overlay.
  - 07:55 - Hemostasis maintained. Blood loss < 25 mL.
  - 07:58 - Dissection progressing along posterior margin. Recurrent laryngeal
    nerve identified and preserved (neural mapping active).
  - 07:59 - Surgery continues into Hour 08. Estimated 70 minutes remaining.
- Status at end of hour: Surgery 33% complete. Patient hemodynamically stable
  (MAP 72, SpO2 99% on single-lung ventilation, EtCO2 38).

### P0025 Companion Session (07:08-07:45)
- Patient: PAT-ODMND-0025 (14F, Pediatric ALL, ECOG 1)
- Robot: COMPN-02 (Social Companion, Instance 2)
- Location: Pediatric Play Room 2
- Context: Adolescent companion session. COMPN-02 configured for teen
  interaction mode with age-appropriate conversation, music selection, and
  treatment journey education.
- Informed consent: Parental consent and minor assent documented per
  21 CFR Part 50, Subpart D (additional protections for children).
- Outcome: Session ongoing at end of hour. Patient engaged and calm.

### P0026 RT Positioning (07:15-07:55)
- Patient: PAT-ODMND-0026 (66M, Glioblastoma, Stage IV, ECOG 1)
- Robot: RTPOS-03 (RT Positioning, Instance 3)
- Location: Radiotherapy Vault 3
- Procedure: CT simulation and thermoplastic mask fitting for stereotactic
  radiosurgery planning
- Key events:
  - 07:15: Arrival and check-in
  - 07:20: Waiting area (5 min)
  - 07:25: Vault 3 entry. RTPOS-03 activates.
  - 07:28: Thermoplastic mask molding. Patient positioned supine.
  - 07:32: Mask hardened. Immobilization verified (< 1 mm shift).
  - 07:35: CT simulation scan acquired (1 mm slices, contrast-enhanced)
  - 07:40: Isocenter marked via laser alignment system
  - 07:45: Verification imaging. Position confirmed within 0.5 mm tolerance.
  - 07:50: Setup complete. Patient released from mask. Post-positioning check.
  - 07:55: Patient exits vault. Procedure complete.
- Outcome: Successful positioning and CT simulation. Treatment plan to be
  generated for subsequent fraction delivery.

### P0027 Cobot Biopsy (07:20-07:50)
- Patient: PAT-ODMND-0027 (41F, Forearm sarcoma, Grade III, ECOG 0)
- Robot: COBOT-04 (Cobot, Instance 4, Biopsy Station 4)
- Procedure: Ultrasound-guided core needle biopsy of right forearm mass
- Key events:
  - 07:20: Arrival and check-in
  - 07:25: Informed consent confirmed (IC-2026-0491), Physical AI disclosure
  - 07:28: Local anesthesia administered (1% lidocaine, 5 mL)
  - 07:30: COBOT-04 activates. Ultrasound probe positioned by cobot arm.
  - 07:32: Lesion localized on ultrasound (3.8 x 2.5 cm hyperechoic mass)
  - 07:34: First core biopsy pass. Sample collected (18-gauge needle).
  - 07:36: Second core biopsy pass. Adequate tissue confirmed.
  - 07:38: Third core biopsy pass. Three cores total obtained.
  - 07:40: Needle removed. Hemostasis achieved with pressure.
  - 07:45: Post-biopsy dressing applied. No immediate complications.
  - 07:50: Patient moved to observation area. 30-minute monitoring.
- Outcome: Three adequate core samples obtained. Sent to pathology.

### P0028 RT Motion-Tracking (07:28-07:58)
- Patient: PAT-ODMND-0028 (53M, NSCLC adenocarcinoma, Stage IIIA, ECOG 1)
- Robot: TRACK-01 (RT Motion-Tracking, Instance 1, Vault 2)
- Procedure: RT fraction delivery with respiratory gating
- Key events:
  - 07:28: Arrival and check-in
  - 07:32: Waiting area (4 min)
  - 07:36: Vault 2 entry. TRACK-01 activates.
  - 07:38: Patient positioned. Marker block placed. Breathing baseline: 3.8 mm
  - 07:40: Calibration complete. Beam-on, first field (1.0 Gy).
  - 07:45: Field 1 complete. Gantry rotation.
  - 07:46: Beam-on, second field (0.6 Gy).
  - 07:50: Field 2 complete. Gantry rotation.
  - 07:51: Beam-on, third field (0.4 Gy).
  - 07:54: Field 3 complete. Total: 2.0 Gy delivered.
  - 07:55: Marker block removed. Patient assisted to seated position.
  - 07:58: Patient exits vault. Procedure complete.
- Beam gating efficiency: 95.1%
- Marker displacement: 1.6 mm average (within tolerance)
- Treatment interruptions: 0
- Outcome: Successful fraction delivery. Full dose delivered.

### P0029 Imaging and Steerable Needle Prep (07:35-end of hour)
- Patient: PAT-ODMND-0029 (77F, Liver mets colorectal, Stage IV, ECOG 2)
- Robots: IMAGE-02 (Imaging, Instance 2), then STEER-02 (Steerable Needle,
  Instance 2)
- Procedure: Pre-procedural imaging followed by steerable needle intervention
- Key events:
  - 07:35: Arrival and check-in
  - 07:38: Informed consent confirmed (IC-2026-0494)
  - 07:40: Imaging Bay 2 entry. IMAGE-02 activates.
  - 07:42: Robotic ultrasound scan initiated for lesion mapping
  - 07:45: ADVERSE EVENT - Patient reports nausea (Grade 1 per CTCAE v5.0)
    during imaging prep. Imaging paused.
  - 07:46: Attending physician notified. Ondansetron 4 mg IV administered per
    IND protocol (21 CFR Part 312). Event documented in adverse event log.
  - 07:50: Nausea resolved. Patient reports feeling better.
  - 07:52: Imaging resumed. Lesion mapped: 2.1 cm metastatic deposit in
    segment VI.
  - 07:55: Imaging complete. Patient transferred to Ablation Suite 2.
  - 07:58: STEER-02 prep initiated. Steerable needle procedure to begin
    in Hour 08.
- Adverse event documentation: AE-2026-0029-001, Grade 1 nausea, related to
  procedure (imaging prep positioning in elderly patient with ECOG 2).
  Managed with ondansetron per IND protocol. No dose modifications required.
  Reported per ICH E6(R3) Section 6.

### P0030 Humanoid Therapy Session (07:42-end of hour)
- Patient: PAT-ODMND-0030 (9M, Pediatric osteosarcoma, ECOG 1)
- Robot: HUMAN-02 (Humanoid, Instance 2, Therapy Room 2)
- Procedure: Physical therapy and mobility assessment with humanoid assistance
- Key events:
  - 07:42: Arrival with parent. Check-in.
  - 07:45: Informed consent (parental) and minor assent per 21 CFR Part 50,
    Subpart D
  - 07:48: Therapy Room 2 entry. HUMAN-02 activates.
  - 07:50: HUMAN-02 demonstrates range-of-motion exercises. Child mirrors
    movements.
  - 07:52: Gait assessment. HUMAN-02 walks alongside patient, monitoring
    stride symmetry and balance.
  - 07:55: Strength assessment via gamified resistance exercises.
  - 07:58: Session ongoing. Estimated completion in Hour 08.
- Outcome at end of hour: Session in progress. Patient engaged and motivated.

### P0031 Needle Placement Biopsy (07:50-end of hour)
- Patient: PAT-ODMND-0031 (62F, Parotid tumor, Stage I, ECOG 0)
- Robot: NEEDLE-02 (Needle-Placement, Instance 2, CT Suite 2)
- Procedure: CT-guided fine needle aspiration of parotid mass
- Key events:
  - 07:50: Arrival and check-in
  - 07:52: Informed consent confirmed (IC-2026-0497)
  - 07:55: CT Suite 2 entry. NEEDLE-02 activates.
  - 07:56: Planning CT scan acquired. Target lesion identified (1.4 cm
    superficial lobe mass)
  - 07:58: Needle trajectory calculated by AI. Entry point marked.
  - 07:59: Procedure continues into Hour 08.
- Outcome at end of hour: In progress. Expected completion early Hour 08.

## Investigational Drug Administration

| Time | Patient | Drug | Dose | Route | Protocol Ref |
|------|---------|------|------|-------|-------------|
| 07:10 | PAT-ODMND-0024 | Cefazolin | 2 g | IV | Pre-op antibiotics per IND |
| 07:46 | PAT-ODMND-0029 | Ondansetron | 4 mg | IV | AE management per IND |
| 07:35 (post-session) | PAT-ODMND-0005 | Vincristine/Prednisone | Per protocol | IV/PO | Standard ALL protocol |

All drug administrations documented per 21 CFR Part 312 requirements with
appropriate source documentation and case report form entries.

## Adverse Events

| AE ID | Patient | Time | Event | Grade | Action | Outcome |
|-------|---------|------|-------|-------|--------|---------|
| AE-2026-0029-001 | PAT-ODMND-0029 | 07:45 | Nausea during imaging prep | 1 | Ondansetron 4 mg IV | Resolved 07:50 |

Classification: Grade 1 per CTCAE v5.0. Possibly related to positioning
during imaging in elderly patient (77F) with ECOG 2 and Stage IV disease.
No procedure modification required. Imaging resumed after resolution.
Documented per ICH E6(R3) adverse event reporting requirements.

## End of Hour Status

- Total patients on-site: 14
- Patients arrived this hour: 8
- Patients discharged this hour: 1 (PAT-ODMND-0003)
- Active procedures at 07:59: 5 (P0024 surgery, P0025 companion, P0029
  steerable needle prep, P0030 humanoid therapy, P0031 needle biopsy)
- Completed procedures this hour: 5 (P0005 companion, P0022 ablation,
  P0026 RT positioning, P0027 cobot biopsy, P0028 RT motion-tracking)
- Adverse events this hour: 1 (Grade 1, resolved)
- Robot utilization: approximately 35%
- Site PSL: 64.1 (Surgical Dim C +0.1 for first surgery initiation)
- Safety officer: SSO-D1 (day shift)
