# Hour 16: 16:00-16:59 - Evening Peak Begins

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 16 marks the beginning of the evening peak period as working patients
arrive for after-hours treatment slots. Ten new patients check in during this
hour, the highest single-hour arrival count of the simulation thus far. The
on-demand scheduling model demonstrates a core advantage: patients P0124,
P0125, and P0132 selected post-work appointment windows that would be
unavailable at traditional 08:00-17:00 oncology sites. An after-school
pediatric wave also begins with patients P0127 and P0130 arriving for
treatment and companion sessions respectively. Patient P0116 remains in
surgical suite from an ongoing procedure that started at approximately 15:20.

## Site Status at 16:00

- Total patients on-site: approximately 20 (concurrent)
- Active procedures: 8 (carried over from prior hours plus new starts)
- Robots in active mode: 16 (of 29 instances)
- Robots in standby mode: 13
- Robots in maintenance: 0
- Queue length: 0-1 across stations (brief queues at check-in only)
- Site safety officer on duty: SSO-D2 (day shift 2)

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage | ECOG | Robot Assignment |
|-----------|------|-----|-----|-------------|-------|------|-----------------|
| PAT-ODMND-0124 | 16:02 | 50 | M | NSCLC adenocarcinoma | IIIA | 1 | TRACK-01 |
| PAT-ODMND-0125 | 16:08 | 41 | F | Forearm sarcoma | II | 0 | COBOT-02 |
| PAT-ODMND-0126 | 16:12 | 73 | M | Glioblastoma | IV | 2 | RTPOS-01 |
| PAT-ODMND-0127 | 16:18 | 15 | M | Pediatric osteosarcoma | - | 1 | HUMAN-01, REHAB-01 |
| PAT-ODMND-0128 | 16:24 | 58 | F | Parotid tumor | II | 0 | NEEDLE-02 |
| PAT-ODMND-0129 | 16:30 | 64 | M | HCC | III | 1 | IMAGE-03 |
| PAT-ODMND-0130 | 16:36 | 8 | F | Pediatric ALL | - | 1 | COMPN-04 |
| PAT-ODMND-0131 | 16:42 | 69 | F | Liver metastases | IV | 2 | IMAGE-01, STEER-01 |
| PAT-ODMND-0132 | 16:48 | 46 | M | SCLC | III | 1 | TRACK-02 |
| PAT-ODMND-0133 | 16:54 | 55 | F | Forearm sarcoma | I | 0 | COBOT-03 |

## On-Demand Advantage Demonstration

Three patients this hour exemplify the on-demand model's benefit for working
adults who cannot attend traditional daytime-only clinic schedules:

- PAT-ODMND-0124 (50M, NSCLC): Software engineer. Selected 16:02 slot via
  patient portal after completing work shift at 15:30. At a traditional site,
  his RT fraction would require taking half a day off work each session.
- PAT-ODMND-0125 (41F, forearm sarcoma): Middle school teacher. School ends
  at 15:30; arrived at 16:08 without disrupting her work day. Traditional
  biopsy scheduling would have required a substitute teacher.
- PAT-ODMND-0132 (46M, SCLC): Construction supervisor. Shift ends at 16:00;
  arrived at 16:48 for RT tracking. Traditional sites would be closing intake.

Per 21 CFR Part 312, Section 312.60, investigators must ensure proper conduct
regardless of the hour of treatment delivery. The on-demand model extends
access without compromising protocol adherence.

## After-School Pediatric Wave

Two pediatric patients arrive during the after-school window:

- PAT-ODMND-0127 (15M, osteosarcoma): High school student. Arrived at 16:18
  after school day ends. Dual-robot session with humanoid interaction
  (HUMAN-01) followed by rehabilitation exoskeleton (REHAB-01) for
  post-surgical gait training. Per 21 CFR Part 50, Subpart D, additional
  pediatric protections applied with parental assent verified.
- PAT-ODMND-0130 (8F, ALL): Elementary school student. Arrived at 16:36 for
  companion robot session (COMPN-04). After-school scheduling preserves
  normal school attendance during maintenance chemotherapy phase. Parent
  present throughout per institutional pediatric policy.

## Ongoing Procedure from Prior Hour

### P0116 Surgery (Started approximately 15:20)
- Patient: PAT-ODMND-0116
- Robot: SURG-02 (Surgical Suite 2)
- Procedure: Ongoing robotic-assisted surgery initiated during Hour 15
- Status at 16:00: Procedure in progress, estimated completion 16:45
- Surgeon oversight: Lead surgeon and assistant surgeon present
- Robot telemetry: All parameters nominal, force feedback within limits

## Active Procedures This Hour

### RT Motion-Tracking Session (16:10-16:28)
- Patient: PAT-ODMND-0124
- Robot: TRACK-01 (RT Motion-Tracking, Instance 1)
- Vault: Radiotherapy Vault 1
- Procedure: RT fraction delivery, 2 Gy to right upper lobe lesion
- Duration: 18 minutes (calibration 2 min, treatment 14 min, exit 2 min)
- Beam gating efficiency: 95.1%
- Breathing amplitude: 3.8 mm (within tolerance)
- Marker displacement: 1.6 mm average
- Treatment interruptions: 0
- Outcome: Successful completion. Full dose delivered.

### Cobot Biopsy Session (16:16-16:32)
- Patient: PAT-ODMND-0125
- Robot: COBOT-02 (Cobot, Instance 2)
- Station: Biopsy Station 2
- Procedure: Core needle biopsy, left forearm soft-tissue mass
- Duration: 16 minutes (positioning 3 min, biopsy 10 min, hemostasis 3 min)
- Needle insertions: 3 cores obtained
- Force applied: 4.2 N average insertion force
- Tissue sample quality: Grade A
- Outcome: Successful. Three cores adequate for histopathology.

### RT Positioning Session (16:20-16:52)
- Patient: PAT-ODMND-0126
- Robot: RTPOS-01 (RT Positioning, Instance 1)
- Vault: Radiotherapy Vault 1 (after P0124 vacates)
- Procedure: Cranial RT positioning and treatment for glioblastoma
- Investigational drug: Temozolomide per IND (concurrent chemoradiation)
- Duration: 32 minutes (mask fitting 5 min, CBCT 4 min, positioning 3 min,
  treatment 18 min, exit 2 min)
- Thermoplastic mask fit score: 9.1/10
- 6-DOF couch alignment: 0.3 mm translational, 0.2 degrees rotational
- CBCT-to-plan registration: 98.4% agreement
- Dose delivered: 2.0 Gy (fraction of planned 60 Gy total)
- Outcome: Successful. Position maintained within 1 mm throughout.

### Humanoid Interaction and Rehab Session (16:26-16:56)
- Patient: PAT-ODMND-0127
- Robots: HUMAN-01 (16:26-16:40), then REHAB-01 (16:42-16:56)
- Location: Pediatric Therapy Room 1, then Rehabilitation Bay 1
- Phase 1 - Humanoid (14 min): Pre-rehab anxiety management, gait
  assessment discussion, exercise demonstration. Anxiety score reduced
  from 5/10 to 2/10.
- Phase 2 - Rehab exoskeleton (14 min): Post-surgical gait training for
  left knee osteosarcoma resection (limb-sparing surgery, 4 weeks prior).
  Steps completed: 186. Gait symmetry index: 0.82 (baseline 0.74 at
  last session). Load-bearing tolerance: 65% body weight on affected side.
- Outcome: Gait symmetry improved. Patient engaged and motivated.

### Needle-Placement Session (16:32-16:48)
- Patient: PAT-ODMND-0128
- Robot: NEEDLE-02 (Needle-Placement, Instance 2)
- Suite: CT Suite 2
- Procedure: CT-guided fine needle aspiration, right parotid mass
- Duration: 16 minutes (CT planning 4 min, needle placement 8 min,
  sample acquisition 2 min, hemostasis 2 min)
- Needle trajectory accuracy: 1.1 mm from planned path
- Samples obtained: 4 passes (cytology)
- CT dose: 2.8 mGy (CTDI vol)
- Facial nerve proximity: 4.2 mm (safe margin maintained)
- Outcome: Successful. Samples adequate for cytopathology.

### Imaging Assessment (16:38-16:52)
- Patient: PAT-ODMND-0129
- Robot: IMAGE-03 (Imaging Assistant, Instance 3)
- Bay: Imaging Bay 3
- Procedure: Robotic ultrasound liver assessment for HCC staging
- Duration: 14 minutes
- Probe pressure: 1.9 N steady
- Image quality score: 8.4/10
- Primary tumor measurement: 42 mm x 36 mm
- Portal vein assessment: No invasion detected
- Scan coverage: 94%
- Motion artifacts: 1 (auto-compensated)
- Outcome: Successful. Staging images uploaded to DICOM server.

### Companion Robot Session (16:44-16:59)
- Patient: PAT-ODMND-0130
- Robot: COMPN-04 (Social Companion, Instance 4)
- Location: Pediatric Play Room 4
- Procedure: After-school companion session for anxiety management during
  maintenance chemotherapy phase
- Duration: 15 minutes (ongoing into Hour 17)
- Activities: Homework assistance with educational games, guided breathing
  exercises, treatment schedule review with age-appropriate language
- Anxiety score: Entry 4/10, at 16:59 reduced to 1/10
- Parent present: Mother in adjacent observation area
- Outcome: In progress. Patient relaxed and engaged.

### Imaging and Steerable Needle Session (16:50-continuing)
- Patient: PAT-ODMND-0131
- Robots: IMAGE-01 (16:50-16:58), then STEER-01 (16:58-continuing)
- Location: Imaging Bay 1, then Ablation Suite 1
- Investigational drug: Cabozantinib per IND (HCC with liver metastases)
- Phase 1 - Imaging (8 min): Pre-procedural liver mapping with robotic
  ultrasound. Three metastatic lesions identified (18 mm, 14 mm, 9 mm).
  Image quality: 8.0/10.
- Phase 2 - Steerable needle (starting 16:58): CT-guided steerable needle
  biopsy of largest metastatic lesion. Procedure initiated, continuing
  into Hour 17.
- Outcome: Imaging complete. Needle procedure in progress.

### RT Motion-Tracking Session (16:56-continuing)
- Patient: PAT-ODMND-0132
- Robot: TRACK-02 (RT Motion-Tracking, Instance 2)
- Vault: Radiotherapy Vault 2
- Procedure: RT fraction delivery for SCLC, initiated end of hour
- Duration: Started 16:56, continuing into Hour 17
- Outcome: In progress.

### Cobot Session (16:58-continuing)
- Patient: PAT-ODMND-0133
- Robot: COBOT-03 (Cobot, Instance 3)
- Station: Biopsy Station 3
- Procedure: Core needle biopsy, right forearm soft-tissue mass
- Duration: Started 16:58, continuing into Hour 17
- Outcome: In progress.

## Patient Departures This Hour

| Patient ID | Time | Outcome | Notes |
|-----------|------|---------|-------|
| PAT-ODMND-0124 | 16:35 | Discharged | RT fraction complete, no complications |
| PAT-ODMND-0125 | 16:45 | Discharged | Biopsy complete, hemostasis confirmed |
| PAT-ODMND-0128 | 16:55 | Discharged | FNA complete, no facial nerve symptoms |

## Adverse Events

None this hour.

## Investigational Drug Administrations

### PAT-ODMND-0126 - Temozolomide (IND)
- Drug: Temozolomide 75 mg/m2 oral, daily concurrent with cranial RT
- IND protocol: Concurrent chemoradiation for newly diagnosed glioblastoma
- Administration: Patient self-administered oral dose at 15:30 (1 hour
  prior to RT positioning session) per protocol
- Monitoring: No nausea, no vomiting. CBC within acceptable limits
  (ANC 2,100/uL, platelets 142,000/uL). Hepatic function normal.
- Per 21 CFR Part 312, Section 312.32: No reportable adverse events.
  IND safety monitoring documentation current.
- Per ICH E6(R3), Section 5.4.1: Investigational product accountability
  log updated. Dispensing records maintained by site pharmacist.

### PAT-ODMND-0131 - Cabozantinib (IND)
- Drug: Cabozantinib 60 mg oral, daily for HCC with liver metastases
- IND protocol: Systemic therapy for advanced HCC with concurrent
  local ablation assessment
- Administration: Patient took daily dose at 08:00 this morning per protocol
- Monitoring: Blood pressure 148/88 mmHg (known cabozantinib-related
  hypertension, managed with amlodipine). No hand-foot syndrome. No
  diarrhea. LFTs mildly elevated (ALT 52 U/L, AST 48 U/L - stable).
- Per 21 CFR Part 312, Section 312.32: Hypertension is a known class
  effect, previously reported and managed. No new reportable events.
- Per ICH E6(R3), Section 5.4.1: Drug accountability maintained.

## Site Utilization

- Overall robot utilization: approximately 55% (16 of 29 instances engaged
  at peak concurrent activity)
- Queue lengths: 0-1 (brief queues at check-in kiosks during arrival surge)
- Average wait time: 6.2 minutes (slightly elevated due to arrival volume)
- Robot cleaning cycles: 4 (TRACK-01, COBOT-02, NEEDLE-02, IMAGE-03
  post-procedure)
- Concurrent patients on-site: approximately 20

## Regulatory Compliance Notes

### ICH E6(R3) - Adaption
- Section 1.1.1: All procedures conducted in accordance with ethical
  principles and applicable GCP requirements. Evening operations maintained
  identical safety standards to daytime hours, consistent with on-demand
  model validation.
- Section 2.9.1: Complete audit trails maintained for all 10 new patient
  encounters including RT dose records, biopsy force measurements, imaging
  quality metrics, and rehabilitation gait data with synchronized UTC
  timestamps.
- Section 4.2.1: Data capture systems operating at full capacity during
  evening peak. No data integrity concerns. Real-time upload to central
  FHIR server confirmed for all active procedures.
- Section 5.4.1: Investigational product accountability maintained for
  temozolomide (P0126) and cabozantinib (P0131). Pharmacy records current.

### 21 CFR Part 50 - Adaption
- Section 50.25: All 10 new patients had previously completed informed
  consent including Physical AI system disclosure, PSL readiness scores,
  and right to non-Physical AI alternatives.
- Section 50.30: Pre-procedure safety matrix completed for all procedures:
  authorization verified, patient identity confirmed, clinical data accessed
  via FHIR, robot readiness confirmed, environmental checks passed.
- Subpart D (Pediatric): Additional protections applied for PAT-ODMND-0127
  (15M, osteosarcoma) and PAT-ODMND-0130 (8F, ALL). Parental permission
  and age-appropriate assent documented. Independent pediatric advocate
  notified per institutional policy.

### 21 CFR Part 312 - Adaption
- Section 312.32: Safety reporting systems active. Two IND patients
  monitored (P0126 temozolomide, P0131 cabozantinib). No reportable
  adverse events this hour.
- Section 312.60: Investigator obligations maintained across extended
  operating hours. On-demand scheduling does not diminish protocol
  adherence or safety monitoring intensity.
- Section 312.62: Investigator recordkeeping maintained for all patients
  including Physical AI system interaction logs, vital signs, and
  procedure outcome metrics.

## Complementary Framework References

The Physical Safety Level (PSL) framework evaluates robot-patient clinical
trial performance across three dimensions: Omniscient (sensing and AI
awareness), Omnipresent (coverage and availability), and Omnipotent
(procedural capability). RT Positioning Dim B increased +0.1 this hour
reflecting vault sharing efficiency gains during the dual-shift period.
Cumulative Site PSL: 65.2.

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary robot technical
interoperability scoring. The evening peak period exercises multiple robot
types concurrently, testing interoperability under load.
See physical-ai-oncology-trials/unification/usl/paper/usl_oncology_trials.tex.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) demonstrated autonomous Physical AI trial
orchestration for an individual patient. PAT-ODMND-0127's dual-robot
session (HUMAN-01 then REHAB-01) represents a multi-stage journey within
the on-demand context, with seamless robot-to-robot handoff.
See physical-ai-oncology-trials/patient-journey/paper/patient_journey_paper.tex.
