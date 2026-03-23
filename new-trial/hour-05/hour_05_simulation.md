# Hour 05: 05:00-05:59 - Dawn Ramp-Up Operations

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 05 represents the dawn ramp-up period of the 24-hour on-demand simulation
cycle. Patient volume increases with 5 new arrivals spanning lung RT,
pediatric companion therapy, brain RT, cobot biopsy, and liver imaging. Three
overnight patients are discharged, bringing the end-of-hour census to 7
patients. This hour marks the first use of concurrent temozolomide under
investigational dosing guided by a digital twin pharmacokinetic model.

## Site Status at 05:00

- Total patients on-site: 5 (P0003 recovery, P0005 pediatric, P0010 post-biopsy, P0011 post-imaging, P0012 post-rehab)
- Active procedures: 0
- Robots in active mode: 0
- Robots in standby mode: 29 (all instances)
- Robots in maintenance: 0
- Queue length: 0 across all stations
- Site safety officer on duty: SSO-N1 (night shift, handover to SSO-D1 at 06:00)

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage/Grade | ECOG | Robot Assigned |
|-----------|------|-----|-----|-------------|-------------|------|---------------|
| PAT-ODMND-0013 | 05:05 | 47 | M | NSCLC adenocarcinoma | Stage IIB | 1 | TRACK-01 |
| PAT-ODMND-0014 | 05:15 | 6 | F | Pediatric AML | - | 1 | COMPN-01 |
| PAT-ODMND-0015 | 05:22 | 65 | F | Glioblastoma | Stage IV | 1 | RTPOS-01 |
| PAT-ODMND-0016 | 05:35 | 39 | M | Forearm sarcoma | Grade I | 0 | COBOT-02 |
| PAT-ODMND-0017 | 05:50 | 71 | F | Liver mets (colorectal) | Stage IV | 1 | IMAGE-04 |

Patient PAT-ODMND-0013 is a 47-year-old male with Stage IIB NSCLC
adenocarcinoma presenting for RT motion-tracking radiotherapy. He selected the
early morning window via the patient portal. ECOG performance status 1.
Treatment plan: 2 Gy per fraction with respiratory gating.

Patient PAT-ODMND-0014 is a 6-year-old female with pediatric acute myeloid
leukemia (AML) presenting for a companion robot session prior to her morning
chemotherapy cycle. She arrived with her mother at 05:15, selected for an
early quiet slot to minimize waiting room anxiety. ECOG 1. Pediatric
protections apply per 21 CFR Part 50 Subpart D.

Patient PAT-ODMND-0015 is a 65-year-old female with Stage IV glioblastoma
multiforme presenting for RT positioning and treatment delivery. She is
receiving concurrent temozolomide at investigational dosing per digital twin
pharmacokinetic model (see Investigational Drug Administrations below). ECOG 1.

Patient PAT-ODMND-0016 is a 39-year-old male with Grade I forearm sarcoma
presenting for cobot-assisted biopsy of a newly identified lesion. ECOG 0.
No prior biopsies at this site.

Patient PAT-ODMND-0017 is a 71-year-old female with Stage IV colorectal cancer
with hepatic metastases presenting for imaging assessment and steerable needle
consultation. ECOG 1. Prior colectomy 14 months ago; liver lesion identified
on surveillance imaging.

## Patients Remaining from Prior Hours

| Patient ID | Age | Sex | Cancer Type | Status at 05:00 |
|-----------|-----|-----|-------------|----------------|
| PAT-ODMND-0003 | 61 | M | Mediastinal tumor | Post-surgical recovery |
| PAT-ODMND-0005 | 8 | M | Pediatric ALL | Overnight companion monitoring |
| PAT-ODMND-0010 | 55 | M | Parotid gland tumor | Post-biopsy observation |
| PAT-ODMND-0011 | 48 | F | Hepatocellular carcinoma | Post-imaging observation |
| PAT-ODMND-0012 | 63 | M | Femur osteosarcoma | Post-rehabilitation session |

## Active Procedures This Hour

### RT Motion-Tracking Session (05:12-05:27)
- Patient: PAT-ODMND-0013
- Robot: TRACK-01 (RT Motion-Tracking, Instance 1)
- Vault: Radiotherapy Vault 1
- Procedure: 2 Gy fraction delivery to right upper lobe lesion
- Duration: 15 minutes (calibration 2 min, treatment 11 min, exit 2 min)
- Beam gating efficiency: 93.8%
- Breathing amplitude: 3.8 mm (within tolerance after coaching)
- Marker displacement: 1.6 mm average
- Treatment interruptions: 0
- Outcome: Successful completion. Full dose delivered.

Minute-by-minute summary (active procedure):
- 05:12 - Patient positioned on couch, marker block placed on chest
- 05:13 - Calibration complete, breathing pattern established at 3.8 mm
- 05:14 - Beam-on, first field. Gating active.
- 05:17 - Field 1 complete (0.9 Gy delivered)
- 05:18 - Gantry rotation to field 2
- 05:19 - Beam-on, second field
- 05:22 - Field 2 complete (0.7 Gy delivered)
- 05:23 - Gantry rotation to field 3
- 05:24 - Beam-on, third field
- 05:25 - Field 3 complete (0.4 Gy delivered). Total: 2.0 Gy.
- 05:26 - Marker block removed, patient assisted to seated position
- 05:27 - Patient exits vault. Procedure complete.

### Companion Robot Session (05:20-05:35)
- Patient: PAT-ODMND-0014
- Robot: COMPN-01 (Social Companion, Instance 1)
- Location: Pediatric Play Room 1
- Procedure: Pre-chemotherapy anxiety management session
- Duration: 15 minutes
- Anxiety score: Entry 7/10, Exit 4/10 (reduction of 3 points)
- Engagement metrics: Verbal interaction 82%, gesture response 78%
- Activities: Storytelling (5 min), breathing exercise game (4 min), guided drawing (6 min)
- Parent present: Yes (mother, observing from adjacent area)
- Outcome: Successful. Patient calm and cooperative for upcoming chemotherapy.

### RT Positioning and Treatment (05:28-05:53)
- Patient: PAT-ODMND-0015
- Robot: RTPOS-01 (RT Positioning, Instance 1)
- Vault: Radiotherapy Vault 1
- Procedure: Brain RT with thermoplastic mask positioning, 2 Gy fraction
- Duration: 25 minutes (positioning 8 min, verification 3 min, treatment 12 min, exit 2 min)
- Positioning offset: 1.2 mm (within 1.5 mm tolerance for brain RT)
- 6-DOF couch adjustments: X +0.4 mm, Y -0.3 mm, Z +0.2 mm, pitch +0.1 deg, roll -0.1 deg, yaw 0.0 deg
- Dose delivered: 2.0 Gy
- Treatment interruptions: 0
- Concurrent medication: Temozolomide (investigational dosing, see below)
- Outcome: Successful completion. Full dose delivered.

Minute-by-minute summary (active procedure):
- 05:28 - Patient arrives in vault, thermoplastic mask fitted
- 05:29 - RTPOS-01 begins 6-DOF alignment sequence
- 05:32 - Initial CBCT acquired, offset calculated: 1.2 mm total
- 05:33 - Couch adjustments applied by RTPOS-01
- 05:35 - Verification CBCT confirms alignment within tolerance
- 05:36 - Physicist approval received. Beam-on authorized.
- 05:37 - Beam-on, first arc
- 05:42 - Arc 1 complete (1.2 Gy delivered)
- 05:43 - Gantry repositioning for second arc
- 05:44 - Beam-on, second arc
- 05:49 - Arc 2 complete (0.8 Gy delivered). Total: 2.0 Gy.
- 05:50 - Mask removed, patient assisted to seated position
- 05:51 - Post-treatment neurological check: orientation intact, no acute symptoms
- 05:53 - Patient exits vault. Procedure complete.

### Cobot Biopsy (05:42-05:57)
- Patient: PAT-ODMND-0016
- Robot: COBOT-02 (Cobot, Instance 2)
- Location: Biopsy Station 2
- Procedure: Core needle biopsy of forearm soft-tissue lesion
- Duration: 15 minutes (prep 3 min, imaging 2 min, biopsy 8 min, closure 2 min)
- Repositionings: 2 (initial approach adjusted for vessel proximity, second adjustment for deeper tissue plane)
- Sample quality: Grade A (adequate for histopathology and molecular profiling)
- Samples obtained: 3 cores
- Force feedback: Peak 2.8 N (within 0.5-4.0 N safety envelope)
- Outcome: Successful. Samples sent to pathology.

Minute-by-minute summary (active procedure):
- 05:42 - Patient positioned, forearm stabilized in padded cradle
- 05:43 - Local anesthesia administered (lidocaine 1%, 5 mL)
- 05:44 - Ultrasound imaging localization of lesion (22 x 18 mm)
- 05:45 - COBOT-02 path planning complete. Initial approach vector set.
- 05:46 - First repositioning: vessel detected 3 mm from planned path, approach angle adjusted +12 degrees
- 05:47 - Needle insertion, first core obtained
- 05:48 - Needle retracted, second core trajectory planned
- 05:49 - Second repositioning: deeper tissue plane targeted, insertion depth +4 mm
- 05:50 - Second core obtained
- 05:51 - Third core obtained from lesion periphery
- 05:52 - Needle fully retracted, hemostasis check
- 05:53 - Pressure dressing applied
- 05:54 - COBOT-02 retracted to home position
- 05:55 - Patient moved to recovery observation
- 05:57 - Procedure complete, samples labeled and dispatched

### Imaging Assessment (05:55-06:13)
- Patient: PAT-ODMND-0017
- Robot: IMAGE-04 (Imaging Assistant, Instance 4)
- Location: Imaging Bay 4
- Procedure: Robotic ultrasound liver assessment with steerable needle consultation
- Duration: 18 minutes (extends into Hour 06)
- Probe pressure: 1.9 N average (within 1-3 N range)
- Image quality score: 8.0/10
- Tumor measurement: 42 x 35 mm (dominant hepatic metastasis, segment VI)
- Secondary lesion: 18 x 14 mm (segment VIII)
- Scan coverage: 94%
- Motion artifacts: 1 (minor, auto-compensated)
- Steerable needle consultation: Preliminary trajectory analysis generated for interventional team review
- Outcome: Successful. Images uploaded to DICOM server. Steerable needle approach feasibility confirmed.

## Patient Departures This Hour

| Patient ID | Time | Outcome | Notes |
|-----------|------|---------|-------|
| PAT-ODMND-0010 | 05:10 | Discharged | Post-biopsy observation complete, parotid site stable, no complications |
| PAT-ODMND-0011 | 05:15 | Discharged | Post-imaging observation complete, no contrast reactions |
| PAT-ODMND-0012 | 05:20 | Discharged | Post-rehab session complete, gait assessment satisfactory |

## Adverse Events

None this hour.

## Investigational Drug Administrations

PAT-ODMND-0015 (65F, glioblastoma, Stage IV) is receiving concurrent
temozolomide at investigational dosing determined by a digital twin
pharmacokinetic (PK) model. Per 21 CFR 312.23, this investigational dosing
regimen is conducted under an active IND application. The digital twin PK
model integrates patient-specific parameters (weight 68 kg, BSA 1.72 m2,
hepatic function CTP-A, renal clearance 82 mL/min) to optimize temozolomide
exposure while maintaining the standard Stupp protocol RT schedule.

- Drug: Temozolomide (oral)
- Dose: 85 mg/m2 (investigational, model-optimized; standard is 75 mg/m2)
- Administration: Self-administered 90 minutes prior to RT session (03:58)
- Digital twin PK prediction: Cmax 5.8 mcg/mL at T+1.5h, AUC 28.4 mcg-h/mL
- IND number: IND-2026-PAI-0047
- Regulatory basis: 21 CFR 312.23 (IND content and format)
- Monitoring: CBC with differential scheduled 7 days post-fraction
- Sponsor notification: Dose deviation from standard protocol logged per 21 CFR 312.32

## Site Utilization

- Overall robot utilization: approximately 15% (5 of 29 robots active during peak overlapping periods)
- Peak concurrent active robots: 3 (05:42-05:53: RTPOS-01, COBOT-02, COMPN-01 overlap window)
- Queue lengths: 0 across all stations
- Average wait time: 5.4 minutes (P0013: 7 min, P0014: 5 min, P0015: 6 min, P0016: 7 min, P0017: 5 min)
- Robot cleaning cycles: 5 (TRACK-01, COMPN-01, RTPOS-01, COBOT-02, IMAGE-04)
- Patient census at end of hour: 7 (P0003, P0005, P0013, P0014, P0015, P0016, P0017)

## Regulatory Compliance Notes

### ICH E6(R3) - Adaption
- Section 1.1.1: All procedures conducted in accordance with ethical principles
  and applicable GCP requirements. Dawn ramp-up operations maintained identical
  safety standards to overnight and daytime operations.
- Section 2.9.1: Complete audit trail maintained for all five active procedures
  including beam-on times, dose delivery records, biopsy sample chain of
  custody, companion interaction logs, and imaging quality metrics.
- Section 4.2.1: Data capture for all robotic procedures included synchronized
  UTC timestamps, force/pressure measurements, and positional telemetry.

### 21 CFR Part 50 - Adaption
- Section 50.25: All five new patients had previously completed informed consent
  including Physical AI system disclosure, USL readiness scores, and right to
  non-Physical AI alternatives. PAT-ODMND-0015 received additional consent for
  investigational temozolomide dosing.
- Section 50.30: Pre-procedure safety matrix completed for all procedures:
  authorization verified, patient identity confirmed, clinical data accessed
  via FHIR, robot readiness confirmed, environmental checks passed.
- Subpart D: Pediatric protections applied for PAT-ODMND-0014 (6F, AML).
  Parental consent and child assent documented. IRB-approved pediatric
  companion robot protocol followed. Mother present during session.

### 21 CFR Part 312 - Adaption
- Section 312.23: Investigational dosing of temozolomide for PAT-ODMND-0015
  conducted under active IND (IND-2026-PAI-0047). Digital twin PK model
  parameters documented. Dose rationale filed with sponsor.
- Section 312.32: Safety reporting systems active and monitoring all patients.
  No reportable events this hour. Investigational drug administration logged.
- Section 312.62: Investigator recordkeeping maintained for all patients
  including Physical AI system interaction logs and vital sign records.

## Complementary Framework References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary robot technical
interoperability scoring. All five robot types activated this hour (TRACK-01,
COMPN-01, RTPOS-01, COBOT-02, IMAGE-04) operate on platforms evaluated at USL
scores consistent with the Intermediate-to-Advanced bands.
See physical-ai-oncology-trials/unification/usl/paper/usl_oncology_trials.tex.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) demonstrated autonomous Physical AI trial
orchestration for an individual patient. The dawn ramp-up period introduces
five distinct cancer journeys concurrently, demonstrating multi-patient
on-demand scheduling at increasing volume.
See physical-ai-oncology-trials/patient-journey/paper/patient_journey_paper.tex.
