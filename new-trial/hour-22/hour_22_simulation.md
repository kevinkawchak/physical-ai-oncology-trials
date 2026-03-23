# Hour 22: 22:00-22:59 - Wind-Down Operations

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 22 represents the wind-down phase of the 24-hour on-demand simulation
cycle. Most daytime and evening procedures have completed. Three new arrivals
present for late-evening services including radiotherapy, imaging, and a
pediatric overnight admission. The approaching 23:00 night shift change
triggers SSO-N2 preparation to replace SSO-E1. SURG-01 enters a scheduled
preventive maintenance window. Concurrent patient census drops to
approximately 6 as discharges outpace arrivals.

## Site Status at 22:00

- Total patients on-site: 6 (3 continuing, 3 new arrivals this hour)
- Active procedures: 1 (winding down from prior hour)
- Robots in active mode: 4
- Robots in standby mode: 24
- Robots in maintenance: 1 (SURG-01 preventive maintenance from 22:30)
- Queue length: 0 across all stations
- Site safety officer on duty: SSO-E1 (evening shift, handoff to SSO-N2 at 23:00)
- Robot utilization: approximately 15%

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage | ECOG | Robot Needed |
|-----------|------|-----|-----|-------------|-------|------|-------------|
| PAT-ODMND-0171 | 22:10 | 63 | M | NSCLC adenocarcinoma | IIIA | 1 | RT Motion-Tracking (7) |
| PAT-ODMND-0172 | 22:30 | 55 | F | HCC | II | 1 | Imaging Assistant (8) |
| PAT-ODMND-0173 | 22:45 | 8 | F | Pediatric ALL | - | 1 | Social Companion (5) |

Patient PAT-ODMND-0171 is a 63-year-old male with Stage IIIA NSCLC
adenocarcinoma presenting for scheduled late-evening radiotherapy. He selected
the 22:00-23:00 window via the patient portal due to preference for quieter
facility hours and reduced travel time. ECOG performance status 1. Treatment
plan: 2 Gy per fraction, fraction 18 of 30. Prior fractions delivered at this
site with consistent beam gating efficiency above 93%.

Patient PAT-ODMND-0172 is a 55-year-old female with Stage II hepatocellular
carcinoma presenting for pre-treatment liver imaging assessment. She selected
the late-evening slot to accommodate a next-day procedure schedule. ECOG
performance status 1.

Patient PAT-ODMND-0173 is an 8-year-old female with pediatric acute
lymphoblastic leukemia (ALL) admitted for overnight stay in advance of
morning chemotherapy scheduled for Hour 06. She is accompanied by a parent
guardian. COMPN-02 assigned for overnight companion monitoring, paralleling
the Hour 00 care model used for PAT-ODMND-0005. ECOG performance status 1.

## Continuing Patients on Site

| Patient ID | Age | Sex | Cancer Type | Status | Since |
|-----------|-----|-----|-------------|--------|-------|
| PAT-ODMND-0154 | 58 | M | Colorectal adenocarcinoma | Post-surgical recovery | 18:45 |
| PAT-ODMND-0168 | 47 | F | Breast invasive ductal | Post-biopsy observation | 21:15 |
| PAT-ODMND-0170 | 72 | M | Prostate adenocarcinoma | Post-RT monitoring | 21:30 |

## Active Procedures This Hour

### RT Motion-Tracking Session (22:18-22:36)
- Patient: PAT-ODMND-0171
- Robot: TRACK-01 (RT Motion-Tracking, Instance 1)
- Vault: Radiotherapy Vault 1
- Procedure: Fraction 18 of 30, 2 Gy delivery to right middle lobe lesion
- Duration: 18 minutes (calibration 2 min, treatment 14 min, exit 2 min)
- Beam gating efficiency: 95.1%
- Breathing amplitude: 3.8 mm (within 2-3 mm tolerance after coaching)
- Marker displacement: 1.6 mm average
- Treatment interruptions: 0
- Outcome: Successful completion. Full dose delivered.

Minute-by-minute summary (active procedure):
- 22:18 - Patient positioned on couch, marker block placed on chest
- 22:19 - Calibration complete, breathing pattern baseline captured
- 22:20 - Beam-on, first field. Gating active. 120 Hz marker tracking engaged.
- 22:24 - Field 1 complete (1.0 Gy delivered)
- 22:25 - Gantry rotation to field 2
- 22:26 - Beam-on, second field
- 22:30 - Field 2 complete (0.6 Gy delivered)
- 22:31 - Gantry rotation to field 3
- 22:32 - Beam-on, third field
- 22:34 - Field 3 complete (0.4 Gy delivered). Total: 2.0 Gy.
- 22:35 - Marker block removed, patient assisted to seated position
- 22:36 - Patient exits vault. Procedure complete.

### Imaging Assessment (22:38-22:52)
- Patient: PAT-ODMND-0172
- Robot: IMAGE-01 (Imaging Assistant, Instance 1)
- Bay: Imaging Bay 1
- Procedure: Robotic ultrasound liver assessment
- Duration: 14 minutes
- Probe pressure: 1.7 N steady (within 1-3 N range)
- Image quality score: 8.5/10
- Tumor measurement: 31 mm x 24 mm (primary HCC lesion)
- Scan coverage: 94%
- Motion artifact count: 1 (minor, auto-compensated)
- Outcome: Successful. Images uploaded to DICOM server for digital twin
  calibration and treatment planning.

### Pediatric Companion Monitoring (22:50 onward)
- Patient: PAT-ODMND-0173
- Robot: COMPN-02 (Social Companion, Instance 2)
- Location: Pediatric Ward, Room 2
- Mode: Overnight companion monitoring (continuous)
- Activities: Initial engagement with patient and parent, orientation to ward
  environment, nightlight mode activation, ambient monitoring initialized.
  Heart rate monitoring via room sensors established at 22:55.
- Parent guardian: Present, accommodated in bedside recliner.
- Outcome: Ongoing into Hour 23 and overnight hours.

## Patient Departures This Hour

| Patient ID | Time | Outcome | Notes |
|-----------|------|---------|-------|
| PAT-ODMND-0168 | 22:20 | Discharged | Post-biopsy observation complete, no complications |
| PAT-ODMND-0170 | 22:40 | Discharged | Post-RT monitoring complete, vitals stable |
| PAT-ODMND-0171 | 22:42 | Discharged | RT fraction 18 complete, no adverse effects |
| PAT-ODMND-0172 | 22:56 | Discharged | Imaging complete, images uploaded |

## Adverse Events

None this hour. All procedures completed within normal parameters. No robot
faults, no patient safety incidents, no protocol deviations.

## Investigational Drug Administrations

None this hour. PAT-ODMND-0171 receiving standard-of-care RT only.
PAT-ODMND-0173 chemotherapy scheduled for Hour 06 the following morning.

## Site Utilization

- Overall robot utilization: approximately 15% (4-5 of 29 robots active at
  any given time during wind-down)
- Queue lengths: 0 across all stations
- Average wait time: 6.3 minutes (immediate robot availability)
- Robot cleaning cycles: 2 (TRACK-01 post-procedure, IMAGE-01 post-procedure)

## Night Shift Change Preparation

SSO-E1 (evening shift safety officer) began handoff preparation at 22:45 for
the 23:00 shift change. SSO-N2 (night shift replacement) scheduled to arrive
at 22:50 for briefing. Handoff checklist includes:
- Current patient census review (2 remaining: P0154 recovery, P0173 overnight)
- Active robot status and SURG-01 maintenance window status
- Pending orders for overnight period
- Emergency protocol review for overnight staffing levels

## SURG-01 Preventive Maintenance Window

SURG-01 entered scheduled preventive maintenance at 22:30 in the Robot
Maintenance Bay. Maintenance scope:
- Joint calibration verification across all 7 axes
- Instrument channel integrity check
- Force sensor recalibration
- Software update staging (to be applied during maintenance)
- Estimated completion: 04:00 (5.5-hour maintenance window)
- Coverage: SURG-02 and SURG-03 remain available for any emergency surgical
  needs during the maintenance period.

Per 21 CFR 820.72, calibration records and maintenance logs are maintained in
the device history record. Maintenance activities follow the site preventive
maintenance schedule documented in the quality management system.

## Regulatory Compliance Notes

### ICH E6(R3) - Adaption
- Section 1.1.1: All procedures conducted in accordance with ethical principles
  and applicable GCP requirements. Wind-down period maintained identical safety
  standards to peak daytime operations.
- Section 2.9.1: Complete audit trail maintained for RT Motion-Tracking session
  including beam-on times, dose delivery records, and gating efficiency logs.
- Section 4.2.1: Data capture for imaging session included probe pressure
  measurements at 50 Hz, image quality metrics, and tumor measurements with
  automated DICOM upload and digital twin synchronization.
- Section 2.10.1: Adverse event monitoring maintained continuously through
  wind-down period with no events to report.

### 21 CFR Part 50 - Adaption
- Section 50.25: Informed consent documentation verified for all three new
  arrivals. PAT-ODMND-0173 consent obtained from parent/legal guardian per
  21 CFR 50.55 requirements for pediatric subjects in clinical investigations.
  Physical AI disclosure provided including companion robot overnight monitoring
  capabilities and limitations.
- Section 50.55: Pediatric assent obtained from PAT-ODMND-0173 in
  age-appropriate language. Parent consent documented as IC-2026-1847.

### 21 CFR Part 820 - Quality System
- Section 820.72: SURG-01 preventive maintenance initiated per scheduled
  calibration interval. Maintenance procedure documented in device history
  record. Equipment taken out of service with appropriate status labeling.
- Section 820.90: SURG-01 maintenance status communicated to all relevant
  personnel including incoming night shift SSO-N2.

### 21 CFR Part 11 - Electronic Records
- Section 11.10: All electronic records generated during wind-down procedures
  maintained with appropriate access controls, audit trails, and electronic
  signatures. Night shift handoff documentation signed electronically by
  SSO-E1.
