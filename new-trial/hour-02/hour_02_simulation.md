# Hour 02: 02:00-02:59 - Overnight Low Volume Operations

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 02 continues the overnight low-volume period with 1 new arrival and 1
discharge. The facility maintains minimal staffing with 3 patients on-site
at hour end. A scheduled preventive calibration for COBOT-03 occurs during
the first 30 minutes, and HUMAN-02 undergoes a charging cycle. The single
new arrival, PAT-ODMND-0007, presents for an early morning radiotherapy
session in Vault 2.

## Site Status at 02:00

- Total patients on-site: 3 (PAT-ODMND-0003, PAT-ODMND-0005, PAT-ODMND-0006)
- Active procedures: 0
- Robots in active mode: 1 (COMPN-03 passive monitoring)
- Robots in standby mode: 27
- Robots in maintenance: 1 (COBOT-03 preventive calibration 02:00-02:30)
- Queue length: 0 across all stations
- Site safety officer on duty: SSO-N1 (night shift)

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage | ECOG | Robot Needed |
|-----------|------|-----|-----|-------------|-------|------|-------------|
| PAT-ODMND-0007 | 02:35 | 58 | F | NSCLC squamous cell | IIIB | 1 | RT Motion-Tracking (7), RT Positioning (3) |

Patient PAT-ODMND-0007 is a 58-year-old female with Stage IIIB NSCLC
squamous cell carcinoma presenting for an early morning radiotherapy session.
She selected the 02:00-03:00 window via the patient portal to accommodate
her preference for early treatment before daily responsibilities. ECOG
performance status 1. Treatment plan: 2 Gy per fraction, fraction 8 of 30.
Prior fractions delivered at this site. Assigned to TRACK-02 and RTPOS-02
in Vault 2.

## Overnight Recovery Patients (Continuing)

| Patient ID | Age | Sex | Cancer Type | Status | Since |
|-----------|-----|-----|-------------|--------|-------|
| PAT-ODMND-0003 | 61 | M | Mediastinal tumor | Post-surgical recovery (improving) | 22:30 prior day |
| PAT-ODMND-0005 | 8 | M | Pediatric ALL | Overnight companion monitoring (sleeping) | 21:00 prior day |
| PAT-ODMND-0006 | 45 | M | Liver mets | Post-imaging review (discharged 02:10) | Hour 01 |

## Active Procedures This Hour

### RT Motion-Tracking Session (02:40-02:58)
- Patient: PAT-ODMND-0007
- Robots: TRACK-02 (RT Motion-Tracking, Instance 2), RTPOS-02 (RT Positioning, Instance 2)
- Vault: Radiotherapy Vault 2
- Procedure: Fraction 8 of 30, 2 Gy delivery to right hilum lesion
- Duration: 18 minutes (positioning 3 min, calibration 2 min, treatment 11 min, exit 2 min)
- Beam gating efficiency: 93.8%
- Breathing amplitude: 3.6 mm (within 2-3 mm tolerance after coaching)
- Marker displacement: 1.5 mm average
- Treatment interruptions: 0
- Outcome: Successful completion. Full dose delivered.

Minute-by-minute summary (active procedure):
- 02:40 - RTPOS-02 positions patient on 6-DOF couch, immobilization verified
- 02:41 - Couch alignment confirmed, CBCT verification image acquired
- 02:42 - TRACK-02 calibration, marker block placed on chest
- 02:43 - Breathing pattern baseline established, gating window set
- 02:44 - Beam-on, first field. Gating active.
- 02:47 - Field 1 complete (0.8 Gy delivered)
- 02:48 - Gantry rotation to field 2
- 02:49 - Beam-on, second field
- 02:52 - Field 2 complete (0.7 Gy delivered)
- 02:53 - Gantry rotation to field 3
- 02:54 - Beam-on, third field
- 02:55 - Field 3 complete (0.5 Gy delivered). Total: 2.0 Gy.
- 02:56 - Marker block removed, patient assisted to seated position
- 02:57 - Post-treatment vitals check
- 02:58 - Patient exits vault. Procedure complete.

## Patient Departures This Hour

| Patient ID | Time | Outcome | Notes |
|-----------|------|---------|-------|
| PAT-ODMND-0006 | 02:10 | Discharged | Imaging results reviewed, no acute findings requiring admission |

PAT-ODMND-0006 (45M, liver metastases) was discharged at 02:10 following
review of post-imaging results from the Hour 01 session. The attending
radiologist confirmed no acute findings necessitating inpatient admission.
Follow-up imaging scheduled at next outpatient visit.

## Adverse Events

None this hour.

## Investigational Drug Administrations

None this hour. (PAT-ODMND-0007 receiving standard-of-care RT only.)

## Site Utilization

- Overall robot utilization: approximately 5% (TRACK-02 and RTPOS-02 active for P0007, COMPN-03 passive monitoring, COBOT-03 maintenance cycle)
- Queue lengths: 0 across all stations
- Average wait time: 0 minutes (immediate robot availability)
- Robot cleaning cycles: 1 (TRACK-02 post-procedure initiated at 02:58)
- Maintenance events: 1 (COBOT-03 preventive calibration 02:00-02:30)

## Regulatory Compliance Notes

### ICH E6(R3) - Adaption
- Section 1.1.1: All procedures conducted in accordance with ethical principles
  and applicable GCP requirements. Overnight operations maintained identical
  safety standards to daytime operations.
- Section 2.9.1: Complete audit trail maintained for RT Motion-Tracking session
  including beam-on times, dose delivery records, and gating efficiency logs.
  COBOT-03 preventive calibration documented with pre- and post-calibration
  measurements per equipment maintenance SOPs.
- Section 4.2.7: COBOT-03 maintenance records archived with timestamped
  calibration data. HUMAN-02 charging cycle logged.

### 21 CFR Part 50 - Adaption
- Section 50.25: PAT-ODMND-0007 had previously completed informed consent
  including Physical AI system disclosure, USL readiness scores, and right
  to non-Physical AI alternatives.
- Section 50.30: Pre-procedure safety matrix completed for PAT-ODMND-0007:
  authorization verified, patient identity confirmed, clinical data accessed
  via FHIR, robot readiness confirmed for both TRACK-02 and RTPOS-02,
  environmental checks passed.
- Subpart D: PAT-ODMND-0005 continues under pediatric protections with
  parent/guardian present in adjacent family area.

### 21 CFR Part 312 - Adaption
- Section 312.62: Investigator recordkeeping maintained for all overnight
  patients including Physical AI system interaction logs and vital sign records.
  PAT-ODMND-0006 discharge documentation completed with imaging findings.
- Section 312.32: Safety reporting systems active and monitoring all patients.
  No reportable events this hour.

## Complementary Framework References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary robot technical
interoperability scoring. RT Motion-Tracking Robot TRACK-02 and RT
Positioning Robot RTPOS-02 operate on platforms evaluated at USL scores
consistent with the Intermediate-to-Advanced band, reflecting strong
simulation switching and AI integration capabilities.
See physical-ai-oncology-trials/unification/usl/paper/usl_oncology_trials.tex.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) demonstrated autonomous Physical AI trial
orchestration for an individual patient. PAT-ODMND-0007's ongoing RT course
represents Stage 5-equivalent treatment delivery within a multi-patient,
multi-cancer-type, on-demand operational context.
See physical-ai-oncology-trials/patient-journey/paper/patient_journey_paper.tex.
