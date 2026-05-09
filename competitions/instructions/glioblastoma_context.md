# Glioblastoma Clinical Context

This file fixes the patient, disease, and procedure that every future commit references. The future session must use the values listed here verbatim. Variation between iterations is permitted only through the seed and parameter sweep mechanism defined in `commit_04_iteration_design.md`.

## Patient

- Patient ID: PAT-GBM-0001
- Age: 62 years
- Sex: female
- ECOG performance status: 1
- Karnofsky performance status: 80
- Comorbidities: controlled hypertension, no diabetes, no anticoagulation
- Allergies: none

The patient identifier follows the existing repository convention used in `patient-journey/master_journey.py` for PAT-2026-0042 and in `new-trial/national-24-7-trial/hour-00/` for PAT-CONT-0001 through PAT-CONT-0004. The PAT-GBM-0001 identifier is unique to this v3.9.0 simulation.

## Disease

- Tumor: glioblastoma, IDH-wildtype, WHO CNS5 grade 4
- MGMT promoter: methylated
- Anatomic location: right frontal lobe, deep to the middle frontal gyrus
- Maximum diameter on T1 post-contrast: 4.2 cm
- T2 FLAIR penumbra: 6.4 cm maximum diameter
- Eloquent cortex proximity: 1.8 cm to motor strip, 3.1 cm to Broca area (left dominant; tumor on non-dominant side)
- Preoperative KPS: 80
- Preoperative motor exam: 5/5 throughout
- Preoperative speech exam: fluent

## Procedure

- Procedure: stereotactic-guided right frontal craniotomy with maximal safe resection
- Anesthesia: general endotracheal
- Position: supine, head turned 30 degrees left, fixed in Mayfield clamp
- Navigation: rigid registration with preoperative MRI plus intraoperative MRI at 30 minutes
- Adjuncts: 5-aminolevulinic acid (5-ALA) fluorescence guidance, intraoperative ultrasound at 15-minute intervals
- Goal: gross total resection of contrast-enhancing tumor

## Procedure Timeline (1 hour, millisecond resolution)

The future Commit 1 architecture document fixes the following timeline. All times are seconds from procedure start. The future Commit 4 iteration design varies only the within-phase parameters and never the phase boundaries.

| Phase | Start (s) | End (s) | Duration (s) | Description |
|-------|-----------|---------|--------------|-------------|
| Setup and registration | 0.000 | 600.000 | 600.000 | Robot calibration, tool zeroing, navigation registration |
| Dural opening and exposure | 600.000 | 900.000 | 300.000 | Dural opening, ultrasound mapping |
| Tumor resection coarse | 900.000 | 2400.000 | 1500.000 | Bulk tumor removal under 5-ALA fluorescence |
| Tumor resection fine | 2400.000 | 3300.000 | 900.000 | Margin assessment, fine resection near eloquent cortex |
| Hemostasis and closure prep | 3300.000 | 3600.000 | 300.000 | Bipolar hemostasis, irrigation, robot withdrawal |

## Regulatory Framework

The future simulation references the existing repository regulatory adaptations. The future Commit 3 control loop must enforce the runtime safety constraints defined in those documents.

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368). Section 2.3 medical care, section 2.10 safety reporting, section 2.12 investigator oversight of physical AI.
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707). Section 50.30 task-order lifecycle, runtime safety monitoring at 1 kHz, forbidden operations.
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628). Section 312.404 human oversight, section 312.62 investigator recordkeeping.
- IEC 80601-2-77. Force limits referenced in `patient-journey/stage_05_surgery.py`: 15.0 N tip force, 5.0 N lateral force.

## Source Files Cited

- `patient-journey/stage_05_surgery.py`. Provides the task-order lifecycle states, force limits, forbidden operations, and 1 kHz runtime safety monitoring pattern.
- `patient-journey/patient_state.py`. Provides the `PatientJourneyState`, `SurgicalRecord`, and `AdverseEvent` dataclasses used by the future Commit 5 outcomes file.
- `new-trial/national-24-7-trial/hour-00/hour_00_simulation.md`. Provides the minute-resolution narrative format that the future Commit 1 architecture document compresses into a 1-hour millisecond-resolution table.
- `new-trial/psl_framework.md`. Provides the PSL scoring framework referenced by the future Commit 5 quality metric.
