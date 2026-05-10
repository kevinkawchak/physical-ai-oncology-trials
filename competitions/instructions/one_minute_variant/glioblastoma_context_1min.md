# Glioblastoma Clinical Context: 1-Minute Variant

This file fixes the patient, disease, and procedure boundaries that every future commit references for the 1-minute variant. The future session must use the values listed here verbatim. Variation between iterations is permitted only through the seed and parameter sweep mechanism defined in `commit_04_iterations_1min.md`. The patient and disease are inherited verbatim from the parent `competitions/instructions/glioblastoma_context.md`; only the procedure timeline differs.

## Patient (inherited verbatim from parent v3.9.0)

- Patient ID: PAT-GBM-0001
- Age: 62 years
- Sex: female
- ECOG performance status: 1
- Karnofsky performance status: 80
- Comorbidities: controlled hypertension, no diabetes, no anticoagulation
- Allergies: none

The patient identifier follows the existing repository convention used in `patient-journey/master_journey.py` for PAT-2026-0042 and in `new-trial/national-24-7-trial/hour-00/` for PAT-CONT-0001 through PAT-CONT-0004. The PAT-GBM-0001 identifier is shared between the parent v3.9.0 1-hour scenario and this v3.9.1 1-minute variant so that competition outcomes can be compared across both robots and both procedure durations.

## Disease (inherited verbatim from parent v3.9.0)

- Tumor: glioblastoma, IDH-wildtype, WHO CNS5 grade 4
- MGMT promoter: methylated
- Anatomic location: right frontal lobe, deep to the middle frontal gyrus
- Maximum diameter on T1 post-contrast: 4.2 cm
- T2 FLAIR penumbra: 6.4 cm maximum diameter
- Eloquent cortex proximity: 1.8 cm to motor strip, 3.1 cm to Broca area (left dominant; tumor on non-dominant side)
- Preoperative KPS: 80
- Preoperative motor exam: 5/5 throughout
- Preoperative speech exam: fluent

## Tumor Volume Computation for the 1-Minute Removal Rate

The 1-minute scenario fixes the contrast-enhancing tumor volume to remove at 38,800 mm cubed, computed as a sphere of diameter 4.2 cm. The required removal rate is therefore 38,800 mm cubed divided by 60 seconds equals 647 mm cubed per second of mean removal across the 60-second window. The peak removal rate during Phase 2 is 800 mm cubed per second to allow for ramp-up at the start of Phase 2 and ramp-down at the end of Phase 3. The peak removal rate is 200 times the maximum CUSA aspirator rate of 2 to 5 mm cubed per second on the current SOTA Medtronic ROSA ONE Brain v3.0; this is why the 1-minute variant requires the hypothetical Medtronic NeuroSpeed 1.0 documented in `robot_specification_neurospeed.md`.

## Procedure (variant overrides parent v3.9.0)

- Procedure: stereotactic-guided right frontal craniotomy with maximal safe resection completed in 60 seconds.
- Anesthesia: general endotracheal, induced and stabilized during the precomputed pre-op window (T-1800 s to T+0 s).
- Position: supine, head turned 30 degrees left, fixed in Mayfield clamp; clamp position frozen at simulation start.
- Navigation: rigid registration with preoperative MRI completed during pre-op; intraoperative 0.5 T MRI at 30 frames per second runs continuously during Phase 2 and Phase 3.
- Adjuncts: 5-aminolevulinic acid (5-ALA) fluorescence guidance at 100 frames per second on arm 4 during Phase 3; intraoperative ultrasound rapid mapping during Phase 1.
- Goal: gross total resection of contrast-enhancing tumor in 60 seconds with hemostasis verification before arm withdrawal.

## Procedure Timeline (1 minute, millisecond resolution)

The 4-phase timeline below is the canonical reference for every future commit. The phase boundaries are immutable; only within-phase parameters may vary across iterations.

| Phase | Start (s) | End (s) | Duration (s) | Description |
|-------|-----------|---------|--------------|-------------|
| Pre-op (precomputed, not in committed simulation) | T-1800 | T+0 | 30 minutes | Anesthesia, registration, dural opening, multi-arm setup; precomputed and frozen at simulation start. |
| Phase 1 dural opening final and exposure | 0.000 | 5.000 | 5 s | Final dural opening, ultrasound rapid mapping, 5-ALA UV on. All four arms positioned at the surgical field; arm 4 imaging at 30 fps. |
| Phase 2 bulk tumor resection | 5.000 | 45.000 | 40 s | All four arms active. Arm 1 cuts at peak 800 mm cubed per second using hybrid ultrasonic plus waterjet plus pulsed plasma. Arm 2 coagulates behind arm 1 with bipolar plus irrigation. Arm 3 suctions and collects tissue for downstream margin pathology. Arm 4 images at 30 fps with 0.5 T MRI plus 5-ALA fluorescence. |
| Phase 3 margin assessment and fine resection | 45.000 | 55.000 | 10 s | Arm 1 reduces removal rate to 200 mm cubed per second under tight margin control. Arm 4 increases imaging to 100 fps for the highest-resolution margin scan. Arm 2 continues hemostasis. Arm 3 continues suction. |
| Phase 4 hemostasis verification and arm withdrawal | 55.000 | 60.000 | 5 s | Arms 1 and 3 retract along precomputed safe egress paths. Arm 2 performs the final hemostasis pass and confirms zero active bleeding. Arm 4 records the final margin scan and posts the resection completeness percentage to the simulation log. |

## Per-Arm Tool Assignment

The four arms are assigned tools in advance and do not change tools during the 1-minute procedure. Tool changeover would consume more than 5 seconds of the 60-second budget and is therefore deferred to the precomputed pre-op window.

| Arm | Tool | Primary task | Sample rates |
|-----|------|--------------|--------------|
| 1 | Hybrid ultrasonic plus waterjet plus pulsed plasma | Bulk tumor resection (Phase 2) and fine margin resection (Phase 3) | 10 kHz force, 1 kHz commands |
| 2 | Bipolar coagulation plus irrigation | Real-time hemostasis behind arm 1 (Phases 2, 3, 4) | 10 kHz force, 1 kHz commands |
| 3 | Suction plus tissue collection | Continuous removal of debris (Phases 2, 3) and final cavity clean (Phase 4) | 10 kHz pressure, 100 Hz commands |
| 4 | 0.5 T MRI plus 5-ALA fluorescence camera plus ultrasound probe | Continuous margin imaging across all phases | 30 fps Phase 1 to 2, 100 fps Phase 3, 1 kHz force |

## Regulatory Framework (inherited from parent v3.9.0 plus 1-minute additions)

The 1-minute variant inherits the parent regulatory framework verbatim. The future Commit 3 control loop must enforce the runtime safety constraints defined in those documents on each of the four arms simultaneously. The cumulative force across all four arms on the patient frame is capped at 12 N per `multi_arm_coordination.md`.

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368). Section 2.3 medical care, section 2.10 safety reporting, section 2.12 investigator oversight of physical AI.
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707). Section 50.30 task-order lifecycle, runtime safety monitoring at 1 kHz per arm, forbidden operations.
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628). Section 312.404 human oversight, section 312.62 investigator recordkeeping.
- IEC 80601-2-77. Per-arm force limits referenced in `robot_specification_neurospeed.md`: 5.0 N tip force per arm, 1.0 N lateral force per arm, 12 N cumulative across all arms on patient frame.
- IEC 62304 software lifecycle for safety-critical software at the 1 kHz heartbeat layer documented in `multi_arm_coordination.md`.

## Source Files Cited

- `competitions/instructions/glioblastoma_context.md`. Source for the patient and disease values that this 1-minute variant inherits verbatim. The 5-phase 1-hour timeline in the parent file is replaced by the 4-phase 1-minute timeline above for this variant.
- `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md`. Source for the per-arm tool capabilities and the 800 mm cubed per second peak removal rate that drive the 4-phase timeline.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the cumulative force limit and the cross-arm safety zone gating that the per-phase descriptions reference.
- `patient-journey/stage_05_surgery.py`. Provides the task-order lifecycle states, force limits, forbidden operations, and 1 kHz runtime safety monitoring pattern used per arm.
- `patient-journey/patient_state.py`. Provides the `PatientJourneyState`, `SurgicalRecord`, and `AdverseEvent` dataclasses used by the future Commit 5 outcomes file.
