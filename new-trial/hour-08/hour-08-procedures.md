# Hour 08: Active Procedures

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Procedure Summary

Hour 08 records the highest concurrent procedure load of the trial to date,
with up to 11 simultaneous active procedures at peak (approximately 08:50).
Procedures span surgical resection, cobot biopsy, RT positioning, RT motion
tracking, needle placement, robotic imaging, humanoid therapy, companion
robot sessions, and rehabilitation exoskeleton sessions. One continuing
surgical case from Hour 07 (PAT-ODMND-0024) remains in progress. One adverse
event occurs during needle placement (see hour-08-adverse-events.md).

## Regulatory Framework References

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368)
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707)
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628)

## Continuing Procedure: PAT-ODMND-0024 Surgery

- Patient: PAT-ODMND-0024 (58M, mediastinal tumor, Stage III)
- Robot: SURG-01 (Surgical Robot, Instance 1)
- Suite: Surgical Suite 1
- Started: 07:40 (Hour 07)
- Expected completion: approximately 09:10 (Hour 09)
- Status at 08:00: 20 minutes into procedure, mediastinal dissection phase
- Status at 08:30: 50 minutes, tumor mobilization in progress
- Status at 08:59: 79 minutes, hemostasis and closure preparation
- Blood loss at 08:59: 180 mL (within acceptable range)
- Surgical force range: 2-12 N (within SURG-01 specification)
- Treatment interruptions: 0
- Multi-patient surgical awareness: SURG-01 AI system now sharing contextual
  data with SURG-02 (P0032) per PSL Dimension A enhancement

## Procedure 1: PAT-ODMND-0032 - Mediastinal Tumor Surgery

- Patient: PAT-ODMND-0032 (54M, mediastinal tumor, Stage III, ECOG 1)
- Robot: SURG-02 (Surgical Robot, Instance 2)
- Suite: Surgical Suite 2
- Pre-op atezolizumab neoadjuvant dose administered per IND (21 CFR 312.23)
- Duration: Estimated 85 minutes (08:30 incision, completion expected ~09:55)
- Procedure: Robotic-assisted anterior mediastinal mass resection

Minute-by-minute (08:15-08:59, prep and early surgery):
- 08:15 - Patient arrives Surgical Suite 2. Identity verified. Timeout called.
- 08:16 - Anesthesia induction initiated. IV access confirmed bilateral.
- 08:18 - General anesthesia achieved. Endotracheal intubation complete.
- 08:20 - Patient positioned supine with arms tucked. Surgical field prepped.
- 08:22 - Sterile draping complete. SURG-02 arm positioned at bedside.
- 08:24 - Instrument calibration verified. Camera white balance set.
- 08:26 - Pre-incision safety check complete. Surgical plan confirmed on imaging.
- 08:28 - Trocar sites marked. Local anesthetic infiltrated.
- 08:30 - Incision. First 12 mm port placed at right anterior axillary line.
- 08:32 - Three additional ports placed. CO2 insufflation to 10 mmHg.
- 08:34 - Camera inserted. Mediastinal anatomy visualized. Mass confirmed.
- 08:36 - SURG-02 begins anterior mediastinal dissection. Grasper and cautery.
- 08:38 - Thymic tissue mobilized. Right phrenic nerve identified, preserved.
- 08:40 - Superior pole dissection. Innominate vein identified, retracted.
- 08:42 - Tumor capsule exposed. 4.8 cm mass consistent with pre-op imaging.
- 08:44 - Dissection continues along left lateral border. Left phrenic noted.
- 08:46 - Inferior pole mobilized. Pericardial fat pad separated.
- 08:48 - Major feeding vessels identified. Clips applied x3.
- 08:50 - Vessel division. Hemostasis confirmed. Force: 4-8 N.
- 08:52 - Continued circumferential dissection. 70% tumor mobilized.
- 08:54 - Right lateral attachments divided. Specimen increasingly mobile.
- 08:56 - Posterior dissection initiated. Aortic arch visualized.
- 08:58 - Careful dissection along great vessels. AI trajectory guidance active.
- 08:59 - Hour ends. Surgery continues into Hour 09. EBL: 90 mL. Stable.

## Procedure 2: PAT-ODMND-0033 - Cobot Biopsy

- Patient: PAT-ODMND-0033 (38F, forearm sarcoma, Grade II, ECOG 0)
- Robot: COBOT-01 (Cobot, Instance 1)
- Station: Biopsy Station 1
- Duration: 20 minutes (08:20-08:40)
- Procedure: Ultrasound-guided core needle biopsy of right forearm mass

Minute-by-minute:
- 08:20 - Patient seated, right forearm on biopsy table. Ultrasound confirms mass.
- 08:21 - Mass measured: 3.2 cm x 2.1 cm. Target coordinates locked.
- 08:22 - Skin prepped with chlorhexidine. Sterile drape applied.
- 08:23 - Local anesthetic: 1% lidocaine, 5 mL, subcutaneous and deep track.
- 08:24 - Anesthetic effect confirmed. COBOT-01 arm positioned.
- 08:25 - 14-gauge core needle introduced. Cobot guiding trajectory.
- 08:26 - First core obtained. Length 18 mm. Adequate tissue.
- 08:27 - Needle redirected 3 mm medially. Second core obtained. 16 mm.
- 08:28 - Third core obtained from deep margin. 15 mm. Sampling complete.
- 08:29 - Needle removed. Manual pressure applied to puncture site.
- 08:30 - Hemostasis achieved. Adhesive bandage applied.
- 08:31 - Specimens placed in formalin. Labeled and logged.
- 08:32 - Patient resting. No numbness or weakness in hand.
- 08:35 - 5-minute observation. Wound dry. No hematoma.
- 08:40 - Procedure complete. Patient to observation area.

Outcomes:
- Tissue adequacy: 3 cores, all Grade A (diagnostic quality)
- Cobot positioning accuracy: 1.1 mm deviation (within 2 mm specification)
- Force applied: 2.4 N peak (within 1-5 N range for cobot biopsy)
- Complications: None
- Specimens sent to histopathology per ICH E6(R3) Section 4.2.1

## Procedure 3: PAT-ODMND-0034 - RT Positioning

- Patient: PAT-ODMND-0034 (70M, glioblastoma, Stage IV, ECOG 2)
- Robot: RTPOS-01 (RT Positioning Robot, Instance 1)
- Vault: RT Vault 1
- Duration: 30 minutes (08:25-08:55)
- Procedure: Radiotherapy mask fitting and positioning verification

Minute-by-minute:
- 08:25 - Patient escorted to RT Vault 1. Identity verified.
- 08:27 - Patient positioned supine on treatment couch. Head in neutral.
- 08:29 - Thermoplastic mask heated and formed to patient face/head.
- 08:31 - Mask cooling. Patient comfort assessed. Mild claustrophobia noted.
- 08:33 - Mask solidified. Fit verified. No pressure points detected.
- 08:35 - RTPOS-01 positions couch to isocenter coordinates.
- 08:37 - CT simulation scan acquired. 1 mm slices through brain.
- 08:39 - Scan complete. Tumor bed and residual disease contoured.
- 08:41 - Critical structures delineated: optic chiasm, brainstem, cochleae.
- 08:43 - RTPOS-01 verifies reproducibility. Couch returned to setup, re-aligned.
- 08:45 - Reproducibility within 1.0 mm on all axes. Acceptable.
- 08:47 - Alignment tattoos placed (3 reference points).
- 08:49 - Final verification scan. Position confirmed within tolerance.
- 08:51 - Mask removed. Patient assisted to seated position.
- 08:53 - Post-session assessment. No skin irritation. No dizziness.
- 08:55 - Session complete. CT data uploaded for treatment planning.

Outcomes:
- Positioning accuracy: 1.0 mm reproducibility (within 2 mm specification)
- Mask fit score: 8.2/10
- CT image quality: diagnostic quality, no motion artifact
- Patient tolerance: good (mild claustrophobia managed with coaching)
- Digital twin synchronized with RT planning data per USL framework

## Procedure 4: PAT-ODMND-0035 - Companion Robot Session

- Patient: PAT-ODMND-0035 (5M, pediatric ALL, ECOG 1)
- Robot: COMPN-04 (Companion Robot, Instance 4)
- Area: Companion Play Area 4
- Duration: 47 minutes (08:12-08:59, ongoing into Hour 09)
- Procedure: Pre-chemotherapy anxiety reduction and distraction therapy
- Dexamethasone pre-chemo dose administered at 08:15

Session summary:
- 08:12 - Patient arrives with parent. COMPN-04 initiates greeting protocol.
- 08:14 - Interactive play begins. Building block activity selected.
- 08:15 - Dexamethasone 4 mg oral administered by nursing staff.
- 08:18 - Patient engagement score: 7.5/10. Anxiety level: moderate (4/10).
- 08:25 - Transition to tablet-based game. COMPN-04 narrates story.
- 08:30 - Anxiety level: mild (2/10). Patient laughing and engaged.
- 08:40 - Treatment familiarization module begins. COMPN-04 explains chemo.
- 08:45 - Patient asks questions about "the medicine." Age-appropriate answers.
- 08:50 - Anxiety level: mild (2/10). Patient ready for transition.
- 08:59 - Session continues. Patient will transition to chemo in Hour 09.

Outcomes:
- Anxiety reduction: from 4/10 to 2/10 (50% reduction)
- Patient engagement: sustained above 7.0/10 for 47 minutes
- Parental satisfaction (real-time survey): 8/10
- Pediatric assent maintained throughout per 21 CFR Part 50 Subpart D

## Procedure 5: PAT-ODMND-0036 - RT Motion Tracking

- Patient: PAT-ODMND-0036 (62F, NSCLC squamous, Stage IIIB, ECOG 1)
- Robot: TRACK-02 (RT Motion-Tracking Robot, Instance 2)
- Vault: RT Vault 2
- Duration: 20 minutes (08:35-08:55)
- Procedure: Real-time respiratory-gated radiotherapy, fraction 8 of 30

Minute-by-minute:
- 08:35 - Patient positioned supine. Arms above head in wing board.
- 08:37 - TRACK-02 optical markers placed on chest. Breathing pattern acquired.
- 08:39 - Baseline respiratory trace stable. Amplitude 8 mm, rate 14/min.
- 08:40 - CBCT verification scan. Tumor position confirmed within 2 mm.
- 08:42 - Treatment beam on. Gating active. Beam fires during exhale plateau.
- 08:44 - First field complete. 62 MU delivered. Duty cycle 38%.
- 08:46 - Couch rotation to 270 degrees. Second field initiated.
- 08:48 - Second field complete. 78 MU. Tracking deviation max 1.4 mm.
- 08:50 - Third field initiated. Patient breathing stable.
- 08:52 - Third field complete. 55 MU. Total dose this fraction: 2.0 Gy.
- 08:53 - TRACK-02 confirms delivery within 2% of planned dose.
- 08:55 - Treatment complete. Patient assisted off couch.

Outcomes:
- Tracking accuracy: 1.4 mm maximum deviation (within 3 mm specification)
- Dose delivery accuracy: within 2% of plan
- Respiratory gating duty cycle: 38% (adequate for squamous NSCLC)
- Treatment interruptions: 0
- Fraction 8 of 30 complete per ICH E6(R3) treatment protocol

## Procedure 6: PAT-ODMND-0037 - Needle Placement (with Adverse Event)

- Patient: PAT-ODMND-0037 (49M, parotid tumor, Stage II, ECOG 0)
- Robot: NEEDLE-01 (Needle-Placement System, Instance 1)
- Suite: CT Suite 1
- Duration: 25 minutes (08:30-08:55)
- Procedure: CT-guided fine needle aspiration of left parotid mass
- ADVERSE EVENT: Grade 1 minor bleeding at puncture site (see hour-08-adverse-events.md)

Minute-by-minute:
- 08:30 - Patient positioned supine, head turned right. CT landmarks placed.
- 08:31 - Planning CT acquired. 2.1 cm left parotid mass confirmed.
- 08:32 - NEEDLE-01 trajectory calculated. Facial nerve proximity mapped.
- 08:33 - Local anesthetic: 2% lidocaine, 3 mL administered.
- 08:34 - Anesthetic effect confirmed. NEEDLE-01 arm positioned.
- 08:35 - 22-gauge needle insertion initiated. AI-guided trajectory active.
- 08:36 - Needle at 15 mm depth. ADVERSE EVENT: Minor bleeding at puncture
  site noted. Blood on skin surface, steady ooze. Rate: approximately
  2 mL/min. NEEDLE-01 pauses advancement. Clinician notified.
- 08:37 - Manual pressure applied to puncture site around needle shaft.
- 08:38 - Bleeding reducing. Pressure maintained.
- 08:39 - Continued pressure. Patient reports no pain increase.
- 08:40 - Bleeding controlled. Decision to continue procedure carefully.
- 08:41 - Hemostasis confirmed. 5 minutes pressure total. Procedure resumes.
- 08:42 - NEEDLE-01 advances needle to target. 1.5 mm from planned position.
- 08:43 - First aspiration pass. Adequate cellular material obtained.
- 08:44 - Second aspiration pass. Redirected 2 mm superiorly.
- 08:45 - Third aspiration pass. Sample adequacy confirmed.
- 08:46 - Needle withdrawn. Firm pressure applied.
- 08:48 - Verification CT: no deep hemorrhage, no facial nerve compromise.
- 08:50 - Hemostasis maintained. Bandage applied with pressure dressing.
- 08:52 - Patient assessed: no facial weakness, symmetric smile.
- 08:55 - Transferred to observation. Extended monitoring ordered (30 min).

Outcomes:
- Needle placement accuracy: 1.5 mm (within 2 mm specification)
- Tissue adequacy: 3 passes, Grade A
- Adverse event: Grade 1, resolved. See hour-08-adverse-events.md.
- Extended observation ordered as precaution per 21 CFR Part 312 Section 312.32

## Procedure 7: PAT-ODMND-0038 - Imaging Assessment (Phase 1)

- Patient: PAT-ODMND-0038 (57F, HCC, Stage III, ECOG 1)
- Robot: IMAGE-03 (Imaging Assistant, Instance 3)
- Bay: Imaging Bay 3
- Sorafenib administered per IND protocol (21 CFR 312.23)
- Duration: 20 minutes (08:40-09:00, extends into Hour 09)
- Procedure: Robotic ultrasound liver assessment for ablation planning

Session summary (08:40-08:59):
- 08:40 - Patient positioned supine. Ultrasound gel applied.
- 08:42 - IMAGE-03 initiates systematic liver survey. Right lobe first.
- 08:44 - Primary HCC lesion identified: segment 6, 4.2 cm x 3.8 cm.
- 08:46 - Satellite lesion: segment 5, 1.4 cm x 1.1 cm.
- 08:48 - Portal vein patency confirmed. No tumor thrombus.
- 08:50 - Hepatic vein anatomy mapped for ablation planning.
- 08:52 - 3D reconstruction in progress. Probe pressure: 1.8 N.
- 08:54 - Left lobe survey. No additional lesions detected.
- 08:56 - Contrast-enhanced ultrasound initiated for vascularity mapping.
- 08:58 - Arterial phase enhancement of primary lesion confirmed.
- 08:59 - Imaging continues into Hour 09. Ablation with STEER-01 to follow.

Outcomes at 08:59 (partial):
- Probe pressure: 1.8 N steady (within 1-3 N range)
- Image quality score: 8.1/10
- Lesion mapping: 2 lesions identified, consistent with prior CT
- Scan coverage at 08:59: 80% (ongoing)

## Procedure 8: PAT-ODMND-0039 - Humanoid Therapy

- Patient: PAT-ODMND-0039 (13F, pediatric osteosarcoma, ECOG 1)
- Robot: HUMAN-03 (Humanoid Robot, Instance 3)
- Station: Humanoid Station 3
- Duration: 25 minutes planned (08:45-09:10, extends into Hour 09)
- Procedure: Humanoid-assisted physical therapy preparation

Session summary (08:45-08:59):
- 08:45 - Patient arrives with parent. HUMAN-03 greets patient by name.
- 08:47 - Exercise demonstration begins. HUMAN-03 models knee flexion.
- 08:49 - Patient follows demonstration. Range of motion assessed: 85 degrees.
- 08:51 - Quad strengthening exercise demonstrated. Patient performs 8 reps.
- 08:53 - Balance exercise demonstrated. Single-leg stand practice.
- 08:55 - HUMAN-03 provides positive reinforcement. Patient confidence high.
- 08:57 - Gait pattern demonstration. Patient walks alongside HUMAN-03.
- 08:59 - Session continues into Hour 09. Progress positive.

Outcomes at 08:59 (partial):
- Knee ROM: 85 degrees (target: 90 degrees)
- Patient engagement: 8.5/10
- Exercises completed: 3 of 5 planned
- Adolescent assent maintained per 21 CFR Part 50 Subpart D

## Procedure 9: PAT-ODMND-0040 - Liver Imaging

- Patient: PAT-ODMND-0040 (66M, liver mets colorectal, Stage IV, ECOG 2)
- Robot: IMAGE-04 (Imaging Assistant, Instance 4)
- Bay: Imaging Bay 4
- Duration: 15 minutes planned (08:52-09:07, extends into Hour 09)
- Procedure: Robotic ultrasound characterization of hepatic metastases

Session summary (08:52-08:59):
- 08:52 - Patient positioned supine. IMAGE-04 calibrated for liver survey.
- 08:54 - Right lobe survey initiated. Multiple echogenic lesions detected.
- 08:56 - Lesion 1: segment 7, 2.8 cm. Lesion 2: segment 6, 1.9 cm.
- 08:58 - Lesion 3: segment 8, 1.2 cm. Probe pressure: 1.6 N.
- 08:59 - Imaging continues into Hour 09. Three of estimated 5+ lesions mapped.

## Procedure 10: PAT-ODMND-0041 - RT Motion Tracking

- Patient: PAT-ODMND-0041 (44F, NSCLC adenocarcinoma, Stage IIB, ECOG 0)
- Robot: TRACK-03 (RT Motion-Tracking Robot, Instance 3)
- Vault: RT Vault 3
- Duration: 18 minutes active (08:50-09:08, extends into Hour 09)
- Note: Patient waited 8 minutes (08:42-08:50) for vault availability
- Procedure: Real-time respiratory-gated radiotherapy, fraction 3 of 25

Session summary (08:50-08:59):
- 08:50 - Patient positioned. TRACK-03 optical markers placed.
- 08:52 - Breathing pattern acquired. Amplitude 6 mm, rate 16/min.
- 08:54 - CBCT acquired. Tumor alignment within 1.5 mm. Acceptable.
- 08:56 - Treatment beam on. First field initiated. Gating active.
- 08:58 - First field complete. 48 MU delivered. Duty cycle 42%.
- 08:59 - Treatment continues into Hour 09.

## Procedure 11: PAT-ODMND-0042 - Rehabilitation Exoskeleton

- Patient: PAT-ODMND-0042 (72M, femur osteosarcoma post-surgical, ECOG 2)
- Robot: REHAB-03 (Rehabilitation Exoskeleton, Instance 3)
- Bay: Rehab Bay 3
- Duration: 25 minutes planned (08:58-09:23, extends into Hour 09)
- Procedure: Exoskeleton-assisted gait training, 8 weeks post-limb-salvage

Session summary (08:58-08:59):
- 08:58 - Patient arrives Rehab Bay 3. REHAB-03 fitted to right lower limb.
- 08:59 - Strap-up in progress. Session continues into Hour 09.

## Concurrent Procedure Timeline

```
08:00    08:10    08:20    08:30    08:40    08:50    08:59
  |        |        |        |        |        |        |
P0024 SURG-01  [=========================================>  (cont)
P0032 SURG-02              [prep=====[========================>  (cont)
P0033 COBOT-01                  [==============]
P0034 RTPOS-01                   [========================]
P0035 COMPN-04    [============================================>  (cont)
P0036 TRACK-02                          [==============]
P0037 NEEDLE-01                    [=======AE==========]
P0038 IMAGE-03                              [==================>  (cont)
P0039 HUMAN-03                                    [============>  (cont)
P0040 IMAGE-04                                       [========>  (cont)
P0041 TRACK-03                               wait[============>  (cont)
P0042 REHAB-03                                            [===>  (cont)
                                                          ^^^^
                                                     Peak: 11
                                                     concurrent
```

## USL and Patient Journey References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) is exercised at unprecedented breadth this
hour as 8 of 10 robot types operate concurrently. Cross-robot data sharing
between SURG-01 and SURG-02 (concurrent mediastinal surgeries) represents
a key USL simulation switching test. IMAGE-03 to STEER-01 handoff for
P0038 exercises USL cross-robot sharing criteria.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) is reflected in procedures spanning
diagnostic biopsy (Stage 2), treatment preparation (Stage 3), active
treatment delivery (Stage 4), and post-surgical rehabilitation (Stage 7).
