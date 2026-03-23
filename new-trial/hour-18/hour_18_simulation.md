# Hour 18: 18:00-18:59 - Evening Peak Operations

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Hour Summary

Hour 18 marks the evening peak period with 9 new patient arrivals, the
highest single-hour intake of the simulation. Post-work patients P0145 and
P0146 arrive first, taking advantage of evening scheduling for minimal
disruption to employment. Pediatric patients P0148 and P0153 receive
multidisciplinary support including humanoid-assisted rehabilitation and
companion robot engagement. P0134 surgery (started approximately 17:15)
remains ongoing in Surgical Suite 2 with SURG-02. Dual-vault radiotherapy
operations proceed with TRACK-02 and TRACK-03 serving P0145 and P0151
respectively. No adverse events occur. Site PSL advances to 65.5.

## Site Status at 18:00

- Total patients on-site: approximately 22 (concurrent)
- Active procedures: 6 (P0134 surgery ongoing, plus 5 new procedures initiating)
- Robots in active mode: 16 (approximately 55% utilization)
- Robots in standby mode: 13
- Robots in maintenance: 0
- Queue length: 0 across all stations (evening staffing adequate)
- Site safety officer on duty: SSO-E2 (evening shift)

## New Patient Arrivals

| Patient ID | Time | Age | Sex | Cancer Type | Stage | ECOG | Robot Assigned |
|-----------|------|-----|-----|-------------|-------|------|----------------|
| PAT-ODMND-0145 | 18:05 | 60 | M | NSCLC adenocarcinoma | IIB | 1 | RT Motion-Tracking (TRACK-02) |
| PAT-ODMND-0146 | 18:10 | 39 | F | Forearm sarcoma | II | 0 | Cobot (COBOT-01) |
| PAT-ODMND-0147 | 18:16 | 72 | M | Meningioma | I | 0 | RT Positioning (RTPOS-03) |
| PAT-ODMND-0148 | 18:22 | 10 | M | Pediatric osteosarcoma | - | 1 | Humanoid (HUMAN-03), Rehab (REHAB-03) |
| PAT-ODMND-0149 | 18:28 | 55 | F | Parotid tumor | II | 0 | Needle-Placement (NEEDLE-02) |
| PAT-ODMND-0150 | 18:34 | 67 | M | HCC | III | 1 | Imaging Assistant (IMAGE-04) |
| PAT-ODMND-0151 | 18:40 | 44 | F | NSCLC squamous | IIIB | 1 | RT Motion-Tracking (TRACK-03) |
| PAT-ODMND-0152 | 18:46 | 73 | M | Liver metastases | IV | 2 | Steerable Needle (STEER-01) |
| PAT-ODMND-0153 | 18:52 | 16 | F | Pediatric ALL | - | 1 | Social Companion (COMPN-01) |

Patient PAT-ODMND-0145 is a 60-year-old male with Stage IIB NSCLC
adenocarcinoma presenting for evening radiotherapy with real-time motion
tracking. He is a post-work patient who selected the 18:00 window to avoid
absence from employment. ECOG performance status 1. Procedure plan:
TRACK-02-guided volumetric modulated arc therapy (VMAT) with respiratory
gating in Vault 2.

Patient PAT-ODMND-0146 is a 39-year-old female with Stage II forearm sarcoma
presenting for cobot-assisted tissue sampling. She is a post-work patient
arriving directly from her workplace. ECOG performance status 0. Procedure
plan: COBOT-01-guided core needle biopsy of left forearm mass under local
anesthesia.

Patient PAT-ODMND-0147 is a 72-year-old male with Grade I meningioma
presenting for stereotactic radiotherapy positioning. ECOG performance
status 0. Procedure plan: RTPOS-03-assisted custom mask fitting and CT
simulation for subsequent stereotactic radiosurgery. Per standard meningioma
protocol, no investigational drug administration.

Patient PAT-ODMND-0148 is a 10-year-old male with pediatric osteosarcoma
presenting for humanoid-assisted rehabilitation and exoskeleton gait
training. ECOG performance status 1. Procedure plan: HUMAN-03 provides
motivational coaching and movement demonstration, followed by REHAB-03
exoskeleton-assisted walking. Parent/guardian consent obtained per 21 CFR
Part 50 Subpart D pediatric protections. Assent obtained from patient.

Patient PAT-ODMND-0149 is a 55-year-old female with Stage II parotid tumor
presenting for CT-guided needle biopsy. ECOG performance status 0. Procedure
plan: NEEDLE-02-guided fine needle aspiration of right parotid mass.

Patient PAT-ODMND-0150 is a 67-year-old male with Stage III hepatocellular
carcinoma presenting for robotic imaging assessment. ECOG performance
status 1. Procedure plan: IMAGE-04-guided ultrasound liver assessment for
treatment response monitoring and digital twin update.

Patient PAT-ODMND-0151 is a 44-year-old female with Stage IIIB NSCLC
squamous cell carcinoma presenting for evening radiotherapy with motion
tracking. ECOG performance status 1. Procedure plan: TRACK-03-guided VMAT
with respiratory gating in Vault 3. Dual-vault RT operations with P0145
in Vault 2.

Patient PAT-ODMND-0152 is a 73-year-old male with Stage IV liver metastases
(colorectal primary) presenting for steerable needle ablation. ECOG
performance status 2. Procedure plan: STEER-01-guided radiofrequency
ablation of two hepatic metastases under CT guidance. Cabozantinib
administered per IND protocol (see Investigational Drug Administrations).

Patient PAT-ODMND-0153 is a 16-year-old female with pediatric ALL presenting
for companion robot emotional support session before evening chemotherapy
preparation. ECOG performance status 1. Procedure plan: COMPN-01 interactive
session for anxiety reduction. Parental consent and patient assent obtained
per 21 CFR Part 50 Subpart D.

## Continuing Patients at 18:00

| Patient ID | Age | Sex | Cancer Type | Status | Since |
|-----------|-----|-----|-------------|--------|-------|
| PAT-ODMND-0134 | - | - | - | Surgery ongoing (SURG-02) | ~17:15 |
| Plus approximately 12 additional patients in various stages of treatment, recovery, and observation |

Note: Full continuing patient roster maintained in hour_18_patient_records.md.
P0134 surgical procedure began approximately 17:15 and continues through
this hour with estimated completion at 19:30.

## Active Procedures This Hour

### P0134 Surgery (Continuing from Hour 17)
- Patient: PAT-ODMND-0134
- Robot: SURG-02 (Surgical Robot, Instance 2)
- Bay: Surgical Suite 2
- Status: Surgery ongoing, estimated completion 19:30
- Procedure minutes this hour: 60 (full hour)
- No complications reported. Telemetry nominal.

### RT Motion-Tracking Session - P0145 (18:15-18:45)
- Patient: PAT-ODMND-0145
- Robot: TRACK-02 (RT Motion-Tracking, Instance 2)
- Bay: Vault 2
- Procedure: VMAT with respiratory gating for NSCLC adenocarcinoma
- Duration: 30 minutes (positioning 5 min, imaging 3 min, treatment 18 min,
  verification 4 min)
- Respiratory tracking accuracy: 0.8 mm (within 1.5 mm specification)
- Beam-on time: 18 minutes (planned 18 min)
- Gating efficiency: 94% (beam paused during irregular breathing cycles)
- Fraction dose delivered: 2.0 Gy (prescribed 2.0 Gy)
- Treatment interruptions: 0
- Outcome: Successful completion. Fraction 12 of 30.

Minute-by-minute summary (active procedure):
- 18:15 - Patient positioned supine. TRACK-02 optical markers placed on chest.
- 18:16 - Immobilization devices applied. Position verified against reference.
- 18:17 - CBCT acquired. Registration to planning CT: 0.4 mm shift applied.
- 18:18 - CBCT verification complete. Treatment plan loaded.
- 18:19 - Respiratory baseline established. Gating window set.
- 18:20 - Beam-on initiated. Arc 1 of 3.
- 18:24 - Arc 1 complete. 0.67 Gy delivered.
- 18:25 - Arc 2 initiated. Gating efficiency 95%.
- 18:30 - Arc 2 complete. Cumulative 1.34 Gy delivered.
- 18:31 - Arc 3 initiated. Patient breathing stable.
- 18:36 - Irregular breathing detected. Beam paused 4 seconds. Resumed.
- 18:38 - Arc 3 complete. Cumulative 2.0 Gy delivered. Fraction complete.
- 18:39 - Verification CBCT acquired. Post-treatment anatomy confirmed.
- 18:41 - Immobilization devices removed. Patient assisted off couch.
- 18:45 - Patient transferred to observation area. No acute toxicity noted.

### Cobot-Assisted Biopsy - P0146 (18:18-18:35)
- Patient: PAT-ODMND-0146
- Robot: COBOT-01 (Cobot, Instance 1)
- Bay: Biopsy Station 1
- Procedure: Core needle biopsy of left forearm sarcoma
- Duration: 17 minutes (positioning 2 min, ultrasound localization 3 min,
  local anesthetic 2 min, biopsy 5 min, hemostasis 3 min, dressing 2 min)
- Needle placement accuracy: 1.1 mm (within 2 mm specification)
- Tissue cores obtained: 4 (adequate for histopathology and molecular testing)
- Bleeding: Minimal, controlled with direct pressure
- Treatment interruptions: 0
- Outcome: Successful completion. Specimens sent to pathology.

### RT Positioning Session - P0147 (18:25-18:55)
- Patient: PAT-ODMND-0147
- Robot: RTPOS-03 (RT Positioning, Instance 3)
- Bay: Vault 3 (simulation mode, no beam)
- Procedure: Custom thermoplastic mask fitting and CT simulation for
  stereotactic radiosurgery planning
- Duration: 30 minutes (mask fabrication 10 min, CT simulation 8 min,
  registration 5 min, verification 7 min)
- Mask fit tolerance: 1.0 mm (within 1.5 mm specification)
- CT simulation coverage: Complete cranial volume acquired
- Fiducial registration accuracy: 0.3 mm (within 0.5 mm specification)
- Treatment interruptions: 0
- Outcome: Successful. Mask stored. CT dataset sent to planning system.
- Note: Per standard meningioma protocol. No investigational drug.

### Humanoid and Rehab Session - P0148 (18:30-18:58)
- Patient: PAT-ODMND-0148
- Robot: HUMAN-03 (Humanoid, Instance 3) 18:30-18:40
- Robot: REHAB-03 (Rehabilitation Exoskeleton, Instance 3) 18:42-18:58
- Bay: Pediatric Therapy 3 then Rehab Bay 3
- Procedure: Motivational coaching followed by exoskeleton gait training
- HUMAN-03 session (10 min): Movement demonstrations, exercise coaching,
  verbal encouragement. Patient engagement score: 8.2/10.
- REHAB-03 session (16 min): Strap-up 3 min, walking 11 min, removal 2 min.
  Walking speed 0.25 m/s. Distance 165 meters. Gait symmetry 0.68.
  Weight-bearing compliance 78% on affected limb (target 75-100%).
  Pain reported: 3/10 during walking, 1/10 at rest.
- Treatment interruptions: 0
- Outcome: Successful. Patient tolerated both sessions well.
- Pediatric protections: Parent present throughout per 21 CFR Part 50 Subpart D.

### Needle Biopsy - P0149 (18:36-18:55)
- Patient: PAT-ODMND-0149
- Robot: NEEDLE-02 (Needle-Placement, Instance 2)
- Bay: CT Suite 2
- Procedure: CT-guided fine needle aspiration of right parotid mass
- Duration: 19 minutes (positioning 3 min, CT scan 2 min, local anesthetic
  2 min, needle placement 3 min, aspiration 4 min, verification CT 2 min,
  needle removal 1 min, hemostasis 2 min)
- Needle placement accuracy: 1.0 mm (within 2 mm specification)
- Aspiration passes: 3 (adequate for cytopathology)
- Tissue sample quality: Grade A
- Bleeding: Minimal
- Treatment interruptions: 0
- Outcome: Successful completion. Sample sent to pathology.

### Imaging Assessment - P0150 (18:42-18:57)
- Patient: PAT-ODMND-0150
- Robot: IMAGE-04 (Imaging Assistant, Instance 4)
- Bay: Imaging Bay 4
- Procedure: Robotic ultrasound liver assessment for HCC
- Duration: 15 minutes
- Probe pressure: 1.6 N steady (within 1-3 N range)
- Image quality score: 8.1/10
- Tumor measurements: Primary HCC lesion 48 mm x 36 mm (stable from prior)
- Scan coverage: 96%
- Motion artifact count: 0
- Outcome: Successful. Images uploaded to DICOM server.

### RT Motion-Tracking Session - P0151 (18:48-18:59+)
- Patient: PAT-ODMND-0151
- Robot: TRACK-03 (RT Motion-Tracking, Instance 3)
- Bay: Vault 3 (P0147 completed, vault transitioned)
- Procedure: VMAT with respiratory gating for NSCLC squamous
- Duration: 30 minutes estimated (extends into Hour 19)
- Status at 18:59: Arc 2 of 3 in progress
- Respiratory tracking accuracy: 0.9 mm (within 1.5 mm specification)
- Partial dose delivered this hour: 1.2 Gy of 2.0 Gy planned
- Treatment interruptions: 0
- Note: Dual-vault evening RT operations with P0145 in Vault 2 (completed)
  and P0151 in Vault 3 (continuing).

### Steerable Needle Ablation - P0152 (18:54-18:59+)
- Patient: PAT-ODMND-0152
- Robot: STEER-01 (Steerable Needle, Instance 1)
- Bay: Ablation Suite 1
- Procedure: CT-guided radiofrequency ablation of hepatic metastases
- Duration: 45 minutes estimated (extends into Hour 19)
- Status at 18:59: Initial CT mapping complete, first needle insertion in progress
- Cabozantinib 40 mg administered orally at 18:50 per IND protocol
- Treatment interruptions: 0
- Note: Extended procedure continues into Hour 19.

### Companion Robot Session - P0153 (18:56-18:59+)
- Patient: PAT-ODMND-0153
- Robot: COMPN-01 (Social Companion, Instance 1)
- Bay: Pediatric Play Room 1
- Procedure: Interactive emotional support session before chemotherapy prep
- Status at 18:59: Session in progress. Initial anxiety score 6/10,
  current score 4/10 (decreasing). Engagement level: high.
- Treatment interruptions: 0
- Note: Session continues into Hour 19.

## Patient Departures This Hour

| Patient ID | Time | Outcome | Notes |
|-----------|------|---------|-------|
| PAT-ODMND-0145 | 18:55 | Completed RT, observation | Post-RT monitoring 15 min then discharge |
| PAT-ODMND-0146 | 18:50 | Post-biopsy observation | Awaiting hemostasis confirmation |

Note: No full discharges from the facility this hour. Patients completing
procedures transition to observation areas.

## Adverse Events

None this hour.

## Investigational Drug Administrations

### PAT-ODMND-0152 - Cabozantinib (IND)
- Drug: Cabozantinib 40 mg tablet, oral
- IND protocol: Per 21 CFR Part 312, IND application on file
- Administration time: 18:50
- Indication: Stage IV colorectal liver metastases, in combination with
  radiofrequency ablation per protocol-specified combination therapy arm
- Lot number: CAB-2026-0412
- Dispensed by: Site pharmacist (PharmD, verified against IND drug
  accountability log)
- Pre-dose labs: Hepatic function within protocol-specified thresholds
  (AST 42 U/L, ALT 38 U/L, total bilirubin 1.1 mg/dL)
- Adverse event monitoring: 4-hour post-dose observation per protocol
- Documentation: IND drug accountability form completed per 21 CFR 312.62

### PAT-ODMND-0147 - Standard Meningioma Protocol
- No investigational drug administered. CT simulation and mask fitting
  conducted per standard meningioma protocol. Procedure is non-drug,
  positioning and planning only.

## Evening Wave Observations

The 18:00-18:59 period represents the evening peak with 9 arrivals, driven
by post-work patients and families scheduling after school and employment
hours. P0145 and P0146 are explicitly post-work patients who selected
evening slots to minimize occupational disruption. P0148 and P0153
(pediatric) arrive with families after school hours.

Dual-vault radiotherapy operations (Vault 2 for P0145, Vault 3 for P0151)
demonstrate the site's capacity to deliver concurrent motion-tracked RT
sessions during high-demand periods. This operational mode supports the
PSL Dimension C increase for RT Motion-Tracking.

## Site Utilization

- Overall robot utilization: approximately 55% (16 of 29 robots active at peak)
- Queue lengths: 0 across all stations
- Average wait time: 6 minutes (range 4-8 min across 9 arrivals)
- Robot cleaning cycles: 4 (TRACK-02, COBOT-01, NEEDLE-02, IMAGE-04 post-procedure)
- Concurrent patients on-site: approximately 22

## End-of-Hour Census

| Patient ID | Age | Sex | Cancer Type | Status | Location |
|-----------|-----|-----|-------------|--------|----------|
| PAT-ODMND-0134 | - | - | - | Surgery ongoing | Surgical Suite 2 |
| PAT-ODMND-0145 | 60 | M | NSCLC adenocarcinoma | Post-RT observation | Recovery Bay |
| PAT-ODMND-0146 | 39 | F | Forearm sarcoma | Post-biopsy observation | Recovery Bay |
| PAT-ODMND-0147 | 72 | M | Meningioma | Post-simulation observation | Recovery Bay |
| PAT-ODMND-0148 | 10 | M | Pediatric osteosarcoma | Post-rehab observation | Pediatric Ward |
| PAT-ODMND-0149 | 55 | F | Parotid tumor | Post-biopsy observation | CT Suite 2 area |
| PAT-ODMND-0150 | 67 | M | HCC | Post-imaging observation | Imaging Bay 4 |
| PAT-ODMND-0151 | 44 | F | NSCLC squamous | Active RT (continuing) | Vault 3 |
| PAT-ODMND-0152 | 73 | M | Liver metastases | Active ablation (continuing) | Ablation Suite 1 |
| PAT-ODMND-0153 | 16 | F | Pediatric ALL | Active companion (continuing) | Pediatric Play 1 |
| Plus approximately 12 continuing patients from prior hours in recovery/observation |

Total patients on-site at 18:59: approximately 22

## Regulatory Compliance Notes

### ICH E6(R3) - Adaption
- Section 1.1.1: All procedures conducted in accordance with ethical principles
  and applicable GCP requirements. Evening operations maintained identical
  safety standards to daytime operations. Post-work patient scheduling
  accommodated per on-demand protocol design.
- Section 2.9.1: Complete audit trails maintained for all 9 procedures
  initiated this hour, including robot telemetry, imaging data, tissue
  chain-of-custody documentation, and drug accountability records.
- Section 4.2.1: Data capture across all active robots included synchronized
  UTC timestamps, sensor fusion records, and procedure outcome metrics.
  Dual-vault RT operations captured independent gating logs for concurrent
  patient safety verification.

### 21 CFR Part 50 - Adaption
- Section 50.25: All nine new patients completed informed consent including
  Physical AI system disclosure, USL readiness scores, and right to
  non-Physical AI alternatives.
- Section 50.30: Pre-procedure safety matrix completed for all procedures.
- Subpart D: Pediatric patients P0148 (10M) and P0153 (16F) treated under
  additional protections. Parental consent and patient assent documented.
  Independent pediatric advocate available. IRB-approved pediatric protocol
  followed for both humanoid interaction and companion robot engagement.

### 21 CFR Part 312 - Adaption
- Section 312.32: Safety reporting systems active. No reportable events.
- Section 312.62: Investigator recordkeeping maintained for all patients.
  IND drug accountability log updated for cabozantinib dispensed to P0152.
  Drug storage temperature verification documented.
- Section 312.50: Sponsor obligations met for IND compound monitoring.
  P0152 cabozantinib administration documented with lot number, dose,
  route, time, and pre-dose laboratory values.

## Complementary Framework References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary robot technical
interoperability scoring. TRACK-02 and TRACK-03 dual-vault operations
demonstrate platform-level capabilities evaluated at USL scores consistent
with the Advanced band, reflecting strong real-time sensor fusion for
concurrent respiratory-gated radiotherapy delivery.
See physical-ai-oncology-trials/unification/usl/paper/usl_oncology_trials.tex.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) demonstrated autonomous Physical AI trial
orchestration for an individual patient. P0148's sequential humanoid
coaching followed by exoskeleton rehabilitation represents a multi-robot
coordinated care pathway within the on-demand multi-patient context,
extending the journey framework's single-patient model to concurrent
evening peak operations.
See physical-ai-oncology-trials/patient-journey/paper/patient_journey_paper.tex.
