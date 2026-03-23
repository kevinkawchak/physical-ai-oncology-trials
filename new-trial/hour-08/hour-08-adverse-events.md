# Hour 08: Adverse Event Report

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Adverse Event Summary

One adverse event occurred during Hour 08. PAT-ODMND-0037 experienced minor
bleeding at the puncture site during CT-guided needle placement of the left
parotid mass. The event was classified as Grade 1 (mild) per CTCAE v5.0
criteria. Hemostasis was achieved with 5 minutes of manual pressure. The
procedure was completed successfully after the bleeding was controlled.
No treatment interruption was required beyond the temporary pause for
hemostasis. No other adverse events occurred during the hour.

## Regulatory Framework References

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368)
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707)
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628)

## Adverse Event AE-0008-001

### Event Identification

| Field | Value |
|-------|-------|
| AE Reference | AE-0008-001 |
| Patient | PAT-ODMND-0037 |
| Age / Sex | 49 / Male |
| Cancer Type | Parotid tumor, Stage II |
| ECOG Status | 0 |
| Procedure | CT-guided fine needle aspiration, left parotid mass |
| Robot | NEEDLE-01 (Needle-Placement System, Instance 1) |
| Location | CT Suite 1 |
| Event Time | 08:36 |
| Detection Method | Visual observation by NEEDLE-01 optical sensor and clinician |

### Event Description

At 08:36, during CT-guided fine needle aspiration of the left parotid mass,
minor bleeding was observed at the skin puncture site when the 22-gauge needle
reached a depth of 15 mm. The bleeding presented as a steady ooze from the
puncture site at an estimated rate of approximately 2 mL/min. The bleeding was
external only, with no evidence of deep hemorrhage on subsequent CT
verification.

NEEDLE-01 optical sensors detected the blood on the skin surface at 08:36:12
and triggered an automatic pause of needle advancement. The system generated
an alert to the supervising clinician. The clinician applied manual pressure
around the needle shaft while the needle remained in position. Bleeding
decreased progressively over 5 minutes. At 08:41, hemostasis was confirmed
and the procedure resumed.

### CTCAE Classification

| Field | Value |
|-------|-------|
| CTCAE Version | 5.0 |
| Category | Injury, poisoning, and procedural complications |
| Term | Puncture site hemorrhage |
| Grade | 1 (Mild) |
| Definition | Mild symptoms; intervention not indicated or minimal intervention |
| Seriousness | Non-serious |
| Expectedness | Expected (listed in protocol risk disclosure) |
| Causality - Robot | Possible (needle insertion is robot-guided) |
| Causality - Disease | Possible (parotid region is vascular) |
| Causality - Drug | Not applicable (no IND drug for this patient) |

### Timeline

```
EVENT TIMELINE FOR AE-0008-001

08:30  Procedure begins. Patient positioned. CT landmarks placed.
  |
08:33  Local anesthetic administered (2% lidocaine, 3 mL).
  |
08:35  Needle insertion initiated. NEEDLE-01 guiding 22-gauge needle.
  |
08:36  NEEDLE AT 15 mm DEPTH
  |    *** ADVERSE EVENT: Minor bleeding at puncture site ***
  |    - NEEDLE-01 optical sensor detects blood on skin (08:36:12)
  |    - Automatic needle advancement pause triggered (08:36:13)
  |    - Clinician alert generated (08:36:14)
  |    - Estimated bleed rate: ~2 mL/min (external ooze)
  |
08:37  Manual pressure applied around needle shaft.
  |    - Patient reports no increase in pain.
  |    - Vital signs stable: BP 128/78, HR 72, SpO2 99%.
  |
08:38  Bleeding decreasing. Continued pressure.
  |
08:39  Bleeding further reduced. Continued pressure.
  |
08:40  Bleeding controlled. Decision to resume procedure.
  |
08:41  HEMOSTASIS CONFIRMED after 5 minutes total pressure.
  |    *** Adverse event resolved ***
  |    - Estimated total blood loss: 5-8 mL
  |    - Procedure resumes with clinician approval.
  |
08:42  Needle advanced to target. 1.5 mm from planned position.
  |
08:45  Aspiration complete. 3 passes. Sample adequate.
  |
08:46  Needle withdrawn. Firm pressure applied.
  |
08:48  Verification CT: no deep hemorrhage, no facial nerve issue.
  |
08:50  Pressure dressing applied. Patient to extended observation.
  |
08:55  Patient transferred to observation area (30 min monitoring).
```

### Robot Performance During Adverse Event

NEEDLE-01 performance during the adverse event was consistent with design
specifications and safety protocols:

- Optical sensor detected blood on skin surface within 1 second of appearance
- Automatic pause triggered within 0.1 seconds of detection
- Needle position held stable during 5-minute pressure period (drift: 0.0 mm)
- No unintended needle movement during manual pressure application
- System correctly flagged event for AE documentation per ICH E6(R3) Section 2.10
- All sensor data logged at 100 Hz throughout event for audit trail

### Contributing Factors Analysis

1. Anatomical: The parotid region is highly vascular with branches of the
   external carotid artery (transverse facial, superficial temporal). Minor
   vessel puncture during skin entry is a known procedural risk.

2. Technical: NEEDLE-01 trajectory was within specification (1.5 mm of planned
   path). The bleeding originated at the superficial puncture site, not from
   deep structures. No trajectory error identified.

3. Patient: No anticoagulant use. INR 1.0 (normal). Platelet count 245,000
   (normal). No bleeding diathesis history. Contributing factor: possible
   small superficial vessel at needle entry point.

### Outcome and Follow-Up

- Outcome: Resolved
- Resolution time: 5 minutes (08:36 to 08:41)
- Residual effects: None
- Extended observation: 30 minutes post-procedure ordered
- Post-observation check at 09:25: No recurrent bleeding, wound dry
- Patient disposition: Cleared for discharge after observation period
- Follow-up: Routine post-biopsy follow-up in 1 week per standard protocol

### Regulatory Reporting

Per 21 CFR Part 312 Section 312.32 (IND Safety Reporting), this Grade 1
non-serious expected adverse event does not meet criteria for expedited
reporting to the FDA or IRB. The event is documented in the trial database
and will be included in:

- Annual IND safety report per 21 CFR Part 312 Section 312.33
- Trial progress monitoring per ICH E6(R3) Section 2.10
- Robot performance database per PSL Dimension A tracking
- Patient safety record per 21 CFR Part 50 Section 50.25

Per ICH E6(R3) Section 2.9.1, complete audit trail maintained including
NEEDLE-01 sensor logs, CT images, clinician notes, vital sign monitoring,
and resolution documentation.

### Corrective and Preventive Actions

No corrective actions required for a Grade 1 expected event. Preventive
measures already in protocol include:

- Pre-procedure coagulation screening (completed, results normal)
- NEEDLE-01 optical bleeding detection system (functioned as designed)
- Automatic pause-on-detection protocol (triggered appropriately)
- Manual pressure kit at bedside (available and used)
- Extended observation protocol for any intra-procedural bleeding (activated)

### Impact on Site PSL

This adverse event has no negative impact on NEEDLE-01 PSL scores. The
system detected the bleeding promptly (Dimension A - omniscient awareness),
paused automatically (Dimension C - procedural capability), and maintained
availability for event management (Dimension B - omnipresent monitoring).
The event demonstrates expected real-world procedural variation within
acceptable safety parameters.

## Adverse Event Statistics - Cumulative Trial Summary

| Hour | Patient | Event | Grade | Robot | Outcome |
|------|---------|-------|-------|-------|---------|
| 08 | P0037 | Puncture site hemorrhage | 1 | NEEDLE-01 | Resolved |

Total adverse events through Hour 08: 1
Grade 1 events: 1
Grade 2+ events: 0
Serious adverse events: 0
Robot-related events: 0 confirmed (1 possible)

## USL and Patient Journey References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) evaluates NEEDLE-01 interoperability including
safety event data sharing. During AE-0008-001, NEEDLE-01 transmitted real-time
event data to the site safety monitoring system, demonstrating USL cross-
system communication capability during adverse conditions.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) addresses adverse event management within the
individual patient journey. PAT-ODMND-0037's experience at Stage 2 (Diagnostic
Workup) illustrates how procedural complications are managed within the on-
demand multi-patient Physical AI trial context without impacting the patient's
overall journey progression.
