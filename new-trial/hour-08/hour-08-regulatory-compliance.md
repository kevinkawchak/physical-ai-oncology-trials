# Hour 08: Regulatory Compliance Report

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Compliance Summary

Hour 08 presents the most complex regulatory compliance landscape of the
trial to date, driven by 12 new patient arrivals, 3 investigational drug
administrations, 1 adverse event, the first patient queue, 2 concurrent
surgical cases, and 2 pediatric patients requiring Subpart D protections.
All regulatory requirements under the three adapted frameworks were met.
No protocol deviations occurred. The adverse event (AE-0008-001) was managed
within established reporting guidelines. Investigational drug administrations
for PAT-ODMND-0032 (atezolizumab), PAT-ODMND-0035 (dexamethasone), and
PAT-ODMND-0038 (sorafenib) were documented per IND requirements.

## Regulatory Framework References

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368) - Good Clinical Practice
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707) - Human Subject Protection
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628) - Investigational New Drugs

## ICH E6(R3) Adaption Compliance

### Section 1.1.1 - Ethical Principles and GCP

All 12 new patient arrivals processed in accordance with ethical principles
and applicable GCP requirements adapted for Physical AI operations. Peak
morning operations maintained identical safety standards to overnight and
early morning periods. The transition to high-volume operations did not
result in any compromise of ethical oversight or GCP adherence.

Key compliance points:
- All patients received care consistent with their informed consent
- No patient was pressured to accept Physical AI intervention
- Alternative non-Physical AI pathways remained available for all procedures
- Pediatric patients (P0035, P0039) received age-appropriate protections

### Section 2.9.1 - Audit Trail

Complete audit trails maintained for all 12 procedures initiated this hour.
Audit trail scope includes:

```
AUDIT TRAIL COVERAGE - HOUR 08

Data Type                     Capture Rate   Storage
---------------------------  -------------  ---------
Robot sensor telemetry        100 Hz         Local + cloud
Patient vital signs            1 Hz          EHR + trial DB
Procedure timestamps          UTC sync       Trial DB
Drug administration records   Per event      Pharmacy + trial DB
Adverse event documentation   Per event      Safety DB + trial DB
Informed consent verification Per patient    Consent DB
Robot calibration data        Per session    Equipment DB
Image data (CT, US, CBCT)     Per acquisition DICOM + trial DB
Queue and wait times          Per event      Operations DB
Environmental monitoring      Continuous     Facility DB
```

All data streams synchronized to UTC timestamps per ICH E6(R3) Section 4.2.1.

### Section 2.10 - Adverse Event Detection

NEEDLE-01 optical sensor system detected the puncture site bleeding for
PAT-ODMND-0037 within 1 second of onset (08:36:12). The detection triggered
automatic safety protocols (needle pause, clinician alert) consistent with
ICH E6(R3) Section 2.10 requirements for adverse event detection sensitivity.
The event was documented in real-time and classified per CTCAE v5.0 criteria.

### Section 4.2.1 - Data Capture

```
DATA CAPTURE VOLUMES - HOUR 08

Robot Type              Procedures  Data Points  Total Records
---------------------  ----------  -----------  -------------
Surgical (SURG-01)      1 (cont)    360,000/hr   360,000
Surgical (SURG-02)      1 (new)     360,000/hr   180,000
Cobot (COBOT-01)        1           150,000/hr    50,000
RT Positioning          1           200,000/hr   100,000
Companion (COMPN-04)    1            50,000/hr    39,000
RT Tracking (TRACK-02)  1           250,000/hr    83,000
Needle-Placement        1           200,000/hr    83,000
Imaging (IMAGE-03)      1           180,000/hr    57,000
Humanoid (HUMAN-03)     1            80,000/hr    19,000
Imaging (IMAGE-04)      1           180,000/hr    21,000
RT Tracking (TRACK-03)  1           250,000/hr    38,000
Rehab (REHAB-03)        1           100,000/hr     3,000
                       ---                     ----------
Total this hour                                1,033,000
Cumulative (H00-H08)                           4,210,000
```

Data integrity verification: 100% of records passed checksum validation.
No data corruption or loss detected.

### Appendix C - Documentation

Comprehensive documentation maintained for all procedures per ICH E6(R3)
Appendix C adapted requirements. Documentation includes:

- Patient identification and demographics
- Procedure planning and execution records
- Robot assignment and performance metrics
- Investigational drug administration records
- Adverse event reports with full timeline
- Queue event documentation
- Equipment cleaning and turnover records
- Digital twin synchronization logs

## 21 CFR Part 50 Adaption Compliance

### Section 50.25 - Informed Consent Elements

All 12 new patients had previously completed informed consent including:

- Nature and purpose of the Physical AI-mediated procedure
- Disclosure of robot type, model, and USL readiness score
- Description of risks and potential benefits
- Alternative non-Physical AI treatment options
- Right to withdraw consent at any time without penalty
- Data collection and privacy protections
- Contact information for questions and concerns

For the queue event (PAT-ODMND-0041, 8-minute wait), the patient was informed
of the delay and offered the option to reschedule or proceed with a non-
Physical AI alternative. The patient elected to wait, and this decision was
documented per Section 50.25 requirements.

### Section 50.30 - Pre-Procedure Safety Matrix

Pre-procedure safety matrix completed for all 12 new patient procedures:

```
PRE-PROCEDURE SAFETY MATRIX - HOUR 08
(All entries: PASS)

Patient  Auth  ID    FHIR  Robot  Env   Consent  Matrix
-------  ----  ----  ----  -----  ----  -------  ------
P0032    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0033    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0034    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0035    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0036    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0037    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0038    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0039    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0040    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0041    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0042    PASS  PASS  PASS  PASS   PASS  PASS     PASS
P0043    PASS  PASS  PASS  PASS   PASS  PASS     PASS

Auth = Authorization verified
ID = Patient identity confirmed (biometric + wristband)
FHIR = Clinical data accessed via FHIR interface
Robot = Robot readiness confirmed (self-check passed)
Env = Environmental checks passed (room, equipment, supplies)
Consent = Informed consent on file and verified
Matrix = Overall pre-procedure clearance
```

### Subpart C - Cybersecurity Monitoring

Cybersecurity monitoring maintained continuous coverage during the highest-
activity hour of the trial. With 15 robot instances engaged and 12 patients
in various stages of processing, the attack surface is at its largest to
date. All systems passed continuous integrity monitoring. No unauthorized
access attempts detected. Network segmentation between clinical and
administrative systems verified.

### Subpart D - Pediatric Protections

Two pediatric patients processed this hour under Subpart D protections:

PAT-ODMND-0035 (5M, pediatric ALL):
- Parental consent: Father present, consent on file, verified at 08:12
- Age-appropriate assent: Verbal assent obtained using picture-based
  explanation of companion robot interaction
- Dexamethasone administration: Parent informed and consented
- COMPN-04 programmed for age-appropriate interaction (ages 4-6 protocol)
- Parent present throughout session in Companion Play Area 4

PAT-ODMND-0039 (13F, pediatric osteosarcoma):
- Parental consent: Mother present, consent on file, verified at 08:32
- Adolescent assent: Written assent obtained, patient demonstrated
  understanding of humanoid therapy purpose and robot capabilities
- HUMAN-03 programmed for adolescent interaction (ages 12-15 protocol)
- Parent present throughout session in Humanoid Station 3

Both pediatric patients had IRB-approved pediatric protocols in effect per
21 CFR Part 50 Subpart D adapted requirements.

## 21 CFR Part 312 Adaption Compliance

### Section 312.23 - IND Applications

Three investigational drug administrations occurred this hour:

```
IND DRUG ADMINISTRATIONS - HOUR 08

Patient  Drug            Dose         Route  Time   Indication
-------  --------------  -----------  -----  -----  ---------------------
P0032    Atezolizumab    1200 mg      IV     08:10  Neoadjuvant pre-op
P0035    Dexamethasone   4 mg         PO     08:15  Pre-chemo supportive
P0038    Sorafenib       400 mg BID   PO     08:25  HCC concurrent therapy
```

PAT-ODMND-0032: Atezolizumab (anti-PD-L1 checkpoint inhibitor) administered
as neoadjuvant dose per IND protocol for mediastinal tumor. Pre-operative
administration timed 20 minutes before surgical prep. Drug sourced from
investigational pharmacy (lot CQD-ATZ-2026-03, expiry 2027-01). Temperature
chain verified (2-8 degrees C maintained). Infusion completed without
reaction.

PAT-ODMND-0035: Dexamethasone administered as pre-chemotherapy supportive
care. While dexamethasone is not an investigational drug, its administration
is documented under the IND protocol as a required pre-medication for
the investigational chemotherapy regimen to follow. Oral dose administered
by nursing staff with parental observation.

PAT-ODMND-0038: Sorafenib (multi-kinase inhibitor) administered per IND
protocol for hepatocellular carcinoma as concurrent systemic therapy during
imaging and planned ablation. Drug sourced from investigational pharmacy
(lot CQD-SOR-2026-02, expiry 2026-12). Patient confirmed compliance with
fasting requirements (1 hour pre-dose).

### Section 312.32 - Safety Reporting

Adverse event AE-0008-001 (PAT-ODMND-0037, Grade 1 puncture site hemorrhage)
assessed for IND safety reporting requirements:

- Seriousness: Non-serious (no hospitalization, no disability, not
  life-threatening)
- Expectedness: Expected (listed in protocol risk disclosure for needle
  placement procedures)
- Reporting requirement: Does not meet criteria for expedited 15-day or
  7-day IND safety report
- Documentation: Recorded in trial safety database for inclusion in annual
  report per Section 312.33
- No IND hold implications per Section 312.42

### Section 312.33 - Annual Reporting

Data from Hour 08 will be included in the annual IND safety report:
- 1 adverse event (Grade 1, non-serious, expected)
- 3 IND drug administrations (0 drug-related adverse events)
- 12 patient encounters documented
- Robot performance data for all 15 engaged instances

### Section 312.40 - IND Compliance

All investigational drug administrations this hour were conducted in
compliance with IND requirements:
- Drugs stored per approved protocols
- Administration by qualified personnel
- Patient eligibility verified against inclusion/exclusion criteria
- Concurrent medications reviewed for interactions
- Documentation complete per investigator recordkeeping requirements

### Section 312.62 - Investigator Recordkeeping

Investigator records maintained for all 22 patients on site during Hour 08
(10 continuing + 12 new arrivals). Records include:

- Physical AI system interaction logs for each robot-patient encounter
- Vital sign records at protocol-specified intervals
- Procedure outcome documentation with robot performance metrics
- Drug administration records with lot numbers and chain of custody
- Adverse event documentation with complete timeline
- Queue event documentation with patient notification records
- Informed consent verification records
- Digital twin synchronization logs

## Protocol Deviation Assessment

No protocol deviations occurred during Hour 08.

The 8-minute queue wait for PAT-ODMND-0041 was assessed for potential protocol
deviation but was determined to be within acceptable operational parameters:
- The protocol does not specify a maximum wait time for RT vault access
- The patient was informed and consented to wait
- No clinical impact from the delay
- The event is documented for operational review

The adverse event (AE-0008-001) was managed within protocol-specified
procedures for intra-procedural bleeding. No deviation from the adverse event
management protocol occurred.

## Compliance Metrics Summary

```
REGULATORY COMPLIANCE DASHBOARD - HOUR 08

Metric                               Value   Target  Status
------------------------------------  ------  ------  ------
Informed consent completion rate      100%    100%    MET
Pre-procedure safety matrix pass      12/12   12/12   MET
IND drug documentation complete       3/3     3/3     MET
Adverse event detection time          <1 sec  <5 sec  MET
Adverse event documentation time      <5 min  <15 min MET
Audit trail completeness              100%    100%    MET
Data integrity (checksum pass)        100%    100%    MET
Pediatric Subpart D compliance        2/2     2/2     MET
Cybersecurity alerts                  0       0       MET
Protocol deviations                   0       0       MET
Queue events documented               1/1     1/1     MET
Equipment cleaning documented         4/4     4/4     MET
```

## USL and Patient Journey References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) supports regulatory compliance through
standardized data formats and cross-robot communication protocols. USL-
evaluated interoperability enables the comprehensive audit trails required
by ICH E6(R3) and the real-time safety monitoring required by 21 CFR Part 312.
At 52% robot utilization, USL cross-robot sharing capabilities are essential
for maintaining complete data coverage across all concurrent procedures.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) established baseline regulatory compliance
patterns for individual patient interactions. The multi-patient on-demand
model scales these compliance requirements across 22 concurrent patients,
3 IND drug administrations, and 15 robot instances while maintaining 100%
compliance rates on all measured metrics.
