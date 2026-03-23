# Hour 08: Patient Arrivals - 12 New On-Demand Patients

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Arrival Summary

Hour 08 records 12 new patient arrivals, the highest single-hour intake in the
trial to date. This marks the transition into the peak morning period
(08:00-15:00) as described in the site specification, which projects peak
concurrent occupancy of 60-80 patients. The 12 arrivals span 8 of the 10
robot types, 9 distinct cancer types, and include 2 pediatric patients. Ages
range from 5 to 72 years. ECOG performance status ranges from 0 to 2.

## Regulatory Framework References

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368)
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707)
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628)

All arrivals processed under on-demand scheduling with informed consent
previously completed per 21 CFR Part 50 Section 50.25, including Physical AI
system disclosure, USL readiness scores, and right to non-Physical AI
alternatives. Pediatric arrivals (PAT-ODMND-0035 and PAT-ODMND-0039) processed
under 21 CFR Part 50 Subpart D with parental/guardian consent and age-
appropriate assent documentation.

## Arrival Table

| # | Patient ID | Time | Age | Sex | Cancer Type | Stage/Grade | ECOG | Robot | Instance |
|---|-----------|------|-----|-----|-------------|-------------|------|-------|----------|
| 1 | PAT-ODMND-0032 | 08:02 | 54 | M | Mediastinal tumor | Stage III | 1 | Surgical | SURG-02 |
| 2 | PAT-ODMND-0033 | 08:05 | 38 | F | Forearm sarcoma | Grade II | 0 | Cobot | COBOT-01 |
| 3 | PAT-ODMND-0034 | 08:10 | 70 | M | Glioblastoma | Stage IV | 2 | RT Positioning | RTPOS-01 |
| 4 | PAT-ODMND-0035 | 08:12 | 5 | M | Pediatric ALL | N/A | 1 | Companion | COMPN-04 |
| 5 | PAT-ODMND-0036 | 08:18 | 62 | F | NSCLC squamous | Stage IIIB | 1 | RT Tracking | TRACK-02 |
| 6 | PAT-ODMND-0037 | 08:22 | 49 | M | Parotid tumor | Stage II | 0 | Needle-Placement | NEEDLE-01 |
| 7 | PAT-ODMND-0038 | 08:28 | 57 | F | HCC | Stage III | 1 | Imaging + Steerable | IMAGE-03, STEER-01 |
| 8 | PAT-ODMND-0039 | 08:32 | 13 | F | Pediatric osteosarcoma | N/A | 1 | Humanoid | HUMAN-03 |
| 9 | PAT-ODMND-0040 | 08:38 | 66 | M | Liver mets (colorectal) | Stage IV | 2 | Imaging | IMAGE-04 |
| 10 | PAT-ODMND-0041 | 08:42 | 44 | F | NSCLC adenocarcinoma | Stage IIB | 0 | RT Tracking | TRACK-03 |
| 11 | PAT-ODMND-0042 | 08:48 | 72 | M | Femur osteosarcoma | Post-surgical | 2 | Rehab Exoskeleton | REHAB-03 |
| 12 | PAT-ODMND-0043 | 08:55 | 31 | F | Forearm sarcoma | Grade I | 0 | Cobot | COBOT-02 |

## Individual Patient Narratives

### PAT-ODMND-0032 - Mediastinal Tumor Surgery

54-year-old male with Stage III mediastinal tumor presenting for robotic-
assisted surgical resection. Previous imaging confirmed a 4.8 cm anterior
mediastinal mass with no major vascular invasion. Patient selected the 08:02
morning slot for surgical availability. ECOG performance status 1. Pre-
operative atezolizumab neoadjuvant dose administered per IND protocol
(21 CFR Part 312 Section 312.23). Patient journey stage: Stage 4 (Treatment
Delivery) within multi-patient on-demand context. Pre-procedure safety matrix
completed per 21 CFR Part 50 Section 50.30. Assigned to SURG-02 in Surgical
Suite 2. Surgery preparation begins at 08:15 with anticipated incision at
08:30.

### PAT-ODMND-0033 - Forearm Sarcoma Biopsy

38-year-old female with Grade II forearm soft-tissue sarcoma presenting for
cobot-assisted core needle biopsy. A 3.2 cm mass in the right forearm
extensor compartment was identified on MRI two weeks prior. Patient selected
the on-demand morning slot. ECOG performance status 0. Patient journey stage:
Stage 2 (Diagnostic Workup). Assigned to COBOT-01 at Biopsy Station 1.
Procedure scheduled to begin at 08:20.

### PAT-ODMND-0034 - Glioblastoma RT Positioning

70-year-old male with Stage IV glioblastoma multiforme presenting for
radiotherapy positioning and mask fitting. Post-surgical resection (external
site, 4 weeks prior), now initiating adjuvant RT planning. ECOG performance
status 2 (ambulatory, capable of self-care, unable to carry out work
activities). Patient journey stage: Stage 4 (Treatment Delivery - RT
component). Assigned to RTPOS-01 in RT Vault 1. Session scheduled to begin
at 08:25.

### PAT-ODMND-0035 - Pediatric ALL Companion Session

5-year-old male with pediatric acute lymphoblastic leukemia presenting for
pre-chemotherapy companion robot session. The companion robot provides
anxiety reduction through interactive play, distraction therapy, and
treatment familiarization before scheduled chemotherapy administration.
ECOG performance status 1. Dexamethasone pre-chemotherapy dose administered
per standard supportive care protocol. Parental consent and age-appropriate
assent completed per 21 CFR Part 50 Subpart D. Patient journey stage:
Stage 3 (Treatment Preparation). Assigned to COMPN-04 at Companion Play
Area 4. Session begins immediately upon arrival.

### PAT-ODMND-0036 - NSCLC Squamous RT Tracking

62-year-old female with Stage IIIB non-small cell lung cancer (squamous cell
carcinoma) presenting for real-time motion-tracked radiotherapy. Tumor located
in the right upper lobe with mediastinal lymph node involvement. ECOG
performance status 1. Patient journey stage: Stage 4 (Treatment Delivery -
RT fraction 8 of 30). Assigned to TRACK-02 in RT Vault 2. Treatment
scheduled to begin at 08:35.

### PAT-ODMND-0037 - Parotid Tumor Needle Placement

49-year-old male with Stage II parotid tumor presenting for CT-guided needle
biopsy of a 2.1 cm left parotid mass. ECOG performance status 0. Patient
journey stage: Stage 2 (Diagnostic Workup). Assigned to NEEDLE-01 in CT
Suite 1. Procedure scheduled to begin at 08:30. Note: This patient will
experience a Grade 1 adverse event (minor bleeding at puncture site) during
the procedure. See hour-08-adverse-events.md.

### PAT-ODMND-0038 - HCC Imaging and Ablation

57-year-old female with Stage III hepatocellular carcinoma presenting for
combined robotic imaging assessment followed by steerable needle ablation.
This is a two-robot sequential procedure: IMAGE-03 performs ultrasound-guided
tumor mapping, followed by STEER-01 for targeted radiofrequency ablation.
ECOG performance status 1. Sorafenib administered per IND protocol
(21 CFR Part 312 Section 312.23) as concurrent systemic therapy. Patient
journey stage: Stage 4 (Treatment Delivery). Assigned to IMAGE-03 in Imaging
Bay 3 (phase 1) and STEER-01 in Ablation Suite 1 (phase 2, expected Hour 09).

### PAT-ODMND-0039 - Pediatric Osteosarcoma Humanoid Therapy

13-year-old female with pediatric osteosarcoma (distal femur) presenting for
humanoid-assisted physical therapy preparation. The humanoid robot
demonstrates exercises, provides motivational coaching, and models movement
patterns for the patient to follow. ECOG performance status 1. Parental
consent and adolescent assent completed per 21 CFR Part 50 Subpart D.
Patient journey stage: Stage 4 (Treatment Delivery - rehabilitation
component). Assigned to HUMAN-03 at Humanoid Station 3. Session scheduled
to begin at 08:45.

### PAT-ODMND-0040 - Liver Metastases Imaging

66-year-old male with Stage IV colorectal cancer with liver metastases
presenting for robotic ultrasound imaging assessment. Multiple hepatic
lesions identified on prior CT; robotic imaging provides high-resolution
characterization and digital twin calibration for treatment planning. ECOG
performance status 2. Patient journey stage: Stage 2 (Diagnostic Workup -
metastatic characterization). Assigned to IMAGE-04 in Imaging Bay 4.
Procedure scheduled to begin at 08:52.

### PAT-ODMND-0041 - NSCLC Adenocarcinoma RT Tracking

44-year-old female with Stage IIB non-small cell lung cancer (adenocarcinoma)
presenting for real-time motion-tracked radiotherapy. Tumor in the left lower
lobe, 3.5 cm, no nodal involvement. ECOG performance status 0. Patient
journey stage: Stage 4 (Treatment Delivery - RT fraction 3 of 25). Assigned
to TRACK-03 in RT Vault 3. Note: This patient experiences the first queue
event of the trial, waiting 8 minutes (08:42-08:50) for TRACK-03 availability
due to vault preparation after the previous session. See hour-08-robot-
utilization.md for queue analysis.

### PAT-ODMND-0042 - Femur Osteosarcoma Rehabilitation

72-year-old male with post-surgical femur osteosarcoma presenting for
exoskeleton-assisted rehabilitation. He is 8 weeks post-limb-salvage surgery
and selected the morning slot for rehabilitation. ECOG performance status 2.
Patient journey stage: Stage 7 (Recovery and Rehabilitation). Assigned to
REHAB-03 in Rehab Bay 3. Session scheduled to begin at 08:58.

### PAT-ODMND-0043 - Forearm Sarcoma Biopsy

31-year-old female with Grade I forearm soft-tissue sarcoma presenting for
cobot-assisted core needle biopsy. A 1.8 cm mass in the left forearm flexor
compartment was identified on ultrasound. ECOG performance status 0. Patient
journey stage: Stage 2 (Diagnostic Workup). Assigned to COBOT-02 at Biopsy
Station 2. Late arrival at 08:55; pre-procedure preparation begins this
hour with biopsy procedure expected to begin in Hour 09.

## Arrival Demographics Summary

```
ARRIVALS BY CANCER TYPE                 ARRIVALS BY AGE GROUP
--------------------------              --------------------------
Mediastinal tumor .... 1                Pediatric (0-17) ..... 2
Forearm sarcoma ...... 2                Young adult (18-39) .. 2
Glioblastoma ......... 1                Middle age (40-59) ... 4
Pediatric ALL ........ 1                Senior (60-74) ....... 4
NSCLC squamous ....... 1
Parotid tumor ........ 1                ARRIVALS BY ECOG STATUS
HCC .................. 1                --------------------------
Pediatric osteo ...... 1                ECOG 0 ............... 4
Liver mets (CRC) ..... 1                ECOG 1 ............... 5
NSCLC adeno .......... 1                ECOG 2 ............... 3
Femur osteosarcoma ... 1
                                        ARRIVALS BY SEX
ARRIVALS BY ROBOT TYPE                  --------------------------
--------------------------              Male ................. 6
Surgical ............. 1                Female ............... 6
Cobot ................ 2
RT Positioning ....... 1                IND DRUG PATIENTS
RT Tracking .......... 2                --------------------------
Needle-Placement ..... 1                Atezolizumab ......... 1
Imaging .............. 2                Sorafenib ............ 1
Steerable Needle ..... 1                Dexamethasone ........ 1
Companion ............ 1
Humanoid ............. 1
Rehab Exoskeleton .... 1
```

## Arrival Rate Analysis

```
ARRIVAL DISTRIBUTION ACROSS HOUR 08
Minutes 00-09: ** (2 patients: P0032, P0033)
Minutes 10-19: *** (3 patients: P0034, P0035, P0036)
Minutes 20-29: ** (2 patients: P0037, P0038)
Minutes 30-39: ** (2 patients: P0039, P0040)
Minutes 40-49: ** (2 patients: P0041, P0042)
Minutes 50-59: * (1 patient: P0043)

Average inter-arrival time: 4.8 minutes
Minimum inter-arrival time: 2 minutes (P0034 to P0035)
Maximum inter-arrival time: 6 minutes (P0036 to P0037, P0040 to P0041)
```

## USL and Patient Journey References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) evaluates each robot type's technical
interoperability readiness. With 8 of 10 robot types engaged simultaneously
in Hour 08, USL cross-robot sharing capabilities are exercised for the first
time at near-full breadth. IMAGE-03 and STEER-01 coordination for P0038
represents a multi-robot sequential handoff evaluated under USL simulation
switching criteria.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) maps individual patient stages. Hour 08
arrivals span journey stages 2 through 7, demonstrating the on-demand model's
capacity to serve patients at different points in their treatment trajectory
simultaneously.
