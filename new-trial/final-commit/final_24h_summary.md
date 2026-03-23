# Final 24-Hour Summary: On-Demand Physical AI Oncology Trial Simulation

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Regulatory Framework

This 24-hour on-demand simulation was conducted under three adapted
regulatory frameworks:

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368) - Good Clinical Practice
  guidelines adapted for Physical AI autonomous clinical trial operations.
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707) - Protection of human
  subjects adapted for robotic-mediated informed consent and safety oversight.
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628) - Investigational
  New Drug regulations adapted for Physical AI trial IND management.

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary robot technical
interoperability scoring. The single-patient cancer journey framework
(Kawchak, 2026; DOI: 10.5281/zenodo.19119939) demonstrated autonomous
Physical AI trial orchestration for an individual patient.

## Executive Summary

A 24-hour on-demand Physical AI oncology trial simulation was completed
from 00:00 to 23:59 on 23 March 2026, demonstrating continuous autonomous
robotic oncology care delivery. The facility served 175 unique patients
(PAT-ODMND-0001 through PAT-ODMND-0175, including 3 carryover patients from
the prior day cycle) using 10 Physical AI robot types across 29 instances.
Approximately 185 procedures were completed with an average wait time of
8 minutes from arrival to procedure start. Seven adverse events occurred,
all Grade 1-2, all managed successfully with no patient harm. The site PSL
score progressed from 63.4 to 64.4 (+1.0) over the 24-hour cycle. Zero
robot malfunctions required patient rescheduling.

## Patient Volume Summary

### Total Patient Counts
- Unique patient IDs issued: 175 (PAT-ODMND-0001 through PAT-ODMND-0175)
- New arrivals during 24-hour cycle: 172
- Carryover patients from prior day: 3 (P0003, P0004, P0005)
- Total patient encounters (touches): 171
- Procedures completed: approximately 185 (some patients had multiple)
- Patients remaining on-site at 23:59: 4 (P0154, P0173, P0174, P0175)

### Patients by Cancer Type

| Cancer Type | Patient Count | Percentage |
|-------------|--------------|------------|
| NSCLC (adenocarcinoma and squamous) | 28 | 16.0% |
| HCC (hepatocellular carcinoma) | 18 | 10.3% |
| Colorectal cancer (including liver mets) | 16 | 9.1% |
| Brain metastases | 14 | 8.0% |
| Pediatric ALL | 12 | 6.9% |
| Forearm/soft-tissue sarcoma | 11 | 6.3% |
| Breast cancer | 11 | 6.3% |
| Mediastinal tumors | 10 | 5.7% |
| Meningioma | 9 | 5.1% |
| Parotid/salivary tumors | 8 | 4.6% |
| Pediatric osteosarcoma | 8 | 4.6% |
| Femur osteosarcoma | 7 | 4.0% |
| Pediatric AML | 6 | 3.4% |
| Ewing sarcoma | 5 | 2.9% |
| Pancreatic cancer | 4 | 2.3% |
| Other (thyroid, renal, prostate, etc.) | 8 | 4.6% |
| **Total** | **175** | **100%** |

### Hourly Arrival Distribution

| Hour | New Arrivals | Cumulative | Utilization |
|------|-------------|------------|-------------|
| 00 | 2 | 2 | 2.3% |
| 01 | 2 | 4 | 3.4% |
| 02 | 3 | 7 | 6.9% |
| 03 | 3 | 10 | 10.3% |
| 04 | 4 | 14 | 13.8% |
| 05 | 5 | 19 | 17.2% |
| 06 | 7 | 26 | 24.1% |
| 07 | 10 | 36 | 34.5% |
| 08 | 12 | 48 | 51.7% |
| 09 | 15 | 63 | 72.4% |
| 10 | 14 | 77 | 69.0% |
| 11 | 13 | 90 | 65.5% |
| 12 | 12 | 102 | 58.6% |
| 13 | 11 | 113 | 55.2% |
| 14 | 10 | 123 | 48.3% |
| 15 | 9 | 132 | 44.8% |
| 16 | 8 | 140 | 37.9% |
| 17 | 7 | 147 | 31.0% |
| 18 | 6 | 153 | 24.1% |
| 19 | 5 | 158 | 20.7% |
| 20 | 5 | 163 | 17.2% |
| 21 | 4 | 167 | 13.8% |
| 22 | 4 | 171 | 10.3% |
| 23 | 2 | 173 | 6.9% |

- Peak hour: Hour 09 (15 arrivals, 72.4% utilization)
- Lowest hour: Hour 00 (2 arrivals, 2.3% utilization)
- Morning ramp (06-09): 44 arrivals (25.4% of daily volume)
- Daytime peak (09-15): 84 arrivals (48.6% of daily volume)
- Evening decline (16-22): 39 arrivals (22.5% of daily volume)
- Overnight (23-05): 19 arrivals (11.0% of daily volume)

## Procedures by Robot Type

| Robot Type | Instances | Procedures | Avg Duration | Key Metric |
|------------|-----------|------------|--------------|------------|
| Surgical Robots | 3 (SURG-01/02/03) | 7 | 142 min | 100% R0 resection |
| Cobots | 4 (COBOT-01/02/03/04) | 24 | 18 min | 100% tissue quality |
| RT Positioning | 3 (RTPOS-01/02/03) | 22 | 25 min | <1 mm positioning |
| Needle-Placement | 2 (NEEDLE-01/02) | 16 | 32 min | 100% target accuracy |
| Social Companion | 5 (COMPN-01 to 05) | 28 | 45 min (sessions) | 95% anxiety reduction |
| Humanoids | 3 (HUMAN-01/02/03) | 18 | 35 min | 92% engagement rate |
| RT Motion-Tracking | 3 (TRACK-01/02/03) | 26 | 22 min | 94.2% beam gating |
| Imaging Assistant | 4 (IMAGE-01/02/03/04) | 30 | 20 min | 100% diagnostic quality |
| Steerable Needle | 2 (STEER-01/02) | 8 | 48 min | <0.5 mm tip error |
| Rehab Exoskeletons | 2 (REHAB-01/02) (note: site has 3 bays) | 6 | 40 min | 88% ROM improvement |
| **Total** | **29** (note: 2 REHAB + 1 bay idle) | **~185** | | |

## Surgical Summary

| Surgery | Patient | Cancer Type | Duration | Outcome |
|---------|---------|-------------|----------|---------|
| Robotic thoracoscopic resection | P0024 | Mediastinal tumor | 185 min | R0 |
| Robotic esophagectomy | P0154 | Esophageal cancer | 210 min | R0 |
| Minimally invasive lobectomy | P0032 | NSCLC | 155 min | R0 |
| Robotic mediastinal resection | P0044 | Mediastinal tumor | 130 min | R0 |
| Robotic hepatectomy | P0089 | HCC | 145 min | R0 |
| Robotic Whipple procedure | P0110 | Pancreatic cancer | 195 min | R0 |
| Robotic thyroidectomy | P0135 | Thyroid cancer | 95 min | R0 |

All 7 surgeries achieved R0 resection (complete tumor removal with negative
margins). No intraoperative conversions to open surgery. No surgical site
infections at 24-hour assessment.

## Adverse Event Summary

| # | Patient | Hour | Event | Grade | Robot | Action | Resolution |
|---|---------|------|-------|-------|-------|--------|------------|
| 1 | P0029 | 04 | Nausea | 1 | COMPN-03 | Ondansetron 4 mg IV | Resolved in 15 min |
| 2 | P0024 | 07 | Hypotension | 1 | SURG-02 | 500 mL NS bolus | Resolved in 10 min |
| 3 | P0037 | 08 | Bleeding (puncture site) | 1 | NEEDLE-01 | Manual pressure 5 min | Resolved in 8 min |
| 4 | P0081 | 12 | Pain (post-procedure) | 2 | COBOT-02 | Morphine 2 mg IV | Resolved in 20 min |
| 5 | P0118 | 15 | Cough (during RT) | 1 | TRACK-01 | Beam hold, coaching | Resolved in 5 min |
| 6 | P0142 | 18 | O2 desaturation (89%) | 1 | REHAB-01 | O2 2L NC, rest | Resolved in 12 min |
| 7 | P0158 | 20 | Anxiety (pre-procedure) | 1 | COMPN-04 | Lorazepam 0.5 mg, companion | Resolved in 15 min |

- Total adverse events: 7
- Grade 1: 6 (85.7%)
- Grade 2: 1 (14.3%)
- Grade 3-5: 0 (0.0%)
- All events resolved within the same hour
- No adverse events required procedure cancellation
- No adverse events required emergency medical transfer
- Adverse event rate: 7/185 procedures = 3.8%

## Drug Administration Summary

| Drug Category | Administrations | Notes |
|---------------|----------------|-------|
| Investigational (IND protocol) | 12 | Per 21 CFR Part 312 adapted |
| Chemotherapy (standard of care) | 18 | FOLFOX, carboplatin/paclitaxel, etc. |
| Supportive care (antiemetics) | 22 | Ondansetron, dexamethasone |
| Analgesics | 15 | Acetaminophen, morphine, ketorolac |
| Anxiolytics | 8 | Lorazepam, midazolam (pediatric) |
| Anesthesia (surgical) | 7 | General anesthesia for surgeries |
| Contrast agents | 30 | Imaging procedures |
| Other (antibiotics, fluids) | 14 | Prophylactic and therapeutic |
| **Total administrations** | **~126** | |

All IND drug administrations followed adapted 21 CFR Part 312 protocols
with real-time robot-mediated dosing verification and digital audit trail.

## Wait Time and Throughput

| Metric | Value |
|--------|-------|
| Average wait time (arrival to procedure) | 8 minutes |
| Median wait time | 6 minutes |
| Maximum wait time | 22 minutes (Hour 09 peak) |
| Minimum wait time | 0 minutes (overnight hours) |
| Average procedure duration | 28 minutes |
| Average total visit time | 52 minutes |
| Patient throughput rate | 7.3 patients/hour (average) |
| Peak throughput | 15 patients/hour (Hour 09) |

## Patient Satisfaction (Simulated)

| Metric | Score |
|--------|-------|
| Overall satisfaction | 4.7/5.0 |
| Wait time satisfaction | 4.8/5.0 |
| Robot interaction comfort | 4.5/5.0 |
| Pain management | 4.6/5.0 |
| Information clarity | 4.7/5.0 |
| Scheduling convenience | 4.9/5.0 |
| Pediatric experience (parent-rated) | 4.6/5.0 |
| Would recommend to others | 94% |

Scheduling convenience scored highest (4.9/5.0), reflecting the value of
the on-demand 24-hour model where patients select their preferred time
window via the patient portal.

## Site Uptime and Availability

| Metric | Value |
|--------|-------|
| Overall robot fleet uptime | 99.7% |
| Planned maintenance downtime | SURG-01: 4 hours (22:15-01:59) |
| Unplanned downtime | 0 hours |
| Hours with at least 1 active procedure | 24 of 24 |
| Maximum concurrent robots active | 21 of 29 (Hour 09) |
| Minimum concurrent robots active | 1 of 29 (Hour 00-01) |
| Site safety officer coverage | 100% (3 shifts) |
| Emergency stop activations | 0 |
| Radiation interlock trips | 0 |

## PSL Framework Summary

| Metric | Value |
|--------|-------|
| Starting site PSL (Hour 00) | 63.4 |
| Ending site PSL (Hour 23) | 64.4 |
| Total change | +1.0 |
| Site classification | Advanced Site |
| Highest robot PSL | RT Motion-Tracking: 7.1 |
| Lowest robot PSL | Rehab Exoskeletons: 5.6, Social Companion: 5.7 |
| Largest robot improvement | Surgical Robots: +0.2, Imaging: +0.2 |
| PSL constraint violations | 0 (all changes within 0.3/hr/dim max) |

## Comparison vs Traditional Oncology Facility

| Metric | This Simulation | Traditional Facility | Improvement |
|--------|----------------|---------------------|-------------|
| Operating hours | 24 hours/day | 8-10 hours/day | 2.4-3x availability |
| Average wait time | 8 minutes | 45-90 minutes | 5.6-11.3x faster |
| Scheduling flexibility | Patient-chosen any hour | Fixed appointment slots | Full flexibility |
| Procedures per day | ~185 | 40-60 | 3.1-4.6x throughput |
| Surgical R0 rate | 100% (7/7) | 85-92% typical | +8-15% improvement |
| Adverse event rate | 3.8% (all Grade 1-2) | 8-15% (includes Grade 3+) | 2.1-3.9x safer |
| Staff required | 6-8 per shift | 40-60 per shift | 5-10x reduction |
| Documentation | 100% automated | 60-80% manual | Full automation |
| Patient satisfaction | 4.7/5.0 | 3.8-4.2/5.0 | +0.5-0.9 points |

## Conclusion

The 24-hour on-demand Physical AI oncology trial simulation successfully
demonstrated that a fully autonomous robotic oncology facility can operate
continuously with high patient throughput, low adverse event rates, and
strong patient satisfaction. The on-demand scheduling model enabled patients
to select treatment windows that accommodate work, caregiver, and personal
schedules, including overnight hours that are unavailable at traditional
facilities. The PSL framework confirmed incremental performance improvement
across the cycle, and all operations remained within adapted regulatory
framework requirements. This simulation validates the feasibility of
scaling Physical AI oncology trial sites for 24/7 patient-driven care
delivery.
