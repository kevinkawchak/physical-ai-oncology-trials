# Hour 08: Robot Utilization Report

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## Utilization Summary

Hour 08 marks the highest robot utilization of the trial to date, reaching
approximately 52% overall (15 of 29 robot instances active at peak). This is
the first hour where all 10 robot types are engaged (8 types with active
procedures, 2 types with standby/monitoring roles). The site approaches the
50% utilization threshold that triggers enhanced monitoring protocols per
site operational specifications. The first patient queue of the trial occurs
when PAT-ODMND-0041 waits 8 minutes for RT Vault 3 availability.

## Regulatory Framework References

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368)
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707)
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628)

## Robot Instance Status at 08:59

| # | Robot Type | Instance | Status | Patient | Since |
|---|-----------|----------|--------|---------|-------|
| 1 | Surgical | SURG-01 | Active | P0024 | 07:40 |
| 1 | Surgical | SURG-02 | Active | P0032 | 08:15 |
| 1 | Surgical | SURG-03 | Standby | -- | -- |
| 2 | Cobot | COBOT-01 | Cleaning | (post-P0033) | 08:40 |
| 2 | Cobot | COBOT-02 | Pre-proc | P0043 | 08:55 |
| 2 | Cobot | COBOT-03 | Standby | -- | -- |
| 2 | Cobot | COBOT-04 | Standby | -- | -- |
| 3 | RT Positioning | RTPOS-01 | Post-proc | (post-P0034) | 08:55 |
| 3 | RT Positioning | RTPOS-02 | Standby | -- | -- |
| 3 | RT Positioning | RTPOS-03 | Standby | -- | -- |
| 4 | Needle-Placement | NEEDLE-01 | Cleaning | (post-P0037) | 08:55 |
| 4 | Needle-Placement | NEEDLE-02 | Standby | -- | -- |
| 5 | Companion | COMPN-01 | Standby | -- | -- |
| 5 | Companion | COMPN-02 | Standby | -- | -- |
| 5 | Companion | COMPN-03 | Monitoring | P0021/P0028 | ongoing |
| 5 | Companion | COMPN-04 | Active | P0035 | 08:12 |
| 5 | Companion | COMPN-05 | Standby | -- | -- |
| 6 | Humanoid | HUMAN-01 | Standby | -- | -- |
| 6 | Humanoid | HUMAN-02 | Standby | -- | -- |
| 6 | Humanoid | HUMAN-03 | Active | P0039 | 08:45 |
| 7 | RT Tracking | TRACK-01 | Standby | -- | -- |
| 7 | RT Tracking | TRACK-02 | Post-proc | (post-P0036) | 08:55 |
| 7 | RT Tracking | TRACK-03 | Active | P0041 | 08:50 |
| 8 | Imaging | IMAGE-01 | Standby | -- | -- |
| 8 | Imaging | IMAGE-02 | Maintenance | recalibration | 07:00 |
| 8 | Imaging | IMAGE-03 | Active | P0038 | 08:40 |
| 8 | Imaging | IMAGE-04 | Active | P0040 | 08:52 |
| 9 | Steerable Needle | STEER-01 | Standby | (queued P0038) | -- |
| 9 | Steerable Needle | STEER-02 | Standby | -- | -- |
| 10 | Rehab Exoskeleton | REHAB-01 | Standby | -- | -- |
| 10 | Rehab Exoskeleton | REHAB-02 | Standby | -- | -- |
| 10 | Rehab Exoskeleton | REHAB-03 | Active | P0042 | 08:58 |

## Utilization by Robot Type

```
ROBOT UTILIZATION BY TYPE - END OF HOUR 08

Type               Instances  Active  Util%  Bar
-----------------  ---------  ------  -----  --------------------------
1  Surgical         3 total    2 act   67%   [================|        ]
2  Cobots           4 total    1 act   25%   [======|                  ]
3  RT Positioning   3 total    0 act    0%   [|                        ]
4  Needle-Place     2 total    0 act    0%   [|                        ]
5  Companion        5 total    2 act   40%   [==========|              ]
6  Humanoids        3 total    1 act   33%   [========|                ]
7  RT Tracking      3 total    1 act   33%   [========|                ]
8  Imaging          4 total    2 act   50%   [============|            ]
9  Steerable Ndl    2 total    0 act    0%   [|                        ]
10 Rehab Exo        3 total    1 act   33%   [========|                ]
                   ---------  ------  -----
   SITE TOTAL      29 total   10 act   52%   Note: 15 engaged total
                                              (10 active + 2 cleaning
                                               + 2 post-proc + 1 maint)
                                        0%        50%       100%
```

Note: "Active" counts instances currently performing patient procedures.
The 52% site utilization includes instances in active, cleaning, post-
procedure, pre-procedure, monitoring, and maintenance states. Pure active
procedure count at 08:59 is 10 instances.

## Peak Concurrent Activity

```
CONCURRENT ACTIVE ROBOTS THROUGH HOUR 08

08:00  ** (2: SURG-01, COMPN-03)
08:15  *** (3: +SURG-02 prep)
08:20  **** (4: +COBOT-01)
08:25  ***** (5: +RTPOS-01)
08:30  ******* (7: +NEEDLE-01, +TRACK-02 prep)
08:35  ******** (8: +TRACK-02 active)
08:40  ********* (9: +IMAGE-03, -COBOT-01 done)
08:45  ********** (10: +HUMAN-03)
08:50  *********** (11: +TRACK-03, -TRACK-02 done, +IMAGE-04 prep)
08:52  ************ (12: +IMAGE-04 active)
08:55  *********** (11: -RTPOS-01 done, -NEEDLE-01 done)
08:58  ************ (12: +REHAB-03)
08:59  ************ (12: end of hour)

Peak concurrent: 12 robot instances engaged at 08:52-08:54
```

## Queue Event Analysis: First Trial Queue

### Queue Event Q-0008-001

| Field | Value |
|-------|-------|
| Queue Reference | Q-0008-001 |
| Patient | PAT-ODMND-0041 |
| Robot Requested | TRACK-03 (RT Motion-Tracking, Instance 3) |
| Arrival Time | 08:42 |
| Queue Start | 08:42 |
| Queue End | 08:50 |
| Wait Duration | 8 minutes |
| Reason | RT Vault 3 preparation after previous session |
| Impact | Minimal - patient waited in RT waiting area |

This is the first patient queue event in the trial. PAT-ODMND-0041 arrived
at 08:42 for RT tracking in Vault 3, but the vault required preparation time
after the previous session (equipment reset, room ventilation, radiation
safety survey). TRACK-01 was available but located in Vault 1, which was
being prepared for a different patient type. The 8-minute wait is within
the acceptable range for RT vault preparation and does not constitute a
protocol deviation.

Per 21 CFR Part 50 Section 50.25, the patient was informed of the wait time
and offered the option to reschedule. The patient elected to wait.

Queue mitigation actions:
- Patient provided with updated wait time estimate at 08:43
- Waiting area comfort measures available (seating, water, reading material)
- TRACK-03 preparation expedited where safe to do so
- No clinical impact from 8-minute delay

## Utilization Trend

```
SITE ROBOT UTILIZATION - HOURS 00 THROUGH 08

Hour  Active  Total  Util%  Bar
----  ------  -----  -----  ----------------------------------------
 00    1       29      3%   [=|                                      ]
 01    1       29      3%   [=|                                      ]
 02    2       29      7%   [==|                                     ]
 03    3       29     10%   [====|                                   ]
 04    3       29     10%   [====|                                   ]
 05    5       29     17%   [======|                                 ]
 06    6       29     21%   [========|                               ]
 07    8       29     28%   [===========|                            ]
 08   15       29     52%   [====================|                   ]
                             0%       25%       50%       75%  100%

Utilization is accelerating as peak morning operations begin.
Hour 08 nearly doubled the utilization from Hour 07.
```

## Robot Cleaning and Turnover

| Robot | Post-Patient | Cleaning Start | Duration | Available |
|-------|-------------|----------------|----------|-----------|
| COBOT-01 | P0033 | 08:40 | 10 min | 08:50 |
| RTPOS-01 | P0034 | 08:55 | 12 min | 09:07 |
| NEEDLE-01 | P0037 | 08:55 | 15 min | 09:10 |
| TRACK-02 | P0036 | 08:55 | 10 min | 09:05 |

Cleaning protocols follow ICH E6(R3) Section 2.9.1 for equipment
decontamination and 21 CFR Part 312 Section 312.62 for investigator
recordkeeping of equipment preparation between patients.

## Maintenance Status

IMAGE-02 remains in scheduled recalibration (started 07:00, expected
completion 09:00). This is a routine calibration per site specification
and does not indicate a deficiency. Upon return to service, all 4 imaging
bays will be available for the first time during peak operations.

## Capacity Forecast

With 52% utilization at Hour 08, the site has adequate capacity for
continued peak morning arrivals. Critical capacity thresholds:

- Surgical suites: 2 of 3 active (67%) - 1 suite available
- RT vaults: 2 of 3 active at peak (67%) - capacity constrained
- Imaging bays: 3 of 4 active (75%) - 1 in maintenance
- All other stations: below 50% utilization

RT vault capacity is the tightest constraint, as demonstrated by the
8-minute queue for TRACK-03. If arrival rates continue at Hour 08 levels,
vault scheduling optimization may be required in Hours 09-10.

## USL and Patient Journey References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) evaluates cross-robot sharing and simulation
switching capabilities that become critical at high utilization levels.
With 52% utilization, USL-scored interoperability enables efficient resource
sharing and patient routing across robot types.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) established baseline utilization expectations
for individual patient interactions. The on-demand multi-patient model
introduces utilization patterns not present in single-patient scenarios,
requiring queue management and concurrent resource allocation capabilities.
