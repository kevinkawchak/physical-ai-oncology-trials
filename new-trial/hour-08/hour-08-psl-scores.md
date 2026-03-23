# Hour 08: Physical AI Standard Level (PSL) Scores

Released on 23 March 2026 | CEO Kevin Kawchak | ChemicalQDevice

## PSL Score Summary

Site PSL advances from 63.9 (end of Hour 07) to 64.3 at the end of Hour 08.
The +0.4 increase reflects two notable improvements: Surgical robots gain
+0.1 on Dimension A (multi-patient surgical awareness now active with two
concurrent mediastinal surgeries sharing contextual data between SURG-01 and
SURG-02), and Imaging Assistant robots gain +0.1 on Dimension B (3 of 4
imaging bays are now active, demonstrating improved omnipresent coverage).
No PSL decrements occurred despite the Grade 1 adverse event during needle
placement, as NEEDLE-01's detection and response performance met design
specifications.

## Regulatory Framework References

- ICH E6(R3) Adaption (DOI: 10.5281/zenodo.18973368) - Dimension A basis
- 21 CFR Part 50 Adaption (DOI: 10.5281/zenodo.19040707) - Dimension B basis
- 21 CFR Part 312 Adaption (DOI: 10.5281/zenodo.19057628) - Dimension C basis

## PSL Score Table - End of Hour 08

| # | Robot Type | Dim A | Dim B | Dim C | Per-Robot PSL | Change |
|---|-----------|-------|-------|-------|---------------|--------|
| 1 | Surgical Robots | 6.7 | 6.3 | 6.4 | 6.5 | +0.03 |
| 2 | Cobots | 6.4 | 6.4 | 6.3 | 6.4 | -- |
| 3 | RT Positioning Robots | 6.5 | 6.2 | 6.3 | 6.3 | -- |
| 4 | Needle-Placement Systems | 6.5 | 6.3 | 6.4 | 6.4 | -- |
| 5 | Social Companion Robots | 6.3 | 6.5 | 6.2 | 6.3 | -- |
| 6 | Humanoids | 6.2 | 6.3 | 6.2 | 6.2 | -- |
| 7 | RT Motion-Tracking Robots | 6.5 | 6.4 | 6.4 | 6.4 | -- |
| 8 | Imaging Assistant Robots | 6.4 | 6.5 | 6.3 | 6.4 | +0.03 |
| 9 | Steerable Needle Robots | 6.3 | 6.2 | 6.4 | 6.3 | -- |
| 10 | Rehab Exoskeletons | 6.2 | 6.1 | 6.1 | 6.1 | -- |
| | **Site Total** | **64.0** | **63.2** | **63.0** | **64.3** | **+0.4** |

All 10 robot types remain in the Advanced band (6.0-7.9). The site score
of 64.3 is within the Advanced Site band (60.0-79.9).

## Score Change Details

### Surgical Robots: Dim A +0.1 (Multi-Patient Surgical Awareness)

For the first time in the trial, two surgical robots operate concurrently on
the same cancer type (mediastinal tumors). SURG-01 (operating on P0024 since
07:40) and SURG-02 (operating on P0032 from 08:30) share real-time contextual
data including:

- Tissue plane identification patterns
- Bleeding rate comparisons
- Force application distributions
- Instrument trajectory optimizations
- Anatomical variation alerts

This multi-patient awareness capability directly enhances Dimension A
(omniscient - complete knowledge) per ICH E6(R3) Section 4.2.1 requirements
for real-time data capture and cross-patient learning. The +0.1 increment
reflects the demonstrated ability to leverage concurrent cases for improved
surgical decision support.

### Imaging Assistant Robots: Dim B +0.1 (3 of 4 Bays Active)

Three of four imaging bays are simultaneously active during Hour 08:
- IMAGE-03: PAT-ODMND-0038 (HCC liver assessment)
- IMAGE-04: PAT-ODMND-0040 (colorectal liver mets characterization)
- IMAGE-01: Standby but available (cleared from earlier use)
- IMAGE-02: Scheduled recalibration

With 75% bay activation, the Imaging Assistant robot type demonstrates
improved Dimension B (omnipresent - present everywhere at once) capability.
The +0.1 increment reflects coverage approaching the multi-bay simultaneous
operation design intent per 21 CFR Part 50 Section 50.25 requirements for
continuous patient monitoring capability.

### Needle-Placement Systems: No Change Despite Adverse Event

NEEDLE-01 experienced a Grade 1 adverse event (AE-0008-001) during P0037's
procedure. The PSL score remains unchanged because:

- Dimension A: Bleeding was detected within 1 second (sensor awareness intact)
- Dimension B: System maintained monitoring throughout event (presence intact)
- Dimension C: Automatic pause and position hold functioned correctly
  (capability intact)

The adverse event was within expected procedural variation and the robot
performed within specification at all times. Per PSL scoring methodology
(Section 3.3), score decrements occur only when robot performance falls below
specification, not when expected clinical events occur within normal parameters.

## PSL Dimension Analysis

### Dimension A - Omniscient (Complete Knowledge)

```
DIMENSION A SCORES - END OF HOUR 08
Robot Type              Score  Bar
---------------------  ------  ----------------------------------------
1  Surgical             6.7   [================================|       ]
2  Cobots               6.4   [==============================|         ]
3  RT Positioning       6.5   [===============================|        ]
4  Needle-Placement     6.5   [===============================|        ]
5  Companion            6.3   [=============================|          ]
6  Humanoids            6.2   [============================|           ]
7  RT Tracking          6.5   [===============================|        ]
8  Imaging              6.4   [==============================|         ]
9  Steerable Needle     6.3   [=============================|          ]
10 Rehab Exoskeleton    6.2   [============================|           ]
                        ----
               Dim A Total: 64.0
                              0    2    4    6    8   10
```

Surgical robots lead Dimension A at 6.7, boosted by multi-patient awareness.
The concurrent operation of SURG-01 and SURG-02 on the same tumor type
enables cross-case knowledge sharing that is unique to multi-patient on-demand
operations and not achievable in single-patient scenarios.

### Dimension B - Omnipresent (Present Everywhere at Once)

```
DIMENSION B SCORES - END OF HOUR 08
Robot Type              Score  Bar
---------------------  ------  ----------------------------------------
1  Surgical             6.3   [=============================|          ]
2  Cobots               6.4   [==============================|         ]
3  RT Positioning       6.2   [============================|           ]
4  Needle-Placement     6.3   [=============================|          ]
5  Companion            6.5   [===============================|        ]
6  Humanoids            6.3   [=============================|          ]
7  RT Tracking          6.4   [==============================|         ]
8  Imaging              6.5   [===============================|        ]
9  Steerable Needle     6.2   [============================|           ]
10 Rehab Exoskeleton    6.1   [===========================|            ]
                        ----
               Dim B Total: 63.2
                              0    2    4    6    8   10
```

Companion robots and Imaging Assistant robots share the Dimension B lead at
6.5. Companion robots maintain strong omnipresence with 3 instances
concurrently monitoring pediatric patients (COMPN-03, COMPN-04, and one
in standby coverage mode). Imaging robots achieve 6.5 with 3 of 4 bays active.

### Dimension C - Omnipotent (Ability to Do Anything)

```
DIMENSION C SCORES - END OF HOUR 08
Robot Type              Score  Bar
---------------------  ------  ----------------------------------------
1  Surgical             6.4   [==============================|         ]
2  Cobots               6.3   [=============================|          ]
3  RT Positioning       6.3   [=============================|          ]
4  Needle-Placement     6.4   [==============================|         ]
5  Companion            6.2   [============================|           ]
6  Humanoids            6.2   [============================|           ]
7  RT Tracking          6.4   [==============================|         ]
8  Imaging              6.3   [=============================|          ]
9  Steerable Needle     6.4   [==============================|         ]
10 Rehab Exoskeleton    6.1   [===========================|            ]
                        ----
               Dim C Total: 63.0
                              0    2    4    6    8   10
```

Surgical, Needle-Placement, RT Tracking, and Steerable Needle robots share the
Dimension C lead at 6.4. These are the robot types performing the most
technically demanding interventional procedures, reflecting strong capability
demonstration under clinical conditions.

## Site PSL Trend

```
CUMULATIVE SITE PSL - HOURS 00 THROUGH 08

Hour  PSL   Delta  Bar
----  ----  -----  ----------------------------------------
 00   63.0  --     [=============================|          ]
 01   63.0  +0.0   [=============================|          ]
 02   63.1  +0.1   [=============================|          ]
 03   63.2  +0.1   [=============================|          ]
 04   63.6  +0.4   [==============================|         ]
 05   63.7  +0.1   [==============================|         ]
 06   63.7  +0.0   [==============================|         ]
 07   63.9  +0.2   [==============================|         ]
 08   64.3  +0.4   [==============================|         ]
                    0   20   40   60   80  100

Trajectory: Steady upward trend. Peak morning operations driving
            accelerated PSL improvement through multi-patient
            and multi-robot concurrent utilization.
```

## PSL Scoring Methodology Notes

Per PSL Framework Section 3.3, scores may fluctuate by up to 0.3 points per
dimension per hour. The two +0.1 increments this hour (Surgical Dim A and
Imaging Dim B) are within this limit. Score changes are driven by
demonstrated performance during active procedures, not by robot idle time.

Per-Robot PSL = (Dimension A + Dimension B + Dimension C) / 3
Cumulative Site PSL = Sum of all 10 Per-Robot PSL scores

## USL and Patient Journey References

The Unification Standard Level (USL) framework (Kawchak, 2026;
DOI: 10.5281/zenodo.18778220) provides complementary technical
interoperability scoring. The SURG-01/SURG-02 multi-patient awareness that
drives the Dimension A improvement is enabled by USL cross-robot sharing
capabilities. USL scores in the Advanced band support the real-time data
exchange between concurrent surgical instances.

The single-patient cancer journey framework (Kawchak, 2026;
DOI: 10.5281/zenodo.19119939) established baseline PSL expectations for
individual patient interactions. The multi-patient on-demand context
introduces PSL dimensions not exercised in single-patient scenarios,
particularly multi-robot type coordination and concurrent same-type
operations.
