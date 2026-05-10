# Robot Specification: Medtronic NeuroSpeed 1.0 (Hypothetical 2030)

This file fixes the make, model, kinematics, sensor suite, and safety limits of the surgical robot platform that every future commit references for the 1-minute variant. The future session must use the values listed here verbatim. Iteration sweeps may vary controller gains and sensor noise within the documented tolerances; iteration sweeps may not vary make, model, kinematic limits, or safety limits.

## Why a Future Robot is Required

The current SOTA Medtronic ROSA ONE Brain v3.0 cannot perform a 1-minute glioblastoma resection. The constraint table below shows that every key parameter falls short by 5 to 200 times. The 1-minute variant therefore specifies a hypothetical 2030 product, the Medtronic NeuroSpeed 1.0, that meets the requirement.

| Constraint | ROSA ONE Brain v3.0 (current SOTA) | Required for 1-minute surgery | Gap |
|------------|------------------------------------|-------------------------------|-----|
| Tumor volume to remove | n/a | 38,800 mm cubed (4.2 cm sphere) | n/a |
| Required removal rate | n/a | 647 mm cubed per second mean | n/a |
| Standard CUSA aspirator rate | 2 to 5 mm cubed per second | 700 mm cubed per second mean | 200 times slower |
| End-effector linear velocity | 50 mm per second | 1,000 mm per second | 20 times slower |
| End-effector linear acceleration | 200 mm per second squared | 10,000 mm per second squared | 50 times slower |
| Joint angular velocity | 30 to 180 deg per second | 360 deg per second | 2 times slower at fastest joint |
| E-stop latency | 50 ms | 5 ms | 10 times slower |
| Positioning accuracy at speed | 0.5 mm RMS | 0.1 mm RMS | 5 times worse |
| Force resolution | 0.01 N | 0.001 N | 10 times coarser |
| Continuous duty cycle | hours | 1 minute peak | n/a |

## Primary Surgical Robot

- Make: Medtronic in partnership with Boston Dynamics
- Model: NeuroSpeed 1.0
- Hardware revision: v1.0 (hypothetical 2030 product; illustrative for the simulation)
- Firmware: 1.0.0 (illustrative; the simulation is for research)
- Class: multi-arm parallel stereotactic neurosurgical robot (new regulatory class)
- Configuration: 4 cooperating arms with 7 degrees of freedom each, 28 DOF total. Each arm is redundant for collision avoidance.
- Workspace: 0.5 m radius hemisphere centered on the surgical target. Tighter than ROSA but sufficient for a single resection.
- Mounting: floor-mounted base, 4 articulating arms with overhead boom rail. Operates around the Mayfield clamp without obstructing the surgeon's view.
- Footprint: 1.4 m by 1.0 m base plus boom. Larger than ROSA.
- Weight: 480 kg, 1.7 times ROSA.
- Power: 8 kW peak, 13 times ROSA.
- Cooling: liquid nitrogen for high-speed actuators. Required for sustained 10 kHz duty.
- Continuous operation duration: 5 minutes peak, set by the liquid nitrogen cooling cycle limit.
- Regulatory clearance: FDA Breakthrough Device, hypothetical De Novo class III submission. Illustrative; the simulation is for research.

## Kinematics (per arm)

- Degrees of freedom per arm: 7
- Total DOF across 4 arms: 28
- Linear positioning accuracy at peak speed: 0.1 mm root mean square at the end effector. 5 times better than ROSA.
- Linear positioning resolution: 0.01 mm
- Angular positioning accuracy: 0.05 degree per joint root mean square
- Angular positioning resolution: 0.005 degree per joint
- Maximum end-effector linear velocity: 1,000 mm per second. 20 times ROSA.
- Maximum end-effector linear acceleration: 10,000 mm per second squared. 50 times ROSA.
- Maximum joint angular velocity: 360 degrees per second. 2 times ROSA fastest joint.
- Maximum joint angular acceleration: 1,800 degrees per second squared. Matched to angular velocity.

## End Effector and Per-Arm Tool Assignment

The four arms carry different tools. Tool changeover during the 60-second procedure is forbidden because the changeover would consume more than 5 seconds of the budget. The future Commit 1 architecture document fixes the assignment below.

| Arm | Tool | Primary task | Force sensor sample rate | Command sample rate |
|-----|------|--------------|--------------------------|---------------------|
| 1 | Hybrid ultrasonic plus waterjet plus pulsed plasma | Bulk tumor resection (Phase 2 at 800 mm cubed per second peak) and fine margin resection (Phase 3 at 200 mm cubed per second) | 10 kHz | 1 kHz |
| 2 | Bipolar coagulation plus irrigation | Real-time hemostasis behind arm 1 | 10 kHz | 1 kHz |
| 3 | Suction plus tissue collection | Continuous removal of debris and tissue collection for downstream margin pathology | 10 kHz | 100 Hz |
| 4 | 0.5 T MRI plus 5-ALA fluorescence camera plus ultrasound probe | Continuous margin imaging at 30 fps Phase 1 to 2, 100 fps Phase 3 | 10 kHz | 1 kHz |

## Hybrid Tissue Removal Mechanism (Arm 1)

The arm 1 tool combines three established techniques to reach 800 mm cubed per second peak removal:

- Ultrasonic vibration at 23 kHz fragments soft tumor tissue, mirroring the existing CUSA mechanism but at higher amplitude.
- Waterjet at 0.5 mm orifice and 100 bar provides hydrodissection ahead of the ultrasonic tip.
- Pulsed plasma at 5 microsecond pulses at 1 kHz repetition rate coagulates the cavity wall as the tip advances.

Tissue removal peak rate: 800 mm cubed per second. Sustained rate over the 40-second Phase 2 window: 700 mm cubed per second mean. The hybrid mechanism is novel and combines aspiration, hydrodissection, and coagulation in one tool.

## Sensor Suite (per arm)

The future Commit 2 sensor specification document expands each channel below into a JSON Schema and a Protocol Buffers message definition. The channel list below is the canonical source for one arm; the four-arm total is 200 channels. The detailed mixed-rate specification lives in `sensor_specification_10khz.md`.

| Channel ID | Quantity | Unit | Sample rate | Resolution | Notes |
|------------|----------|------|-------------|------------|-------|
| j1_pos to j7_pos | Joint positions 1 to 7 | radian | 1 kHz | 8.7e-5 rad | Encoder readings, 7 DOF arm |
| j1_vel to j7_vel | Joint velocities 1 to 7 | radian per second | 1 kHz | 8.7e-5 rad/s | Filtered first derivative |
| j1_trq to j7_trq | Joint torques 1 to 7 | newton meter | 1 kHz | 0.001 Nm | Strain gauge, 10 times finer than ROSA |
| ee_x, ee_y, ee_z | End-effector position | millimeter | 1 kHz | 0.01 mm | Forward kinematics output |
| ee_qw, ee_qx, ee_qy, ee_qz | End-effector orientation | unit quaternion | 1 kHz | 1e-5 | Forward kinematics output |
| ee_fx, ee_fy, ee_fz | End-effector force | newton | 10 kHz | 0.001 N | Wrist 6-axis sensor at 10x ROSA rate and 10x finer resolution |
| ee_tx, ee_ty, ee_tz | End-effector torque | newton meter | 10 kHz | 0.0001 Nm | Wrist 6-axis sensor at 10x ROSA rate |
| nav_dx, nav_dy, nav_dz | Navigation deviation from plan | millimeter | 1 kHz | 0.01 mm | StealthStation S8 cross-stream |
| ttip_temp | Tool tip temperature | degree Celsius | 1 kHz | 0.1 C | Thermocouple |
| irr_flow | Irrigation flow rate | milliliter per minute | 1 kHz | 1 mL/min | Inline turbine; arms 2 and 3 only, 0.0 on others |
| suc_flow | Suction flow rate | milliliter per minute | 1 kHz | 1 mL/min | Inline turbine; arm 3 only, 0.0 on others |
| co2_insuf | CO2 insufflation | not used in cranial | n/a | n/a | Channel reserved, value held at 0.0 |
| us_present | Intraoperative ultrasound active flag | boolean | 1 kHz | 1 bit | 0 or 1; arm 4 only |
| ala_uv | 5-ALA UV illumination active flag | boolean | 1 kHz | 1 bit | 0 or 1; arm 4 only |
| imri_active | Intraoperative MRI scan active flag | boolean | 1 kHz | 1 bit | 0 or 1; arm 4 only |
| estop_state | E-stop circuit state | boolean | 1 kHz | 1 bit | 0 = nominal, 1 = engaged |
| safety_zone | Safety zone classification | enum | 1 kHz | 8 levels | NONE, OUTER, INNER, ELOQUENT, FORBIDDEN, etc. |
| robot_state | Task-order lifecycle state | enum | 1 kHz | 8 levels | IDLE, SETUP, DOCKED, READY, ACTIVE, PAUSE, COMPLETE, ABORT |
| arm_id | Arm identifier | enum | 1 kHz | 4 levels | ARM_1, ARM_2, ARM_3, ARM_4 |
| heartbeat_ok | Inter-arm heartbeat status flag | boolean | 1 kHz | 1 bit | 1 = nominal heartbeat, 0 = missed |

Per-arm channel count: 50 channels. Across 4 arms the total is 200 channels.

## Safety Limits (per arm and aggregate)

The future Commit 3 control loop must enforce the limits below per arm. Limits are inherited from `patient-journey/stage_05_surgery.py` and IEC 80601-2-77 with adjustments for higher-precision tracking. Iteration sweeps may not relax these limits.

- Tip force limit per arm: 5.0 N. Lower than ROSA's 15.0 N because higher-precision tracking enables tighter tolerance.
- Lateral force limit per arm: 1.0 N. Lower than ROSA's 5.0 N for the same reason.
- Cumulative tip force across all 4 arms on patient frame: 12 N maximum.
- E-stop latency limit: 5 ms. 10 times faster than ROSA. Enforced by the deterministic real-time bus described in `multi_arm_coordination.md`.
- Forbidden operations: autonomous tissue cutting outside the planned tumor volume, autonomous vessel ligation, operation beyond per-arm workspace boundary, operation with E-stop disabled, simultaneous resection and imaging in the same 1 mm cubed voxel.
- Eloquent cortex stand-off: 2.0 mm minimum end-effector distance to motor strip and Broca area boundary as defined by preoperative tractography. Tighter than ROSA's 5 mm because of 5x positioning accuracy.
- Maximum continuous active duration without 5-second pause and re-confirmation: 60 s per arm (covers the entire procedure).

## Coordinate Frame (shared across all 4 arms)

- World frame origin: Mayfield clamp pin midpoint
- Positive X: patient left
- Positive Y: patient anterior
- Positive Z: patient superior
- Units: millimeters for translation, radians for rotation
- Quaternion convention: scalar-first (qw, qx, qy, qz)

All four arms share the world frame so that the cross-arm safety zone gating documented in `multi_arm_coordination.md` can compare arm tip positions in a common coordinate space.

## AI Decision and Imaging Subsystems

- AI decision rate per arm: 1 kHz adaptive trajectory replanning. Currently 1 Hz human-in-loop on ROSA. The 1 kHz cadence is required to keep up with the 1,000 mm per second end-effector velocity.
- Real-time imaging: 0.5 T MRI at 30 frames per second on arm 4. 100 times the current 0.3 fps iMRI rate.
- 5-ALA fluorescence imaging: 100 frames per second on arm 4 during Phase 3. 33 times the current 3 fps clinical standard.

## Companion Equipment Cited

- StealthStation S8 (Medtronic): preoperative MRI registration and intraoperative navigation source for the per-arm `nav_dx`, `nav_dy`, `nav_dz` channels.
- Mayfield clamp: head fixation. Position frozen at simulation start.
- iMRI 0.5 T (hypothetical 2030 high-frame-rate variant): drives the per-arm-4 `imri_active` flag and the 30 fps imaging stream during Phase 1 and Phase 2.
- ROBO ALA-560 ultraviolet illumination unit: drives the per-arm-4 `ala_uv` flag.
- Boston Dynamics ATLAS-derived parallel arm controller: drives the 360 deg per second joint angular velocity per arm.

## Source Files Cited

- `competitions/instructions/robot_specification.md`. Source for the parent ROSA ONE Brain v3.0 specification that this 1-minute variant exceeds across every constraint dimension. The parent specification is preserved unchanged for the v3.9.0 1-hour scenario.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the cross-arm coordination protocol, the 5 ms emergency arm-park trigger, and the cumulative force limit.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the mixed 10 kHz force plus 1 kHz other channel rates that the per-arm sensor suite implements.
- `patient-journey/stage_05_surgery.py`. `FORCE_LIMIT_TIP_N`, `FORCE_LIMIT_LATERAL_N`, `ESTOP_LATENCY_LIMIT_MS`, `TASK_ORDER_STATES`, `FORBIDDEN_OPERATIONS`. The 1-minute variant tightens the per-arm tip force limit to 5.0 N (from 15.0 N) and the E-stop latency limit to 5 ms (from 50 ms).
- `new-trial/README.md`. Reference list for the 10 robot type taxonomy used by the parent repository. The NeuroSpeed 1.0 belongs to a new multi-arm parallel stereotactic neurosurgical class that does not yet appear in the taxonomy.
- `national-platform/usl_standard/`. Source for the USL scoring scale that the future Commit 5 metric uses to assign a quality score to each iteration of this 1-minute variant.
