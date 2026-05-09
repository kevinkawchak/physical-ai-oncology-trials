# Robot Specification

This file fixes the make, model, kinematics, sensor suite, and safety limits of the surgical robot platform that every future commit references. The future session must use the values listed here verbatim. Iteration sweeps may vary controller gains and sensor noise within the documented tolerances; iteration sweeps may not vary make, model, kinematic limits, or safety limits.

## Primary Surgical Robot

- Make: Medtronic
- Model: ROSA ONE Brain
- Hardware revision: v3.0
- Firmware: 3.1.4
- Class: stereotactic neurosurgical robot
- Mounting: floor-mounted base, articulating arm, head holder integration via Mayfield clamp adapter
- Footprint: 0.95 m by 0.65 m base
- Weight: 280 kg
- Power: 100 to 240 V AC, 50 to 60 Hz, 600 W peak
- Regulatory clearance: FDA 510(k) K201246 (illustrative; the simulation is for research)

## Kinematics

- Degrees of freedom: 6
- Workspace: 1.0 m radius hemisphere centered on the tool changer
- Linear positioning accuracy: 0.5 mm root mean square at the end effector
- Linear positioning resolution: 0.05 mm
- Angular positioning accuracy: 0.1 degree per joint root mean square
- Angular positioning resolution: 0.01 degree per joint
- Maximum end-effector linear velocity: 50 mm per second
- Maximum end-effector linear acceleration: 200 mm per second squared
- Maximum joint angular velocity: 30 degrees per second
- Maximum joint angular acceleration: 120 degrees per second squared

## End Effector

- Tool changer: ISO 9409-1-50-4-M6 flange
- Operative tools used in this simulation:
  - Stealth Autoguide (Medtronic) for stereotactic biopsy needle placement during phase 1
  - Bipolar forceps adapter for hemostasis during phases 2 to 5
  - Suction-irrigation adapter active throughout
  - Surgical microscope coupling for Modus V (Synaptive Medical) digital exoscope visualization

## Sensor Suite

The future Commit 2 sensor specification document expands each channel below into a JSON Schema and a Protocol Buffers message definition. The channel list below is the canonical source.

| Channel ID | Quantity | Unit | Sample rate | Resolution | Notes |
|------------|----------|------|-------------|------------|-------|
| j1_pos to j6_pos | Joint positions 1 to 6 | radian | 1 kHz | 1.7e-4 rad | Encoder readings |
| j1_vel to j6_vel | Joint velocities 1 to 6 | radian per second | 1 kHz | 1.7e-4 rad/s | Filtered first derivative |
| j1_trq to j6_trq | Joint torques 1 to 6 | newton meter | 1 kHz | 0.01 Nm | Strain gauge |
| ee_x, ee_y, ee_z | End-effector position | millimeter | 1 kHz | 0.01 mm | Forward kinematics output |
| ee_qw, ee_qx, ee_qy, ee_qz | End-effector orientation | unit quaternion | 1 kHz | 1e-5 | Forward kinematics output |
| ee_fx, ee_fy, ee_fz | End-effector force | newton | 1 kHz | 0.01 N | Wrist 6-axis sensor |
| ee_tx, ee_ty, ee_tz | End-effector torque | newton meter | 1 kHz | 0.001 Nm | Wrist 6-axis sensor |
| nav_dx, nav_dy, nav_dz | Navigation deviation from plan | millimeter | 1 kHz | 0.01 mm | StealthStation S8 cross-stream |
| ttip_temp | Tool tip temperature | degree Celsius | 1 kHz | 0.1 C | Thermocouple |
| irr_flow | Irrigation flow rate | milliliter per minute | 1 kHz | 1 mL/min | Inline turbine |
| suc_flow | Suction flow rate | milliliter per minute | 1 kHz | 1 mL/min | Inline turbine |
| co2_insuf | CO2 insufflation | not used in cranial | n/a | n/a | Channel reserved, value held at 0.0 |
| us_present | Intraoperative ultrasound active flag | boolean | 1 kHz | 1 bit | 0 or 1 |
| ala_uv | 5-ALA UV illumination active flag | boolean | 1 kHz | 1 bit | 0 or 1 |
| imri_active | Intraoperative MRI scan active flag | boolean | 1 kHz | 1 bit | 0 or 1 |
| estop_state | E-stop circuit state | boolean | 1 kHz | 1 bit | 0 = nominal, 1 = engaged |
| safety_zone | Safety zone classification | enum | 1 kHz | 8 levels | NONE, OUTER, INNER, ELOQUENT, FORBIDDEN, etc. |
| robot_state | Task-order lifecycle state | enum | 1 kHz | 8 levels | IDLE, SETUP, DOCKED, READY, ACTIVE, PAUSE, COMPLETE, ABORT |

Total per-tick channel count: 50 channels. At 1 kHz across 1 hour, the canonical 1-hour sensor record set is 50 channels times 3,600,000 ticks equals 180,000,000 numeric values plus the millisecond timestamp column.

## Safety Limits

The future Commit 3 control loop must enforce the limits below. Limits are inherited from `patient-journey/stage_05_surgery.py` and IEC 80601-2-77. Iteration sweeps may not relax these limits.

- Tip force limit: 15.0 N
- Lateral force limit: 5.0 N
- E-stop latency limit: 50 ms
- Forbidden operations: autonomous tissue cutting without human confirmation, autonomous vessel ligation, operation beyond workspace boundary, operation with E-stop disabled
- Eloquent cortex stand-off: 5.0 mm minimum end-effector distance to motor strip and Broca area boundary as defined by preoperative tractography
- Maximum continuous active duration without 5-second pause and re-confirmation: 120 s

## Coordinate Frame

- World frame origin: Mayfield clamp pin midpoint
- Positive X: patient left
- Positive Y: patient anterior
- Positive Z: patient superior
- Units: millimeters for translation, radians for rotation
- Quaternion convention: scalar-first (qw, qx, qy, qz)

## Companion Equipment Cited

- StealthStation S8 (Medtronic): preoperative MRI registration and intraoperative navigation source for `nav_dx`, `nav_dy`, `nav_dz` channels.
- Modus V (Synaptive Medical): robotic digital exoscope for visualization. Not commanded by this simulation; included in the facility ASCII diagram only.
- iMRI 0.55 T (Siemens MAGNETOM Free.Max sample model): one intraoperative MRI scan at 30-minute mark; sensor channel `imri_active` toggles to 1 for the 90-second scan window.
- ROBO ALA-560 ultraviolet illumination unit: drives the `ala_uv` flag.

## Source Files Cited

- `patient-journey/stage_05_surgery.py`. `FORCE_LIMIT_TIP_N`, `FORCE_LIMIT_LATERAL_N`, `ESTOP_LATENCY_LIMIT_MS`, `TASK_ORDER_STATES`, `FORBIDDEN_OPERATIONS`.
- `new-trial/README.md`. Reference list for the 10 robot type taxonomy used by the parent repository (this v3.9.0 simulation focuses on a single robot in robot type 3, RT positioning robots, repurposed for stereotactic neurosurgery).
- `national-platform/usl_standard/`. Source for the USL scoring scale that the future Commit 5 metric uses to assign a quality score to each iteration.
