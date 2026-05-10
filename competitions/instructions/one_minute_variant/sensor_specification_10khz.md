# Sensor Specification: Mixed 10 kHz Force plus 1 kHz Other Channels

This file fixes the canonical sensor record schema for the 1-minute variant. The future Commit 2 sensor specification document expands the per-channel definitions into JSON Schema, Protocol Buffers, and Avro forms. The future session must use the values listed here verbatim.

## Why a Mixed Sample Rate is Required

The 1-minute variant doubles the force sample rate to 10 kHz on each of the 4 arms because the higher robot speeds (1,000 mm per second linear, 360 deg per second angular) require finer force monitoring per IEC 80601-2-77. At 1 kHz force sampling, a 1,000 mm per second tip can advance 1.0 mm between consecutive force samples; at 10 kHz force sampling the tip advances 0.1 mm between samples, which matches the 0.1 mm RMS positioning accuracy budget. The other channel groups remain at 1 kHz because their dynamic range does not require 10 kHz fidelity.

## Channel Group Sample Rate Map (per arm)

| Channel group | Sample rate per arm | Per arm channels | Total channels across 4 arms |
|---------------|---------------------|------------------|------------------------------|
| Joint position, velocity, torque | 1 kHz | 21 | 84 |
| End-effector pose | 1 kHz | 7 | 28 |
| End-effector force, torque | 10 kHz | 6 | 24 |
| Navigation deviation | 1 kHz | 3 | 12 |
| Tool flags and adjuncts | 1 kHz | 7 | 28 |
| Safety enums and metadata | 1 kHz | 6 | 24 |
| Per arm total mixed | mixed | 50 | 200 |

The 21 joint channels comprise 7 position, 7 velocity, and 7 torque per arm (the NeuroSpeed 1.0 has 7 DOF per arm, one more than the 6 DOF ROSA ONE Brain). The 7 end-effector pose channels comprise 3 position (ee_x, ee_y, ee_z) plus 4 quaternion (ee_qw, ee_qx, ee_qy, ee_qz). The 6 end-effector force and torque channels comprise 3 linear force (ee_fx, ee_fy, ee_fz) and 3 torque (ee_tx, ee_ty, ee_tz). The 3 navigation deviation channels are nav_dx, nav_dy, nav_dz from StealthStation S8. The 7 tool flag and adjunct channels are ttip_temp, irr_flow, suc_flow, co2_insuf, us_present, ala_uv, imri_active. The 6 safety enum and metadata channels are estop_state, safety_zone, robot_state, arm_id, heartbeat_ok, plus a tick alignment flag.

## Per-Arm Tick Schema

The mixed sample rate is encoded in the canonical record by emitting 10 force-only ticks for every 1 mixed tick. The mixed tick at 1 kHz carries all 50 channels. The 9 force-only ticks at 10 kHz carry only the 6 force channels plus the per-arm tick alignment flag. The schema therefore has two record kinds.

Record kind A (mixed at 1 kHz): emitted at tick_us in the set 0, 1000, 2000, 3000, ... The record carries all 50 channels for the arm.

Record kind B (force-only at 10 kHz): emitted at tick_us in the set 100, 200, 300, ..., 900, 1100, 1200, ..., 1900, 2100, ..., where tick_us mod 1000 is not zero. The record carries 6 force channels plus the arm_id plus a flag indicating that this is a force-only sample.

## Per-Arm Tick Counts for the 60-Second Window

The 60-second simulation window produces the following tick counts per arm.

| Record kind | Sample rate | Ticks per second | Ticks per 60 s | Channels per tick | Numeric values per arm |
|-------------|-------------|-------------------|-----------------|--------------------|------------------------|
| A mixed | 1 kHz | 1,000 | 60,000 | 50 | 3,000,000 |
| B force-only | 9 kHz (the 9 sub-1 ms force samples per ms) | 9,000 | 540,000 | 7 (6 force + arm_id) | 3,780,000 |
| Mixed total per arm | n/a | 10,000 | 600,000 | mixed | 6,780,000 |

Across 4 arms the total per-iteration L0 numeric value count is 27,120,000 plus the timestamp column.

## Per-Iteration Storage Estimate (Layer 0 raw)

The L0 raw is the canonical full-fidelity record. It is archived to Zenodo and is never committed to Git per `file_size_pyramid_1min.md`.

| Channel rate group | Channels | Ticks (1 min) | Bytes raw | Bytes zstd-3 | Per arm | Per 4 arms |
|--------------------|----------|----------------|-----------|---------------|---------|-------------|
| 1 kHz channels | 44 | 60,000 | 11.0 MB | 2.8 MB | 2.8 MB | 11 MB |
| 10 kHz force channels | 6 | 600,000 | 15.0 MB | 3.8 MB | 3.8 MB | 15 MB |
| L0 raw total | 50 per arm mixed | n/a | 26.0 MB | 6.6 MB | 6.6 MB | 26 MB |

Per-iteration L0 raw at 1 minute with the 4-arm 10 kHz force schema is 26 MB. This is committed to Zenodo, never to Git.

## Channel Inventory (per arm)

The full channel inventory is reproduced from `robot_specification_neurospeed.md` so that this file is self-standing for the future Commit 2 author.

### Group 1: Joint kinematics at 1 kHz (21 channels per arm)

| Channel ID | Quantity | Unit | Sample rate | Resolution |
|------------|----------|------|-------------|------------|
| j1_pos to j7_pos | Joint positions 1 to 7 | radian | 1 kHz | 8.7e-5 rad |
| j1_vel to j7_vel | Joint velocities 1 to 7 | radian per second | 1 kHz | 8.7e-5 rad/s |
| j1_trq to j7_trq | Joint torques 1 to 7 | newton meter | 1 kHz | 0.001 Nm |

### Group 2: End-effector pose at 1 kHz (7 channels per arm)

| Channel ID | Quantity | Unit | Sample rate | Resolution |
|------------|----------|------|-------------|------------|
| ee_x, ee_y, ee_z | End-effector position | millimeter | 1 kHz | 0.01 mm |
| ee_qw, ee_qx, ee_qy, ee_qz | End-effector orientation | unit quaternion | 1 kHz | 1e-5 |

### Group 3: End-effector force and torque at 10 kHz (6 channels per arm)

| Channel ID | Quantity | Unit | Sample rate | Resolution |
|------------|----------|------|-------------|------------|
| ee_fx, ee_fy, ee_fz | End-effector force | newton | 10 kHz | 0.001 N |
| ee_tx, ee_ty, ee_tz | End-effector torque | newton meter | 10 kHz | 0.0001 Nm |

### Group 4: Navigation deviation at 1 kHz (3 channels per arm)

| Channel ID | Quantity | Unit | Sample rate | Resolution |
|------------|----------|------|-------------|------------|
| nav_dx, nav_dy, nav_dz | Navigation deviation from plan | millimeter | 1 kHz | 0.01 mm |

### Group 5: Tool flags and adjuncts at 1 kHz (7 channels per arm)

| Channel ID | Quantity | Unit | Sample rate | Resolution | Notes |
|------------|----------|------|-------------|------------|-------|
| ttip_temp | Tool tip temperature | degree Celsius | 1 kHz | 0.1 C | Thermocouple |
| irr_flow | Irrigation flow rate | mL per minute | 1 kHz | 1 mL/min | Arms 2 and 3 only |
| suc_flow | Suction flow rate | mL per minute | 1 kHz | 1 mL/min | Arm 3 only |
| co2_insuf | CO2 insufflation | n/a | n/a | n/a | Reserved, held at 0.0 |
| us_present | Ultrasound active flag | boolean | 1 kHz | 1 bit | Arm 4 only |
| ala_uv | 5-ALA UV active flag | boolean | 1 kHz | 1 bit | Arm 4 only |
| imri_active | iMRI scan active flag | boolean | 1 kHz | 1 bit | Arm 4 only |

### Group 6: Safety enums and metadata at 1 kHz (6 channels per arm)

| Channel ID | Quantity | Unit | Sample rate | Resolution | Notes |
|------------|----------|------|-------------|------------|-------|
| estop_state | E-stop circuit state | boolean | 1 kHz | 1 bit | 0 nominal, 1 engaged |
| safety_zone | Safety zone classification | enum | 1 kHz | 8 levels | NONE, OUTER, INNER, ELOQUENT, FORBIDDEN, TUMOR_CORE, TUMOR_MARGIN, VESSEL |
| robot_state | Task-order lifecycle state | enum | 1 kHz | 8 levels | IDLE, SETUP, DOCKED, READY, ACTIVE, PAUSE, COMPLETE, ABORT |
| arm_id | Arm identifier | enum | 1 kHz | 4 levels | ARM_1, ARM_2, ARM_3, ARM_4 |
| heartbeat_ok | Inter-arm heartbeat status flag | boolean | 1 kHz | 1 bit | 1 nominal, 0 missed |
| tick_align_flag | Mixed tick alignment flag | boolean | 1 kHz | 1 bit | 1 if this tick carries all 50 channels, 0 if force-only |

## Schema Output for the Future Commit 2

The future Commit 2 author must produce three machine-readable schemas mirroring the 200-channel inventory above. The recommended schema layout uses a per-arm record with arm_id as a discriminator rather than a wide 200-column record. This keeps the per-record size small and lets the streaming consumer dispatch records per arm.

- `schemas/sensor_record_4arm.schema.json`. JSON Schema 2020-12. One record per arm per tick. Required keys: `tick_us` (microsecond timestamp), `arm_id` (enum ARM_1 to ARM_4), `record_kind` (enum MIXED at 1 kHz or FORCE_ONLY at 10 kHz), all 50 per-arm channels for MIXED records or 6 force channels for FORCE_ONLY records, plus `meta_seed` integer and `meta_iteration_id` string.
- `schemas/sensor_record_4arm.proto`. Protocol Buffers 3. Same schema as above with reserved field numbers 60 to 99 for future expansion.
- `schemas/sensor_record_4arm.avsc`. Apache Avro JSON. Same schema with explicit enum types for arm_id, record_kind, safety_zone, robot_state.

## Validation Rules (per record)

The future Commit 2 ingest script must enforce the following validation rules.

- `tick_us` is a non-negative integer in [0, 60_000_000].
- `arm_id` is one of ARM_1, ARM_2, ARM_3, ARM_4.
- `record_kind` is one of MIXED, FORCE_ONLY.
- For MIXED records, `tick_us` mod 1000 equals 0.
- For FORCE_ONLY records, `tick_us` mod 100 equals 0 and `tick_us` mod 1000 is not 0.
- All force values fall within +/- 50 N.
- All torque values fall within +/- 5 Nm.
- The cumulative ee_fx plus ee_fy plus ee_fz across all 4 arms at the same tick_us must remain under 12 N per `multi_arm_coordination.md`.
- `safety_zone` is one of NONE, OUTER, INNER, ELOQUENT, FORBIDDEN, TUMOR_CORE, TUMOR_MARGIN, VESSEL.
- `robot_state` is one of IDLE, SETUP, DOCKED, READY, ACTIVE, PAUSE, COMPLETE, ABORT.

## Stream Framing

Each record is a single line in JSONL or a length-prefixed binary frame in Protocol Buffers. The 4 arms are multiplexed onto a single stream by interleaving records ordered by `tick_us` ascending and by `arm_id` ascending within the same `tick_us`. At 1 kHz mixed plus 9 kHz force-only the stream emits 4 * 10,000 equals 40,000 records per second across all 4 arms.

## Failure Handling

- Dropped tick reconstruction policy: missing FORCE_ONLY ticks within a 1 ms window are linearly interpolated from the surrounding samples; missing MIXED ticks are flagged as a gap and trigger an emergency arm-park per `multi_arm_coordination.md`.
- Gap detection: any inter-arrival time between consecutive ticks that exceeds 200 microseconds for FORCE_ONLY or 1.5 ms for MIXED triggers a gap report log entry.
- Gap report log: `logs/sensor_gap_report.jsonl` with one record per detected gap.

## Source Files Cited

- `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md`. Source for the per-arm sensor channel inventory and the per-arm sample rates. The 50-channel-per-arm count and the 1 kHz mixed plus 10 kHz force scheme are inherited verbatim.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the cumulative force limit and the heartbeat_ok channel.
- `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`. Source for the L0 raw vs L1 to L3 pyramid that this 26 MB per iteration L0 raw feeds into.
- `competitions/instructions/robot_specification.md`. Source for the parent ROSA ONE Brain v3.0 50-channel-per-robot inventory whose structure this 1-minute variant extends to per-arm.
- `patient-journey/stage_05_surgery.py`. Source for the IEC 80601-2-77 force limits and the 1 kHz runtime safety monitoring cadence that the 10 kHz force channel rate exceeds.
