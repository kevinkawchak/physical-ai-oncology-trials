# Commit 3 (1-Minute Variant): Sensor to XYZ Mapping for 4 Cooperating Arms

This file specifies the files the future Claude Code Opus 4.7 1M Max session must author in its third commit for the 1-minute variant. The session must author exactly the files listed and must not author additional files in this commit. The parent `competitions/instructions/commit_03_xyz_mapping.md` lists 9 files for the v3.9.0 1-hour scenario. This 1-minute variant lists 11 files because the 4-arm topology adds the heartbeat coordination files.

## Goal

Define the deterministic transformation that converts each per-arm sensor record into a per-arm Cartesian (x, y, z) end-effector command for the Medtronic NeuroSpeed 1.0. Author the mapping documentation, command schemas, kinematics configuration, the Python mapper script, the C++ real-time control loop for 4 arms, the C++ heartbeat coordination layer, the canonical Zenodo-pointer for the per-arm xyz command stream, the human-review CSV samples per arm, and the ASCII visualization of the per-arm traversed paths.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1min-trial/docs/coordinate_mapping.md` | Markdown | Hand-authored | 24 KB |
| 2 | `competitions/glioblastoma-1min-trial/docs/multi_arm_coordination.md` | Markdown | Hand-authored | 14 KB |
| 3 | `competitions/glioblastoma-1min-trial/schemas/xyz_command_4arm.schema.json` | JSON Schema 2020-12 | Hand-authored | 10 KB |
| 4 | `competitions/glioblastoma-1min-trial/schemas/xyz_command_4arm.proto` | Protocol Buffers 3 | Hand-authored | 4 KB |
| 5 | `competitions/glioblastoma-1min-trial/config/kinematics_4arm.yaml` | YAML | Hand-authored | 14 KB |
| 6 | `competitions/glioblastoma-1min-trial/src/mapping/sensor_to_xyz_4arm.py` | Python 3.10 | Hand-authored | 22 KB |
| 7 | `competitions/glioblastoma-1min-trial/src/control/robot_loop_4arm.cpp` | C++20 | Hand-authored | 18 KB |
| 8 | `competitions/glioblastoma-1min-trial/src/coordination/arm_heartbeat.cpp` | C++20 | Hand-authored | 12 KB |
| 9 | `competitions/glioblastoma-1min-trial/data/xyz_trace_sample_arm1.csv` through `_arm4.csv` | CSV | Script-generated | 25 KB each (100 KB total) |
| 10 | `competitions/glioblastoma-1min-trial/viz/xyz_path_4arm.txt` | ASCII text | Script-generated | 8 KB |
| 11 | `competitions/glioblastoma-1min-trial/data/xyz_trace_4arm.zenodo_pointer.json` | JSON | Hand-authored at this commit, populated at Commit 5 | 1 KB |

The full per-arm xyz command stream lives on Zenodo per `zenodo_archive_protocol.md` because at 4 arms times 1 kHz times 60 s the joined sensor plus xyz Parquet exceeds the 5 MB committed Parquet cap. The Zenodo pointer at File 11 holds the SHA-256 and DOI; the pointer is populated at Commit 5 after the Zenodo deposition completes.

## File 1: docs/coordinate_mapping.md

Required sections:

1. Mapping rule overview: each per-arm `tick_us` MIXED record produces zero or one `XYZCommand` for that arm. FORCE_ONLY records do not produce commands; they update the local force monitor only.
2. Phase-conditioned mapping per arm:
   - Phase 1 (0 to 5 s): all 4 arms emit position-hold commands; only arm 4 emits image-trigger commands at 30 Hz.
   - Phase 2 (5 to 45 s): arm 1 emits 1 kHz cut commands; arm 2 emits 1 kHz coagulate commands; arm 3 emits 100 Hz suction commands; arm 4 emits 30 Hz image commands.
   - Phase 3 (45 to 55 s): arm 1 reduces commanded velocity by 75 percent; arm 4 increases image cadence to 100 Hz.
   - Phase 4 (55 to 60 s): arms 1 and 3 emit retract trajectories; arm 2 emits final hemostasis pass; arm 4 emits final margin scan.
3. Forward kinematics per arm: 7-DOF DH parameter table for the NeuroSpeed 1.0 arm. Joint twist, joint length, joint offset, joint angle for each of the seven joints. The DH parameters live in `config/kinematics_4arm.yaml`.
4. Inverse kinematics per arm: numerical 7-DOF solver with 6-DOF redundancy used for collision avoidance. The solver uses Levenberg-Marquardt with a 0.1 mm tolerance and a 5 microsecond per-call wall budget on the conventional high-end server profile.
5. Per-arm safety zone gating: commands inside the FORBIDDEN safety zone are clamped to the boundary and trigger the cross-arm emergency-park per `multi_arm_coordination.md`; commands inside the ELOQUENT safety zone are slowed to 25 percent of nominal velocity; commands inside the TUMOR_CORE zone proceed at nominal velocity.
6. Per-arm force feedback fusion: the mapper reads the most recent 10 kHz force sample and clamps commanded velocity if force exceeds 80 percent of the per-arm 5.0 N tip force limit (4.0 N tip).
7. Cumulative force enforcement across all 4 arms: the mapper reads the per-arm force frames from the heartbeat broadcast and clamps each arm's commanded velocity proportionally if the cumulative exceeds 11.0 N (1 N margin under the 12 N cap).
8. Command latency budget per arm: 1 ms end-to-end from sensor sample arrival to the first commanded actuator update. Of that budget, 0.1 ms is reserved for the inverse kinematics solve, 0.5 ms for the cross-arm coordination read, and 0.4 ms for the actuator write. The 1 ms budget is 5 times tighter than the parent v3.9.0 5 ms budget because the 1,000 mm per second arm velocity demands 5 times finer command quantization.
9. Cross-references to schemas, to `src/control/robot_loop_4arm.cpp`, and to `src/coordination/arm_heartbeat.cpp`.

## File 2: docs/multi_arm_coordination.md

The future session must embed a verbatim copy of `competitions/instructions/one_minute_variant/multi_arm_coordination.md` into the output tree at `docs/multi_arm_coordination.md`. The embedded copy lets the simulation be self-standing once the future session ships it, without requiring readers to navigate back to the instruction set.

## File 3: schemas/xyz_command_4arm.schema.json

JSON Schema 2020-12 with the following structure. The schema differs from the parent v3.9.0 `xyz_command.schema.json` by adding an `arm_id` discriminator and by tightening the `linear_vel_mmps` upper bound to 1,000 mm per second (20 times higher than the parent ROSA 50 mm per second cap):

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kevinkawchak.github.io/physical-ai-oncology-trials/v3.9.1/xyz_command_4arm.schema.json",
  "title": "GBM 1min XYZ Command per Arm",
  "type": "object",
  "required": ["tick_us", "arm_id", "x_mm", "y_mm", "z_mm",
               "qw", "qx", "qy", "qz",
               "linear_vel_mmps", "force_clamp_N", "tool",
               "command_state", "phase_id",
               "meta_seed", "meta_iteration_id"],
  "properties": {
    "tick_us": {"type": "integer", "minimum": 0, "maximum": 60000000},
    "arm_id": {"type": "string", "enum": ["ARM_1", "ARM_2", "ARM_3", "ARM_4"]},
    "x_mm": {"type": "number", "minimum": -500, "maximum": 500},
    "y_mm": {"type": "number", "minimum": -500, "maximum": 500},
    "z_mm": {"type": "number", "minimum": -500, "maximum": 500},
    "qw": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "qx": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "qy": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "qz": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "linear_vel_mmps": {"type": "number", "minimum": 0, "maximum": 1000.0},
    "force_clamp_N": {"type": "number", "minimum": 0, "maximum": 5.0},
    "tool": {"type": "string", "enum": ["HYBRID_USP", "BIPOLAR_IRR", "SUCTION_COL", "IMG_5ALA_MRI", "NONE"]},
    "command_state": {"type": "string", "enum": ["EMIT", "CLAMP_TO_BOUNDARY", "FORCE_HOLD", "FORCE_SHARE_CLAMP", "SAFETY_PAUSE", "EMERGENCY_PARK", "ABORT"]},
    "phase_id": {"type": "integer", "minimum": 1, "maximum": 4},
    "meta_seed": {"type": "integer"},
    "meta_iteration_id": {"type": "string", "pattern": "^run_[0-9]{5}$"}
  },
  "additionalProperties": false
}
```

The `linear_vel_mmps` upper bound enforces the 1,000 mm per second end-effector velocity limit from `robot_specification_neurospeed.md`. The `force_clamp_N` upper bound enforces the per-arm 5.0 N tip force limit. The `command_state` enum adds two new values relative to the parent: `FORCE_SHARE_CLAMP` (cumulative force exceeded 11.0 N, this arm is throttled proportionally) and `EMERGENCY_PARK` (cross-arm safety zone gating triggered the 5 ms park sequence).

## File 4: schemas/xyz_command_4arm.proto

Protocol Buffers 3 definition with the same fields as the JSON Schema. Field numbers 1 through 16 cover the required keys; numbers 100 and 101 cover the metadata fields. Reserve numbers 17 through 99 for future expansion. The proto file is consumed by the C++ control loop; the future session must verify generated C++ code compiles cleanly with `protoc --cpp_out=build/proto schemas/xyz_command_4arm.proto`.

## File 5: config/kinematics_4arm.yaml

Required keys (the 7-DOF DH parameters and per-arm joint limits):

```
---
robot:
  make: Medtronic
  model: NeuroSpeed 1.0
  hardware_revision: v1.0
  firmware: "1.0.0"
  arms: 4
  dof_per_arm: 7

dh_parameters_per_arm:
  joint_1:
    twist_rad: 0.0
    length_mm: 0.0
    offset_mm: 200.0
    angle_offset_rad: 0.0
  joint_2:
    twist_rad: -1.5707963
    length_mm: 0.0
    offset_mm: 0.0
    angle_offset_rad: -1.5707963
  joint_3:
    twist_rad: 0.0
    length_mm: 180.0
    offset_mm: 0.0
    angle_offset_rad: 0.0
  joint_4:
    twist_rad: -1.5707963
    length_mm: 60.0
    offset_mm: 200.0
    angle_offset_rad: 0.0
  joint_5:
    twist_rad: 1.5707963
    length_mm: 0.0
    offset_mm: 0.0
    angle_offset_rad: 0.0
  joint_6:
    twist_rad: -1.5707963
    length_mm: 0.0
    offset_mm: 50.0
    angle_offset_rad: 0.0
  joint_7:
    twist_rad: 0.0
    length_mm: 0.0
    offset_mm: 50.0
    angle_offset_rad: 0.0

joint_limits_per_arm:
  joint_1: {min_rad: -3.05, max_rad: 3.05, max_vel_radps: 6.28, max_acc_radpss: 31.42}
  joint_2: {min_rad: -2.09, max_rad: 2.09, max_vel_radps: 6.28, max_acc_radpss: 31.42}
  joint_3: {min_rad: -2.97, max_rad: 2.97, max_vel_radps: 6.28, max_acc_radpss: 31.42}
  joint_4: {min_rad: -3.14, max_rad: 3.14, max_vel_radps: 6.28, max_acc_radpss: 31.42}
  joint_5: {min_rad: -2.09, max_rad: 2.09, max_vel_radps: 6.28, max_acc_radpss: 31.42}
  joint_6: {min_rad: -6.28, max_rad: 6.28, max_vel_radps: 6.28, max_acc_radpss: 31.42}
  joint_7: {min_rad: -6.28, max_rad: 6.28, max_vel_radps: 6.28, max_acc_radpss: 31.42}

end_effector:
  max_linear_vel_mmps: 1000.0
  max_linear_acc_mmpss: 10000.0
  max_tip_force_N_per_arm: 5.0
  max_lateral_force_N_per_arm: 1.0
  max_cumulative_force_N: 12.0

control_loop:
  rate_hz: 1000
  force_sample_rate_hz: 10000
  estop_latency_limit_ms: 5
  command_to_actuator_budget_ms: 1
  heartbeat_rate_hz: 1000
  heartbeat_miss_threshold_frames: 3

per_arm_workspace_sectors:
  arm_1: {azimuth_min_deg: 0, azimuth_max_deg: 90, hemisphere: full}
  arm_2: {azimuth_min_deg: 270, azimuth_max_deg: 360, hemisphere: full}
  arm_3: {azimuth_min_deg: 90, azimuth_max_deg: 270, hemisphere: lower}
  arm_4: {azimuth_min_deg: 0, azimuth_max_deg: 360, hemisphere: upper}

inter_arm_collision:
  min_distance_mm: 8.0
```

The kinematics file must be `yamllint -d relaxed` clean and must pass a unit-conversion sanity check (degrees vs. radians, millimeters vs. meters).

## File 6: src/mapping/sensor_to_xyz_4arm.py

Python 3.10 module with the following responsibilities:

- Read the per-arm sensor record stream from the L0 raw Zenodo pointer or from a local Parquet cache.
- Apply the per-arm phase-conditioned mapping rule.
- Apply per-arm safety zone gating.
- Apply per-arm force feedback fusion.
- Apply cumulative force enforcement by reading the heartbeat broadcast.
- Emit per-arm xyz command records.
- Emit per-arm CSV samples.
- Emit the cross-arm ASCII path visualization.

Required CLI signature using `click`:

```
@click.command()
@click.option("--seed", type=int, default=20260510)
@click.option("--sensor-in", type=click.Path(exists=True), default="data/iterations/run_00001_L0_raw.zenodo_pointer.json")
@click.option("--xyz-out-dir", type=click.Path(), default="data/iterations")
@click.option("--csv-sample-out-dir", type=click.Path(), default="data")
@click.option("--ascii-viz-out", type=click.Path(), default="viz/xyz_path_4arm.txt")
def cli(seed: int, sensor_in: str, xyz_out_dir: str, csv_sample_out_dir: str, ascii_viz_out: str) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean. The script must include a module docstring citing 21 CFR 50.30 task-order lifecycle, IEC 80601-2-77 force limits, and the 12 N cumulative force limit from `multi_arm_coordination.md`.

## File 7: src/control/robot_loop_4arm.cpp

C++20 single-file real-time control loop for all 4 arms. Required responsibilities:

- Read the per-arm xyz command stream via Apache Arrow C++ bindings or via a CSV fallback.
- Issue commands to the 4 simulated actuator buses at 1 kHz.
- Subscribe to the 1 kHz heartbeat broadcast from `arm_heartbeat.cpp`.
- Enforce per-arm E-stop latency budget (5 ms maximum).
- Enforce per-arm force clamp at 5.0 N tip before issuing each command.
- Enforce cumulative 4-arm force clamp at 12 N from heartbeat broadcast samples.
- Trigger the 5 ms emergency-park sequence on any FORBIDDEN safety zone entry or any cumulative force overshoot.
- Emit a synchronized actuator log to `logs/control_loop_4arm.txt`.

The file must include a header comment citing IEC 80601-2-77, IEC 62304, and 21 CFR 50.30. The future session must verify that the file compiles with `g++ -std=c++20 -O2 -o build/robot_loop_4arm src/control/robot_loop_4arm.cpp src/coordination/arm_heartbeat.cpp -larrow -lpthread` on Linux and with `clang++ -std=c++20 -O2 ...` on MacOS.

The control loop runs in software simulation only. It does not connect to physical hardware. The simulation calls the same actuator interface as the future NeuroSpeed 1.0 firmware would, so the same code can be deployed to a real 4-arm robot in a follow-on release without modification.

## File 8: src/coordination/arm_heartbeat.cpp

C++20 single-file implementation of the 1 kHz heartbeat sender and receiver per `multi_arm_coordination.md`. Required responsibilities:

- Allocate per-arm shared memory for inbound and outbound 32-byte heartbeat frames.
- Implement the 1 kHz transmit loop with a deadline-monotonic scheduler hint.
- Implement the per-sibling watchdog timer with a 3 ms threshold.
- Validate inbound frame CRC32 and the monotonic heartbeat_seq.
- On any failure, raise a global emergency-park flag that `robot_loop_4arm.cpp` reads at every command tick.

The future session must compile this file together with `robot_loop_4arm.cpp` for a single-binary deployment of the 4-arm control system.

## File 9: data/xyz_trace_sample_arm1.csv through arm4.csv

Four CSV files, one per arm. The future session produces these files by running the script in File 6 with the sample emission flag. Each file contains the first 1,000 commands emitted by the corresponding arm during Phase 2. Approximate size: 25 KB each, 100 KB total.

## File 10: viz/xyz_path_4arm.txt

ASCII visualization combining the per-arm paths into a single 60-line diagram. The diagram includes 4 panels (one per arm) showing the per-second mean (x_mm, y_mm, z_mm) for that arm across the 60-second window. Total under 8 KB.

## File 11: data/xyz_trace_4arm.zenodo_pointer.json

JSON file with the following keys:

```
{
  "schema_version": "1.0",
  "release_version": "v3.9.1",
  "iteration_id": "run_00001",
  "zenodo_doi": "10.5281/zenodo.PLACEHOLDER",
  "zenodo_record_id": "PLACEHOLDER",
  "zenodo_filename": "run_00001_xyz_trace_4arm.parquet",
  "sha256": "PLACEHOLDER",
  "byte_size": 12000000,
  "compression": "zstd-3",
  "channel_count": 56,
  "row_count": 240000,
  "populated_at_commit": "Commit 5 of v3.9.1 PR"
}
```

The zenodo_doi, zenodo_record_id, and sha256 fields are populated at Commit 5 after the Zenodo deposition completes. The pointer file is committed at Commit 3 with the placeholder values; Commit 5 patches the placeholders with the real values.

## Determinism

The mapper must be deterministic for a fixed seed and a fixed input pointer. The future session must verify by running the canonical emission twice with the same seed and computing SHA-256 of the resulting per-arm CSV samples; the two hashes must match.

## Validation After Commit 3

- `python -m src.mapping.sensor_to_xyz_4arm --seed 20260510` produces all output files.
- `data/xyz_trace_sample_arm1.csv` through `arm4.csv` each contain exactly 1,001 lines.
- `viz/xyz_path_4arm.txt` contains 60 lines or fewer and 80 columns or fewer.
- The C++ control loop and heartbeat code compile on Linux, MacOS, and Windows.
- The cumulative force enforcement test passes: the mapper rejects any command that would push cumulative force above 12 N.
- `ruff format --check .` passes.
- `ruff check .` passes.
- `yamllint -d relaxed competitions/glioblastoma-1min-trial/config/` passes.

## Source Files Cited

- `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md`. Source for the 7-DOF per-arm kinematic limits in `kinematics_4arm.yaml`.
- `competitions/instructions/one_minute_variant/glioblastoma_context_1min.md`. Source for the 4 phase boundaries that gate the per-arm mapping rate.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the 10 kHz force sample rate that drives the per-arm force feedback fusion.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the heartbeat protocol implemented by File 8 and the cumulative force limit enforced by Files 6 and 7.
- `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`. Source for the zstd-3 Parquet compression default.
- `competitions/instructions/one_minute_variant/zenodo_archive_protocol.md`. Source for the Zenodo pointer file schema used by File 11.
- `competitions/instructions/commit_03_xyz_mapping.md`. Source for the parent v3.9.0 9-file Commit 3 structure that this 1-minute variant extends to 11 files.
- `competitions/instructions/ascii_diagram_guide.md`. Source for the ASCII drawing rules used by File 10.
- `competitions/instructions/file_format_conventions.md`. Source for the JSON Schema, Protocol Buffers, YAML, Python, and C++ conventions.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules that File 6 and File 5 must satisfy.
- `patient-journey/stage_05_surgery.py`. Source for the TASK_ORDER_STATES reused by the per-arm `command_state` enum and for the FORCE_LIMIT_TIP_N (here tightened to 5.0 N per arm) and ESTOP_LATENCY_LIMIT_MS (here tightened to 5 ms) constants.
