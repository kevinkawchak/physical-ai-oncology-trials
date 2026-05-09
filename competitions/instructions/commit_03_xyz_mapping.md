# Commit 3: Sensor to XYZ Mapping

This file specifies the nine files the future Claude Code Opus 4.7 1M Max session must author in its third commit. The session must author exactly the files listed and must not author additional files in this commit.

## Goal

Define the deterministic transformation that converts each millisecond sensor record into a Cartesian (x, y, z) end-effector command for the ROSA ONE Brain. Author the mapping documentation, command schemas, kinematics configuration, the Python mapper script, the C++ real-time control loop, the canonical 1-hour command trace Parquet file, the human-review CSV sample, and the ASCII visualization of the traversed path.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1hr-trial/docs/coordinate_mapping.md` | Markdown | Hand-authored | 22 KB |
| 2 | `competitions/glioblastoma-1hr-trial/schemas/xyz_command.schema.json` | JSON Schema 2020-12 | Hand-authored | 8 KB |
| 3 | `competitions/glioblastoma-1hr-trial/schemas/xyz_command.proto` | Protocol Buffers 3 | Hand-authored | 3 KB |
| 4 | `competitions/glioblastoma-1hr-trial/config/kinematics.yaml` | YAML | Hand-authored | 5 KB |
| 5 | `competitions/glioblastoma-1hr-trial/src/mapping/sensor_to_xyz.py` | Python 3.10 | Hand-authored | 18 KB |
| 6 | `competitions/glioblastoma-1hr-trial/src/control/robot_loop.cpp` | C++20 | Hand-authored | 12 KB |
| 7 | `competitions/glioblastoma-1hr-trial/data/xyz_trace_1hr.parquet` | Parquet (Snappy) | Script-generated | 90 MB |
| 8 | `competitions/glioblastoma-1hr-trial/data/xyz_trace_sample.csv` | CSV | Script-generated | 100 KB |
| 9 | `competitions/glioblastoma-1hr-trial/viz/xyz_path.txt` | ASCII text | Script-generated | 6 KB |

The original instruction list named `viz/xyz_path.svg`. The future session must replace the SVG with the ASCII file `viz/xyz_path.txt`. Per `competitions/instructions/ascii_diagram_guide.md` SVG is rejected for high-frequency time series; a 1 kHz path of 3,600,000 segments cannot be rendered cleanly. The future session may additionally author an aggregate `viz/xyz_path_aggregate.svg` (1 KB) that plots the per-second mean x, y, z; this aggregate SVG is allowed because it has only 3,600 points.

## File 1: docs/coordinate_mapping.md

Required sections:

1. Mapping rule overview: each `tick_ms` record produces zero or one `XYZCommand`.
2. Phase-conditioned mapping: setup phase produces no commands; resection phases produce one command per tick at 1 kHz; closure phase produces commands at 100 Hz (one per ten ticks) for slower bipolar work.
3. Forward kinematics: 6-DOF DH parameter table for the ROSA ONE Brain. Joint twist, joint length, joint offset, joint angle for each of the six joints.
4. Inverse kinematics: closed-form analytical solution for the 6-DOF arm reaching the desired (x, y, z, qw, qx, qy, qz) pose.
5. Safety zone gating: commands inside the FORBIDDEN safety zone are clamped to the boundary; commands inside the ELOQUENT safety zone are slowed to 25 percent of nominal velocity; commands inside the TUMOR_CORE zone proceed at nominal velocity.
6. Force feedback fusion: the mapper reads the most recent `ee_fx`, `ee_fy`, `ee_fz` channels and clamps commanded velocity if force exceeds 80 percent of the IEC 80601-2-77 limit (12.0 N tip).
7. Command latency budget: 5 ms end-to-end from sensor sample arrival to the first commanded actuator update. Of that budget, 0.5 ms is reserved for the inverse kinematics solve and 4.5 ms for the C++ control loop write to the actuator bus.
8. Cross-references to schemas and to `src/control/robot_loop.cpp`.

## File 2: schemas/xyz_command.schema.json

JSON Schema 2020-12 with the following structure:

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kevinkawchak.github.io/physical-ai-oncology-trials/v3.9.0/xyz_command.schema.json",
  "title": "GBM 1hr XYZ Command",
  "type": "object",
  "required": ["tick_ms", "x_mm", "y_mm", "z_mm", "qw", "qx", "qy", "qz",
               "linear_vel_mmps", "force_clamp_N", "tool", "command_state",
               "meta_seed", "meta_iteration_id"],
  "properties": {
    "tick_ms": {"type": "integer", "minimum": 0, "maximum": 3599999},
    "x_mm": {"type": "number", "minimum": -1000, "maximum": 1000},
    "y_mm": {"type": "number", "minimum": -1000, "maximum": 1000},
    "z_mm": {"type": "number", "minimum": -1000, "maximum": 1000},
    "qw": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "qx": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "qy": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "qz": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "linear_vel_mmps": {"type": "number", "minimum": 0, "maximum": 50.0},
    "force_clamp_N": {"type": "number", "minimum": 0, "maximum": 15.0},
    "tool": {"type": "string", "enum": ["BIPOLAR", "SUCTION", "BIOPSY_NEEDLE", "RETRACTOR", "NONE"]},
    "command_state": {"type": "string", "enum": ["EMIT", "CLAMP_TO_BOUNDARY", "FORCE_HOLD", "SAFETY_PAUSE", "ABORT"]},
    "meta_seed": {"type": "integer"},
    "meta_iteration_id": {"type": "string", "pattern": "^run_[0-9]{5}$"}
  },
  "additionalProperties": false
}
```

The `linear_vel_mmps` upper bound enforces the 50 mm/s end-effector velocity limit from `competitions/instructions/robot_specification.md`. The `force_clamp_N` upper bound enforces the 15.0 N tip force limit from IEC 80601-2-77.

## File 3: schemas/xyz_command.proto

Protocol Buffers 3 definition with the same fields as the JSON Schema. Field numbers 1 through 14 cover the required keys; numbers 100 and 101 cover the metadata fields. The proto file is consumed by the C++ control loop; the future session must verify generated C++ code compiles cleanly with `protoc --cpp_out=build/proto schemas/xyz_command.proto`.

## File 4: config/kinematics.yaml

Required keys:

```
---
robot:
  make: Medtronic
  model: ROSA ONE Brain
  hardware_revision: v3.0
  firmware: "3.1.4"
  dof: 6

dh_parameters:
  joint_1:
    twist_rad: 0.0
    length_mm: 0.0
    offset_mm: 320.0
    angle_offset_rad: 0.0
  joint_2:
    twist_rad: -1.5707963
    length_mm: 0.0
    offset_mm: 0.0
    angle_offset_rad: -1.5707963
  joint_3:
    twist_rad: 0.0
    length_mm: 270.0
    offset_mm: 0.0
    angle_offset_rad: 0.0
  joint_4:
    twist_rad: -1.5707963
    length_mm: 70.0
    offset_mm: 302.0
    angle_offset_rad: 0.0
  joint_5:
    twist_rad: 1.5707963
    length_mm: 0.0
    offset_mm: 0.0
    angle_offset_rad: 0.0
  joint_6:
    twist_rad: -1.5707963
    length_mm: 0.0
    offset_mm: 72.0
    angle_offset_rad: 0.0

joint_limits:
  joint_1: {min_rad: -3.05, max_rad: 3.05, max_vel_radps: 1.57, max_acc_radpss: 6.28}
  joint_2: {min_rad: -2.09, max_rad: 2.09, max_vel_radps: 1.57, max_acc_radpss: 6.28}
  joint_3: {min_rad: -2.97, max_rad: 2.97, max_vel_radps: 1.57, max_acc_radpss: 6.28}
  joint_4: {min_rad: -3.14, max_rad: 3.14, max_vel_radps: 2.09, max_acc_radpss: 8.37}
  joint_5: {min_rad: -2.09, max_rad: 2.09, max_vel_radps: 2.09, max_acc_radpss: 8.37}
  joint_6: {min_rad: -6.28, max_rad: 6.28, max_vel_radps: 3.14, max_acc_radpss: 12.57}

end_effector:
  max_linear_vel_mmps: 50.0
  max_linear_acc_mmpss: 200.0
  max_tip_force_N: 15.0
  max_lateral_force_N: 5.0

control_loop:
  rate_hz: 1000
  estop_latency_limit_ms: 50
  command_to_actuator_budget_ms: 5
```

The kinematics file must be `yamllint -d relaxed` clean and must pass a unit-conversion sanity check (degrees vs. radians, millimeters vs. meters).

## File 5: src/mapping/sensor_to_xyz.py

Python 3.10 module with the following responsibilities:

- Read the canonical sensor Parquet file by path.
- Apply the phase-conditioned mapping rule (1 kHz during resection, 100 Hz during closure, no command during setup).
- Apply safety zone gating.
- Apply force feedback fusion.
- Emit the canonical xyz command Parquet file.
- Emit the human-review CSV sample.
- Emit the ASCII path visualization.

Required CLI signature using `click`:

```
@click.command()
@click.option("--seed", type=int, default=20260509)
@click.option("--sensor-in", type=click.Path(exists=True), default="data/sensor_1hr.parquet")
@click.option("--xyz-out", type=click.Path(), default="data/xyz_trace_1hr.parquet")
@click.option("--csv-sample-out", type=click.Path(), default="data/xyz_trace_sample.csv")
@click.option("--ascii-viz-out", type=click.Path(), default="viz/xyz_path.txt")
def cli(seed: int, sensor_in: str, xyz_out: str, csv_sample_out: str, ascii_viz_out: str) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean. The script must include a module docstring citing 21 CFR 50.30 task-order lifecycle and IEC 80601-2-77 force limits.

## File 6: src/control/robot_loop.cpp

C++20 single-file real-time control loop. Required responsibilities:

- Read the canonical xyz command Parquet file via Apache Arrow C++ bindings or via a CSV fallback.
- Issue commands to a simulated actuator bus at 1 kHz (or 100 Hz during closure).
- Enforce E-stop latency budget (50 ms maximum).
- Enforce force clamp before issuing each command.
- Emit a synchronized actuator log to `logs/control_loop.txt`.

The file must include a header comment citing IEC 80601-2-77 and 21 CFR 50.30. The future session must verify that the file compiles with `g++ -std=c++20 -O2 -o build/robot_loop src/control/robot_loop.cpp -larrow` on Linux and with `clang++ -std=c++20 -O2 ...` on MacOS.

The control loop runs in software simulation only. It does not connect to physical hardware. The simulation calls the same actuator interface as the physical ROSA ONE Brain firmware, so the same code can be deployed to a real robot in a follow-on release without modification.

## File 7: data/xyz_trace_1hr.parquet

The future session must produce this file by running:

```
python -m src.mapping.sensor_to_xyz --sensor-in data/sensor_1hr.parquet --seed 20260509
```

The file contains one record per emitted command: 0 records during setup phase (600,000 ticks), 1 record per tick during resection phases (2,700,000 ticks), and 1 record per 10 ticks during closure phase (30,000 records). Total records: approximately 2,730,000. Snappy compression. Approximate on-disk size: 90 MB.

## File 8: data/xyz_trace_sample.csv

The future session must produce this file by running the same script with `--csv-sample-out data/xyz_trace_sample.csv`. The CSV contains the first 1,000 commands emitted during the resection phase (centered around `tick_ms = 900,000`).

## File 9: viz/xyz_path.txt

ASCII visualization using template 2 from `competitions/instructions/ascii_diagram_guide.md`. Per-second mean of `x_mm`, `y_mm`, `z_mm` plotted as three single-axis ASCII charts. 60 lines maximum. Generated by `src/mapping/sensor_to_xyz.py --ascii-viz-out viz/xyz_path.txt`.

## Determinism

The mapper must be deterministic for a fixed seed and a fixed input Parquet file. The future session must verify by running the canonical emission twice with the same seed and computing SHA-256 of the resulting Parquet file; the two hashes must match.

## Validation After Commit 3

- `python -m src.mapping.sensor_to_xyz --seed 20260509` produces all four output files.
- `data/xyz_trace_1hr.parquet` contains approximately 2,730,000 records.
- `data/xyz_trace_sample.csv` contains exactly 1,001 lines.
- `viz/xyz_path.txt` contains 60 lines or fewer.
- The C++ control loop compiles and runs to completion against the canonical command Parquet.
- `ruff format --check .` passes.
- `ruff check .` passes.
- `yamllint -d relaxed competitions/glioblastoma-1hr-trial/config/` passes.

## Source Files Cited

- `competitions/instructions/robot_specification.md`. Source for the kinematic limits in `kinematics.yaml`.
- `competitions/instructions/glioblastoma_context.md`. Source for the five procedure phases that gate the mapping rate.
- `competitions/instructions/ascii_diagram_guide.md`. Source for template 2 used by `viz/xyz_path.txt`.
- `competitions/instructions/file_format_conventions.md`. Source for the Snappy Parquet rule and the SVG-vs-ASCII decision rule.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules that File 5 and File 4 must satisfy.
- `patient-journey/stage_05_surgery.py`. Source for the TASK_ORDER_STATES (`IDLE`, `SETUP`, `DOCKED`, `READY`, `ACTIVE`, `PAUSE`, `COMPLETE`, `ABORT`) reused by the `command_state` enum and for the FORCE_LIMIT_TIP_N (15.0 N) and ESTOP_LATENCY_LIMIT_MS (50 ms) constants reused by the kinematics configuration.
- `patient-journey/patient_state.py`. Source for the `SurgicalRecord` dataclass shape that the future Commit 5 outcomes file will mirror.
- `national-platform/usl_standard/`. Source for the Unification Standard Level scoring scale that the future Commit 5 metric uses; the mapper does not consume USL but emits per-iteration metadata that the metric will read.
