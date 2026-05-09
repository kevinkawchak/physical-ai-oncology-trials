# Commit 2: Sensor Data Specifications

This file specifies the eight files the future Claude Code Opus 4.7 1M Max session must author in its second commit. The session must author exactly the files listed and must not author additional files in this commit.

## Goal

Define the canonical millisecond-resolution sensor record for the Medtronic ROSA ONE Brain. Author the human-readable specification, three machine-readable schemas (JSON Schema, Protocol Buffers, Avro), the human-review samples in JSONL and CSV, the canonical 1-hour Parquet file, and the ingest script that validates incoming streams.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1hr-trial/docs/sensor_spec.md` | Markdown | Hand-authored | 28 KB |
| 2 | `competitions/glioblastoma-1hr-trial/schemas/sensor_record.schema.json` | JSON Schema 2020-12 | Hand-authored | 18 KB |
| 3 | `competitions/glioblastoma-1hr-trial/schemas/sensor_record.proto` | Protocol Buffers 3 | Hand-authored | 6 KB |
| 4 | `competitions/glioblastoma-1hr-trial/schemas/sensor_record.avsc` | Apache Avro JSON | Hand-authored | 8 KB |
| 5 | `competitions/glioblastoma-1hr-trial/data/sensor_sample.jsonl` | JSON Lines | Script-generated, then committed | 250 KB |
| 6 | `competitions/glioblastoma-1hr-trial/data/sensor_1hr.parquet` | Parquet (Snappy) | Script-generated, then committed | 60 MB |
| 7 | `competitions/glioblastoma-1hr-trial/data/sensor_sample.csv` | CSV | Script-generated, then committed | 100 KB |
| 8 | `competitions/glioblastoma-1hr-trial/src/sensors/ingest.py` | Python 3.10 | Hand-authored | 14 KB |

## File 1: docs/sensor_spec.md

Required sections:

1. Channel inventory: 50 channels reproduced verbatim from `competitions/instructions/robot_specification.md`.
2. Sampling rate: 1 kHz, 1 sample per millisecond per channel.
3. Tick alignment: monotonic millisecond timestamp from procedure start; first tick is 0, last tick is 3,599,999.
4. Units and tolerances per channel.
5. Coordinate frame and quaternion convention.
6. Validation rules: range, monotonic timestamp, no NaN, force limits enforced.
7. Stream framing: each tick is one record; record boundaries are newlines in JSONL and Protocol Buffers length-prefixed in binary.
8. Storage estimate: 50 channels times 3,600,000 ticks = 180,000,000 numeric values plus the timestamp column. Parquet with Snappy compression and dictionary encoding for enum columns lands at approximately 60 MB.
9. Failure handling: dropped tick reconstruction policy, gap detection, gap report log.
10. Cross-references to schemas in `schemas/`.

The future session must include single dashes only and black text only. The future session must use ASCII text inside fenced code blocks for any in-line diagrams; Mermaid is acceptable for the channel-source-to-storage flow diagram.

## File 2: schemas/sensor_record.schema.json

JSON Schema 2020-12 with the following top-level structure. All 50 channels are required keys. The `tick_ms` integer key is also required.

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kevinkawchak.github.io/physical-ai-oncology-trials/v3.9.0/sensor_record.schema.json",
  "title": "GBM 1hr Sensor Record",
  "type": "object",
  "required": ["tick_ms", "j1_pos", "j2_pos", "j3_pos", "j4_pos", "j5_pos", "j6_pos",
               "j1_vel", "j2_vel", "j3_vel", "j4_vel", "j5_vel", "j6_vel",
               "j1_trq", "j2_trq", "j3_trq", "j4_trq", "j5_trq", "j6_trq",
               "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
               "ee_fx", "ee_fy", "ee_fz", "ee_tx", "ee_ty", "ee_tz",
               "nav_dx", "nav_dy", "nav_dz",
               "ttip_temp", "irr_flow", "suc_flow", "co2_insuf",
               "us_present", "ala_uv", "imri_active", "estop_state",
               "safety_zone", "robot_state",
               "meta_seed", "meta_iteration_id"],
  "properties": {
    "tick_ms": {"type": "integer", "minimum": 0, "maximum": 3599999},
    "j1_pos": {"type": "number"},
    "ee_x": {"type": "number", "minimum": -1000, "maximum": 1000},
    "ee_fz": {"type": "number", "minimum": -50, "maximum": 50},
    "safety_zone": {"type": "string", "enum": ["NONE", "OUTER", "INNER", "ELOQUENT", "FORBIDDEN", "TUMOR_CORE", "TUMOR_MARGIN", "VESSEL"]},
    "robot_state": {"type": "string", "enum": ["IDLE", "SETUP", "DOCKED", "READY", "ACTIVE", "PAUSE", "COMPLETE", "ABORT"]},
    "meta_seed": {"type": "integer"},
    "meta_iteration_id": {"type": "string", "pattern": "^run_[0-9]{5}$"}
  },
  "additionalProperties": false
}
```

The future session must complete the property definitions for all 50 channels using the units and ranges from `competitions/instructions/robot_specification.md`. The `additionalProperties: false` clause prevents drift across versions.

## File 3: schemas/sensor_record.proto

Protocol Buffers 3 definition with the same 50 channels plus the two metadata fields. Use the `optional` keyword judiciously; required fields use the proto3 default presence detection. Example skeleton:

```
syntax = "proto3";
package gbm_trial.v3_9_0;

message SensorRecord {
  uint32 tick_ms = 1;
  double j1_pos = 2;
  double j2_pos = 3;
  // ... continued for all 50 channels ...
  uint64 meta_seed = 100;
  string meta_iteration_id = 101;

  enum SafetyZone {
    SAFETY_ZONE_NONE = 0;
    SAFETY_ZONE_OUTER = 1;
    SAFETY_ZONE_INNER = 2;
    SAFETY_ZONE_ELOQUENT = 3;
    SAFETY_ZONE_FORBIDDEN = 4;
    SAFETY_ZONE_TUMOR_CORE = 5;
    SAFETY_ZONE_TUMOR_MARGIN = 6;
    SAFETY_ZONE_VESSEL = 7;
  }
  SafetyZone safety_zone = 50;

  enum RobotState {
    ROBOT_STATE_IDLE = 0;
    ROBOT_STATE_SETUP = 1;
    ROBOT_STATE_DOCKED = 2;
    ROBOT_STATE_READY = 3;
    ROBOT_STATE_ACTIVE = 4;
    ROBOT_STATE_PAUSE = 5;
    ROBOT_STATE_COMPLETE = 6;
    ROBOT_STATE_ABORT = 7;
  }
  RobotState robot_state = 51;
}
```

Field numbers 1 through 49 cover the 47 numeric joint, end-effector, force, navigation, and adjunct channels; numbers 50 and 51 cover the two enum channels; numbers 100 and 101 cover the metadata fields. Reserve numbers 52 through 99 for future expansion.

## File 4: schemas/sensor_record.avsc

Apache Avro schema with the same 50 channels plus the metadata fields. Avro is used for archival because the schema travels with the data, allowing later format evolution. Skeleton:

```
{
  "type": "record",
  "name": "SensorRecord",
  "namespace": "gbm_trial.v3_9_0",
  "fields": [
    {"name": "tick_ms", "type": "int"},
    {"name": "j1_pos", "type": "double"},
    {"name": "ee_x", "type": "double"},
    {"name": "safety_zone", "type": {
      "type": "enum",
      "name": "SafetyZone",
      "symbols": ["NONE", "OUTER", "INNER", "ELOQUENT", "FORBIDDEN", "TUMOR_CORE", "TUMOR_MARGIN", "VESSEL"]
    }},
    {"name": "robot_state", "type": {
      "type": "enum",
      "name": "RobotState",
      "symbols": ["IDLE", "SETUP", "DOCKED", "READY", "ACTIVE", "PAUSE", "COMPLETE", "ABORT"]
    }},
    {"name": "meta_seed", "type": "long"},
    {"name": "meta_iteration_id", "type": "string"}
  ]
}
```

The future session must complete the field list for all 50 channels.

## File 5: data/sensor_sample.jsonl

The future session must author this file by running the generator script committed in File 8 (`src/sensors/ingest.py --emit-sample`). The sample contains exactly 1,000 records: a stratified sample of 250 records from each of the 4 procedure phases beyond setup. Each record is a single line of JSON conforming to `sensor_record.schema.json`.

## File 6: data/sensor_1hr.parquet

The future session must produce this file by running the generator script committed in File 8 (`src/sensors/ingest.py --emit-canonical --seed 20260509`). The file contains exactly 3,600,000 records, one per millisecond. Snappy compression and dictionary encoding for the two enum columns. Approximate on-disk size: 60 MB.

The Parquet file is committed via Git LFS if the parent repository has LFS enabled. The future session must check `.gitattributes` and add a `*.parquet filter=lfs diff=lfs merge=lfs -text` entry if LFS is in use.

## File 7: data/sensor_sample.csv

The future session must produce this file by running the generator script committed in File 8 (`src/sensors/ingest.py --emit-csv-sample`). The CSV contains the first 1,000 ticks (the first second at 1 kHz) with the header row. Approximate size: 100 KB.

## File 8: src/sensors/ingest.py

Python 3.10 script with the following responsibilities:

- Validate incoming JSONL or Protocol Buffers stream against `sensor_record.schema.json`.
- Emit the human-review sample JSONL (`--emit-sample`).
- Emit the human-review CSV sample (`--emit-csv-sample`).
- Emit the canonical 1-hour Parquet (`--emit-canonical --seed <int>`).
- Detect dropped ticks and write a gap report to `logs/sensor_gap_report.jsonl`.
- Enforce force limits and report violations.

Required CLI signature using `click`:

```
@click.command()
@click.option("--seed", type=int, default=20260509)
@click.option("--out", type=click.Path(), default="data")
@click.option("--emit-sample", is_flag=True)
@click.option("--emit-csv-sample", is_flag=True)
@click.option("--emit-canonical", is_flag=True)
@click.option("--validate", type=click.Path(), default=None)
def cli(seed: int, out: str, emit_sample: bool, emit_csv_sample: bool, emit_canonical: bool, validate: str | None) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean. The script must include a module docstring citing IEC 80601-2-77 force limits and 21 CFR 50.30 task-order lifecycle.

## Determinism

The generator must be deterministic for a fixed seed. The future session must verify by running the canonical emission twice with the same seed and computing SHA-256 of the resulting Parquet file; the two hashes must match.

## Validation After Commit 2

- `python -m src.sensors.ingest --emit-sample` produces `data/sensor_sample.jsonl` with exactly 1,000 records.
- `python -m src.sensors.ingest --emit-csv-sample` produces `data/sensor_sample.csv` with exactly 1,001 lines (header plus 1,000 rows).
- `python -m src.sensors.ingest --emit-canonical --seed 20260509` produces `data/sensor_1hr.parquet` with exactly 3,600,000 records.
- `python -m src.sensors.ingest --validate data/sensor_sample.jsonl` exits 0.
- `ruff format --check .` passes.
- `ruff check .` passes.

## Source Files Cited

- `competitions/instructions/robot_specification.md`. Source for the 50-channel list, units, ranges, and the safety zone and robot state enumerations.
- `competitions/instructions/glioblastoma_context.md`. Source for the 1-hour duration and the five procedure phases.
- `competitions/instructions/chunking_strategy.md`. Source for the sample sizes (1,000 records JSONL, 1,000 rows CSV) and the script-then-Parquet pattern.
- `competitions/instructions/file_format_conventions.md`. Source for the Snappy compression rule and the dictionary encoding for enum columns.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules that File 8 must satisfy.
- `patient-journey/stage_05_surgery.py`. Source for the IEC 80601-2-77 force limits, the task-order lifecycle states, and the 1 kHz runtime safety monitoring cadence that File 8 honors.
- `patient-journey/patient_state.py`. Source for the dataclass shape that the JSON Schema mirrors.
