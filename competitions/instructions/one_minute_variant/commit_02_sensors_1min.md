# Commit 2 (1-Minute Variant): Sensor Data Specifications for 4 Arms

This file specifies the files the future Claude Code Opus 4.7 1M Max session must author in its second commit for the 1-minute variant. The session must author exactly the files listed and must not author additional files in this commit. The parent `competitions/instructions/commit_02_sensor_specifications.md` lists 8 files for the v3.9.0 1-hour scenario. This 1-minute variant lists 9 files because the 4-arm topology adds the per-arm Zenodo pointer for the L0 raw archive.

## Goal

Define the canonical mixed 1 kHz plus 10 kHz force per-arm sensor record for the Medtronic NeuroSpeed 1.0. Author the human-readable specification, three machine-readable schemas (JSON Schema, Protocol Buffers, Avro), the per-arm human-review samples in JSONL and CSV, the per-arm L0 raw Zenodo pointer (no committed Parquet at the L0 level due to the 5 MB committed Parquet cap), and the ingest script that validates incoming streams.

## Files to Author

| Order | Path | Format | Authoring approach | Approximate size |
|-------|------|--------|--------------------|-------------------|
| 1 | `competitions/glioblastoma-1min-trial/docs/sensor_spec.md` | Markdown | Hand-authored | 32 KB |
| 2 | `competitions/glioblastoma-1min-trial/schemas/sensor_record_4arm.schema.json` | JSON Schema 2020-12 | Hand-authored | 22 KB |
| 3 | `competitions/glioblastoma-1min-trial/schemas/sensor_record_4arm.proto` | Protocol Buffers 3 | Hand-authored | 8 KB |
| 4 | `competitions/glioblastoma-1min-trial/schemas/sensor_record_4arm.avsc` | Apache Avro JSON | Hand-authored | 10 KB |
| 5 | `competitions/glioblastoma-1min-trial/data/sensor_sample_4arm.jsonl` | JSON Lines | Script-generated, then committed | 250 KB |
| 6 | `competitions/glioblastoma-1min-trial/data/sensor_sample_4arm.csv` | CSV | Script-generated, then committed | 100 KB |
| 7 | `competitions/glioblastoma-1min-trial/src/sensors/ingest_4arm.py` | Python 3.10 | Hand-authored | 18 KB |
| 8 | `competitions/glioblastoma-1min-trial/data/sensor_l0_raw_4arm.zenodo_pointer.json` | JSON | Hand-authored, populated at Commit 5 | 1 KB |
| 9 | `competitions/glioblastoma-1min-trial/docs/file_size_pyramid_1min.md` | Markdown | Hand-authored | 12 KB |

The full per-iteration L0 raw Parquet stream lives on Zenodo per `zenodo_archive_protocol.md` because at 4 arms times mixed 1 kHz plus 10 kHz force times 60 seconds the L0 raw Parquet is 26 MB, which exceeds the 5 MB committed Parquet cap. The Zenodo pointer at File 8 holds the SHA-256 and DOI; the pointer is populated at Commit 5 after the Zenodo deposition completes.

## File 1: docs/sensor_spec.md

Required sections:

1. Per-arm channel inventory: 50 channels per arm reproduced verbatim from `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`.
2. Sample rate: mixed 1 kHz commands plus 10 kHz force per arm.
3. Tick alignment: monotonic microsecond timestamp from procedure start; first MIXED tick is 0; last MIXED tick is 59,999,000; FORCE_ONLY ticks fill the 9 sub-millisecond positions.
4. Per-channel units and tolerances per arm.
5. Per-arm coordinate frame and quaternion convention.
6. Validation rules: range, monotonic timestamp, no NaN, per-arm force limits enforced (5 N tip), cumulative cross-arm force limit enforced (12 N).
7. Stream framing per arm: each tick is one record; record boundaries are newlines in JSONL and Protocol Buffers length-prefixed in binary; the 4 arms are multiplexed by ascending tick_us then ascending arm_id.
8. Storage estimate: 26 MB per iteration L0 raw across 4 arms; 416 MB across 16 iterations; archived to Zenodo per `zenodo_archive_protocol.md`.
9. Per-arm dropped tick reconstruction policy and gap detection.
10. Cross-references to schemas in `schemas/`.

The future session must include single dashes only and black text only.

## File 2: schemas/sensor_record_4arm.schema.json

JSON Schema 2020-12 with the following top-level structure. The schema uses `arm_id` as a discriminator and `record_kind` to distinguish MIXED records (all 50 channels at 1 kHz) from FORCE_ONLY records (6 force channels at 10 kHz):

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kevinkawchak.github.io/physical-ai-oncology-trials/v3.9.1/sensor_record_4arm.schema.json",
  "title": "GBM 1min Sensor Record per Arm",
  "type": "object",
  "required": ["tick_us", "arm_id", "record_kind",
               "meta_seed", "meta_iteration_id"],
  "properties": {
    "tick_us": {"type": "integer", "minimum": 0, "maximum": 60000000},
    "arm_id": {"type": "string", "enum": ["ARM_1", "ARM_2", "ARM_3", "ARM_4"]},
    "record_kind": {"type": "string", "enum": ["MIXED", "FORCE_ONLY"]},
    "meta_seed": {"type": "integer"},
    "meta_iteration_id": {"type": "string", "pattern": "^run_[0-9]{5}$"}
  },
  "oneOf": [
    {
      "properties": {
        "record_kind": {"const": "MIXED"}
      },
      "required": ["j1_pos", "j2_pos", "j3_pos", "j4_pos", "j5_pos", "j6_pos", "j7_pos",
                   "j1_vel", "j2_vel", "j3_vel", "j4_vel", "j5_vel", "j6_vel", "j7_vel",
                   "j1_trq", "j2_trq", "j3_trq", "j4_trq", "j5_trq", "j6_trq", "j7_trq",
                   "ee_x", "ee_y", "ee_z",
                   "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                   "ee_fx", "ee_fy", "ee_fz",
                   "ee_tx", "ee_ty", "ee_tz",
                   "nav_dx", "nav_dy", "nav_dz",
                   "ttip_temp", "irr_flow", "suc_flow", "co2_insuf",
                   "us_present", "ala_uv", "imri_active",
                   "estop_state", "safety_zone", "robot_state",
                   "heartbeat_ok", "tick_align_flag"]
    },
    {
      "properties": {
        "record_kind": {"const": "FORCE_ONLY"}
      },
      "required": ["ee_fx", "ee_fy", "ee_fz", "ee_tx", "ee_ty", "ee_tz"]
    }
  ],
  "additionalProperties": false
}
```

The future session must complete the property definitions for all 50 per-arm MIXED-record channels using the units and ranges from `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`.

## File 3: schemas/sensor_record_4arm.proto

Protocol Buffers 3 definition with the same channels plus the metadata fields. Use `oneof` for the MIXED vs FORCE_ONLY discriminator. Skeleton:

```
syntax = "proto3";
package gbm_trial_1min.v3_9_1;

message SensorRecord4Arm {
  uint64 tick_us = 1;
  ArmId arm_id = 2;
  RecordKind record_kind = 3;
  uint64 meta_seed = 4;
  string meta_iteration_id = 5;

  oneof payload {
    MixedRecord mixed = 10;
    ForceOnlyRecord force_only = 11;
  }

  enum ArmId {
    ARM_UNSPECIFIED = 0;
    ARM_1 = 1;
    ARM_2 = 2;
    ARM_3 = 3;
    ARM_4 = 4;
  }

  enum RecordKind {
    RECORD_UNSPECIFIED = 0;
    MIXED = 1;
    FORCE_ONLY = 2;
  }
}

message MixedRecord {
  // 7 joint position, 7 velocity, 7 torque
  // 3 ee position, 4 ee quaternion
  // 3 ee force, 3 ee torque
  // 3 nav deviation
  // 7 tool flags and adjuncts
  // 6 safety enums and metadata
  // ... continued for all 50 channels ...
}

message ForceOnlyRecord {
  double ee_fx = 1;
  double ee_fy = 2;
  double ee_fz = 3;
  double ee_tx = 4;
  double ee_ty = 5;
  double ee_tz = 6;
}
```

Reserve field numbers 12 through 99 in the outer message for future expansion.

## File 4: schemas/sensor_record_4arm.avsc

Apache Avro schema with the same channels plus the metadata fields. Use Avro `union` for the MIXED vs FORCE_ONLY discriminator. The future session must complete the field list for all 50 per-arm MIXED-record channels.

## File 5: data/sensor_sample_4arm.jsonl

The future session must author this file by running the generator script committed in File 7 (`src/sensors/ingest_4arm.py --emit-sample`). The sample contains 1,000 records: a stratified sample of 250 records from each of the 4 arms across all 4 phases. Each record is a single line of JSON conforming to `sensor_record_4arm.schema.json`. Approximate size: 250 KB.

## File 6: data/sensor_sample_4arm.csv

The future session must author this file by running the generator script committed in File 7 (`src/sensors/ingest_4arm.py --emit-csv-sample`). The CSV contains 1,000 MIXED records: the first 250 mixed ticks (250 ms) for each of the 4 arms. Approximate size: 100 KB.

## File 7: src/sensors/ingest_4arm.py

Python 3.10 script with the following responsibilities:

- Validate incoming JSONL or Protocol Buffers stream against `sensor_record_4arm.schema.json`.
- Emit the per-arm human-review sample JSONL (`--emit-sample`).
- Emit the per-arm human-review CSV sample (`--emit-csv-sample`).
- Emit the canonical per-iteration L0 raw Parquet to a configurable output path (`--emit-canonical --seed <int>`). The L0 raw is 26 MB across 4 arms and is uploaded to Zenodo by Commit 5; the local file is excluded from Git via `data/.gitignore`.
- Detect dropped ticks per arm and write a gap report to `logs/sensor_gap_report.jsonl`.
- Enforce per-arm force limits and report violations.
- Enforce cumulative 4-arm force limit at 12 N and report violations.

Required CLI signature using `click`:

```
@click.command()
@click.option("--seed", type=int, default=20260510)
@click.option("--out", type=click.Path(), default="data")
@click.option("--emit-sample", is_flag=True)
@click.option("--emit-csv-sample", is_flag=True)
@click.option("--emit-canonical", is_flag=True)
@click.option("--validate", type=click.Path(), default=None)
def cli(seed: int, out: str, emit_sample: bool, emit_csv_sample: bool, emit_canonical: bool, validate: str | None) -> None:
    ...
```

The script must be `ruff format` and `ruff check` clean. The script must include a module docstring citing IEC 80601-2-77 force limits, the cumulative 12 N limit from `multi_arm_coordination.md`, and 21 CFR 50.30 task-order lifecycle.

## File 8: data/sensor_l0_raw_4arm.zenodo_pointer.json

JSON file with the following keys (one canonical pointer for the v3.9.1 release-wide L0 archive, separate from the per-iteration pointers documented in `commit_04_iterations_1min.md`):

```
{
  "schema_version": "1.0",
  "release_version": "v3.9.1",
  "scope": "release_aggregate",
  "zenodo_doi": "10.5281/zenodo.PLACEHOLDER",
  "zenodo_record_id": "PLACEHOLDER",
  "zenodo_filename_pattern": "run_NNNNN_L0_raw_4arm.parquet",
  "iteration_count": 16,
  "size_per_iteration_mb": 26,
  "size_total_mb": 416,
  "compression": "zstd-3",
  "channel_count_per_arm": 50,
  "arm_count": 4,
  "channel_count_total": 200,
  "mixed_sample_rate_hz": 1000,
  "force_sample_rate_hz": 10000,
  "populated_at_commit": "Commit 5 of v3.9.1 PR"
}
```

## File 9: docs/file_size_pyramid_1min.md

The future session must embed a verbatim copy of `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md` into the output tree at `docs/file_size_pyramid_1min.md`. The embedded copy lets the simulation be self-standing once the future session ships it.

## Determinism

The generator must be deterministic for a fixed seed. The future session must verify by running the canonical emission twice with the same seed and computing SHA-256 of the resulting per-arm JSONL samples; the two hashes must match.

## Validation After Commit 2

- `python -m src.sensors.ingest_4arm --emit-sample` produces `data/sensor_sample_4arm.jsonl` with exactly 1,000 records.
- `python -m src.sensors.ingest_4arm --emit-csv-sample` produces `data/sensor_sample_4arm.csv` with exactly 1,001 lines (header plus 1,000 rows).
- `python -m src.sensors.ingest_4arm --emit-canonical --seed 20260510` produces a 26 MB Parquet file in a configurable local directory (excluded from Git).
- `python -m src.sensors.ingest_4arm --validate data/sensor_sample_4arm.jsonl` exits 0.
- The cumulative 4-arm force enforcement test passes: the script flags any tick with cumulative force above 12 N.
- `ruff format --check .` passes.
- `ruff check .` passes.

## Source Files Cited

- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the 50-channel-per-arm list, units, ranges, sample rates, and the 200-channel total across 4 arms.
- `competitions/instructions/one_minute_variant/glioblastoma_context_1min.md`. Source for the 60-second duration and the four procedure phases.
- `competitions/instructions/one_minute_variant/multi_arm_coordination.md`. Source for the cumulative 12 N force limit and the heartbeat_ok channel.
- `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`. Source for the L0 raw 26 MB per-iteration size and the file referenced by File 9.
- `competitions/instructions/one_minute_variant/zenodo_archive_protocol.md`. Source for the Zenodo pointer schema used by File 8.
- `competitions/instructions/commit_02_sensor_specifications.md`. Source for the parent v3.9.0 8-file Commit 2 structure that this 1-minute variant extends to 9 files.
- `competitions/instructions/chunking_strategy.md`. Source for the sample sizes (1,000 records JSONL, 1,000 rows CSV) and the script-then-Parquet pattern.
- `competitions/instructions/file_format_conventions.md`. Source for the zstd-3 Parquet compression default override.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the ruff and yamllint rules that File 7 must satisfy.
- `patient-journey/stage_05_surgery.py`. Source for the IEC 80601-2-77 force limits.
- `patient-journey/patient_state.py`. Source for the dataclass shape that the JSON Schema mirrors.
