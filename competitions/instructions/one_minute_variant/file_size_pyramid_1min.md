# File Size Pyramid for the 1-Minute Variant (Layer 4 Addendum)

This file extends the parent `competitions/instructions/chunking_strategy.md` with a fourth chunking layer specific to the 1-minute variant. The parent file defines Layers 1, 2, and 3 (generators not data, per-commit file budgets, within-file chunking). Layer 4 below adds a per-iteration pyramid that splits the canonical L0 raw record into committed L1 to L3 aggregates plus an event log, so that the 16-iteration sweep fits inside the GitHub 10 MB committed cap while preserving millisecond ground truth on Zenodo.

## Why a Fourth Layer is Required

The 1-minute variant produces 26 MB of L0 raw per iteration across 4 arms at mixed 1 kHz plus 10 kHz force sample rates. Across 16 iterations the L0 raw total is 416 MB. Committing 416 MB to the parent repository would exceed the 10 MB committed budget by 40 times. The parent v3.9.0 1-hour scenario can use Git LFS to commit the 5.7 GB across 64 iterations because the sample rate is uniformly 1 kHz; the 4-arm 1-minute variant cannot use LFS within the 10 MB cap because the 4-arm doubling and the 10 kHz force channels overflow the per-iteration pyramid level that fits in the cap. Layer 4 therefore archives L0 to Zenodo and commits only the L1 to L3 aggregates plus an event log.

## Pyramid Levels (per iteration)

| Level | Sample rate | Per-arm rows in 1 min | Per-arm size | 4-arm size | All 16 iterations | Within 10 MB total? |
|-------|-------------|------------------------|--------------|------------|---------------------|--------------------|
| L0 raw mixed | 1 to 10 kHz mixed | 600,000 | 6.6 MB | 26 MB | 416 MB | Zenodo only, never Git |
| L1 100 Hz aggregate | 100 Hz | 6,000 | 600 KB | 2.4 MB | 38 MB | No, exceeds 10 MB cap |
| L1 20 Hz aggregate (recommended) | 20 Hz | 1,200 | 120 KB | 480 KB | 7.7 MB | YES |
| L2 1 Hz aggregate | 1 Hz | 60 | 6 KB | 24 KB | 384 KB | YES |
| L3 per-phase aggregate | per-phase | 4 | under 1 KB | under 4 KB | under 64 KB | YES |
| Event log | event-driven | typically 50 to 200 | 2 KB | 8 KB | 128 KB | YES |

The 4-arm doubling of channels makes the 100 Hz pyramid level too large for the 10 MB cap. The recommended L1 rate for the 4-arm 1-minute variant is therefore 20 Hz, not 100 Hz. Final per-iteration committed total is approximately 510 KB across L1 plus L2 plus L3 plus events.

## Aggregate Definitions

The future Commit 4 author must implement the following aggregate definitions exactly. The aggregations preserve the safety-relevant signals while reducing storage by 50 to 600 times.

### L1 (20 Hz aggregate, 50 ms window)

For each 50 ms window the L1 record carries:

- Per-arm window timestamp tick_50ms (range 0 to 1199, integer).
- Per-arm mean joint position vector (7 doubles).
- Per-arm mean joint velocity vector (7 doubles).
- Per-arm peak joint torque vector (7 doubles), absolute value.
- Per-arm mean end-effector position (3 doubles).
- Per-arm peak end-effector force vector (3 doubles), absolute value, taken over the 500 force samples in the 50 ms window.
- Per-arm peak end-effector torque vector (3 doubles).
- Per-arm peak navigation deviation vector (3 doubles).
- Per-arm safety_zone enum (most permissive observed in window).
- Per-arm robot_state enum (last observed in window).
- Per-arm cumulative tip-force-violation event count for the window.
- Per-arm heartbeat_ok bit (1 if all 50 windows in this 50 ms range had heartbeat_ok = 1, else 0).
- meta_seed integer.
- meta_iteration_id string.
- arm_id enum.

L1 size estimate per arm: 1,200 records times 50 columns at zstd-3 equals approximately 120 KB. Across 4 arms: 480 KB.

### L2 (1 Hz aggregate, 1 second window)

For each 1 second window the L2 record carries:

- Per-arm window timestamp tick_1s (range 0 to 59, integer).
- Per-arm mean and peak end-effector position (6 doubles).
- Per-arm peak end-effector force vector (3 doubles).
- Per-arm cumulative tip force violations in window.
- Per-arm cumulative E-stop engagement count in window.
- Per-arm cumulative AE injection count in window.
- Per-arm tissue removal volume in window (mm cubed).
- Per-arm safety_zone enum (most permissive observed in window).
- Per-arm robot_state enum (last observed in window).
- meta_seed and meta_iteration_id and arm_id.

L2 size estimate per arm: 60 records times 20 columns at zstd-3 equals approximately 6 KB. Across 4 arms: 24 KB.

### L3 (per-phase aggregate, 4 records per iteration)

For each of the 4 phases (Phase 1 through Phase 4) the L3 record carries:

- Phase ID (1 through 4) and phase name.
- Phase start_us and end_us integers.
- Per-arm cumulative tip force violations in phase.
- Per-arm cumulative E-stop engagement count in phase.
- Per-arm cumulative AE injection count in phase.
- Per-arm tissue removal volume in phase.
- Per-arm peak end-effector force vector across phase.
- Per-arm peak end-effector velocity scalar across phase.
- Per-arm phase-end safety_zone and robot_state.
- meta_seed, meta_iteration_id, arm_id.

L3 size per iteration: 4 records times 4 arms times 20 columns at zstd-3 equals approximately 4 KB.

### Event log (event-driven)

For each detected event (force violation, E-stop engagement, AE injection, gap detection, heartbeat miss, safety zone transition) the event log emits one record with:

- Event timestamp tick_us.
- Event kind enum.
- arm_id enum.
- Event payload as JSON string.
- meta_seed, meta_iteration_id.

Event log size per iteration: typically 50 to 200 events at 40 bytes each plus zstd-3 compression equals approximately 8 KB.

### L0 raw archive pointer

Each iteration includes a single hand-authored file at Commit 4 and populated at Commit 5: `data/iterations/run_NNNNN_L0_raw.zenodo_pointer.json`. This file points to the Zenodo deposition for that iteration's L0 raw and includes the SHA-256 of the L0 Parquet on Zenodo. The pointer file is approximately 1 KB.

## Per-Iteration Output Schema (Layer 4 committed)

| File | Format | Size | Authoring approach |
|------|--------|------|--------------------|
| `data/iterations/run_NNNNN_L1_50ms.parquet` | Parquet zstd-3 | 480 KB across 4 arms | Script-generated, committed |
| `data/iterations/run_NNNNN_L2_1s.parquet` | Parquet zstd-3 | 24 KB across 4 arms | Script-generated, committed |
| `data/iterations/run_NNNNN_L3_phase.parquet` | Parquet zstd-3 | under 4 KB across 4 arms | Script-generated, committed |
| `data/iterations/run_NNNNN_events.parquet` | Parquet zstd-3 | 8 KB | Script-generated, committed |
| `data/iterations/run_NNNNN_L0_raw.zenodo_pointer.json` | JSON | 1 KB | Hand-authored at Commit 4, populated at Commit 5 |

Per-iteration committed total: approximately 510 KB across all five files. The per-iteration file count is 5 files; across 16 iterations this is 80 files.

## Total Repository Storage Budget for v3.9.1

| Bucket | Size |
|--------|------|
| Per-iteration committed (16 iterations times 510 KB) | 8.2 MB |
| Schemas, scripts, configs, README, viz (fixed overhead) | 1.5 MB |
| Total committed | 9.7 MB |
| Zenodo L0 archive (16 iterations times 26 MB) | 416 MB |

Total committed of 9.7 MB sits inside the 10 MB cap with 0.3 MB headroom. The Zenodo free 50 GB tier covers the 416 MB comfortably.

## File Size Cap Enforcement

The future Commit 6 error fix pass must run a check that no committed file exceeds 10 MB and that no committed Parquet file exceeds 5 MB. The check is added to `ci_compliance_checklist.md` as inherited from the parent v3.9.0 plus the 1-minute variant addendum. The check command is:

```
find competitions/glioblastoma-1min-trial -type f -size +10M -print | (! grep -q .) || (echo "ERROR: file over 10 MB"; exit 1)
find competitions/glioblastoma-1min-trial -name '*.parquet' -size +5M -print | (! grep -q .) || (echo "ERROR: parquet over 5 MB"; exit 1)
```

If either find command emits any path, the future session must reduce the offending file's sample rate or compression aggressiveness, or move the file to Zenodo and replace it with a pointer.

## Compression Default Override (zstd-3 vs Snappy)

The parent `file_format_conventions.md` defaults Parquet compression to Snappy. The 1-minute variant overrides this default to zstd-3 because zstd-3 is approximately 30 percent smaller than Snappy at the same decompression speed for the dense numeric Parquet payloads produced by the L1 to L3 aggregates. The override is documented here and is referenced by the future Commit 2 sensor schema and by every Parquet emission in the 1-minute output tree. Snappy remains the default for the parent v3.9.0 1-hour scenario.

## Source Files Cited

- `competitions/instructions/chunking_strategy.md`. Source for the parent three-layer chunking strategy that this Layer 4 addendum extends. The parent strategy applies unchanged to the v3.9.0 1-hour scenario.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the 26 MB per-iteration L0 raw size that drives the Zenodo-vs-Git decision and the 4-arm 200-channel structure that doubles the parent 50-channel pyramid.
- `competitions/instructions/one_minute_variant/zenodo_archive_protocol.md`. Source for the Zenodo deposition layout and the SHA-256 manifest contract that the L0 raw archive follows.
- `competitions/instructions/file_format_conventions.md`. Source for the default Snappy compression that this variant overrides to zstd-3.
- `competitions/instructions/ci_compliance_checklist.md`. Source for the pre-commit lint and yamllint checks; the 10 MB and 5 MB caps documented above are added to the parent checklist via the 1-minute variant addendum.
