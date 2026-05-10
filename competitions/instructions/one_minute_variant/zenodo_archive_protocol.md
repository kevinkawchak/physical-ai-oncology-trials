# Zenodo Archive Protocol for the 1-Minute Variant L0 Raw

This file fixes the Zenodo deposition protocol that the 1-minute variant uses to archive the 416 MB of L0 raw Parquet that cannot be committed to Git under the 10 MB committed cap. The protocol covers DOI assignment, deposition layout, the SHA-256 manifest contract, and the patching of Zenodo pointer JSON files that appear throughout the variant output tree.

## Why Zenodo

The 1-minute variant produces 26 MB of L0 raw Parquet per iteration across 4 arms at mixed 1 kHz plus 10 kHz force sample rates. Across 16 iterations the L0 raw total is 416 MB. The parent repository GitHub size budget caps committed files at 10 MB and the 1-minute variant addendum to `ci_compliance_checklist.md` caps committed Parquet at 5 MB. The L0 raw therefore cannot live in Git. Zenodo provides a free 50 GB deposition tier per record with a permanent DOI and a SHA-256 checksum for every file, satisfying the long-term archival requirement and the per-iteration determinism check.

## Deposition Layout

The 1-minute variant uses one Zenodo record per release version. The v3.9.1 release deposits the following files:

```
zenodo_record_v3_9_1/
  README.md                              # Release notes and pointer back to the GitHub repository
  manifest.json                          # SHA-256 hashes of every file in this deposition
  schema_version.json                    # Schema version of the L0 raw Parquet
  run_00001_L0_raw_4arm.parquet          # Iteration 1 L0 raw, 26 MB
  run_00002_L0_raw_4arm.parquet
  ...
  run_00016_L0_raw_4arm.parquet          # Iteration 16 L0 raw, 26 MB
  run_00001_xyz_trace_4arm.parquet       # Iteration 1 per-arm xyz command trace, 12 MB (joined sensor + xyz)
  run_00002_xyz_trace_4arm.parquet
  ...
  run_00016_xyz_trace_4arm.parquet
  release_aggregate_l0_raw_4arm.parquet  # Concatenation of all 16 L0 raw files for bulk download
```

Total deposition size: 416 MB L0 raw plus 192 MB xyz trace plus a concatenated bulk file at 416 MB equals approximately 1 GB. Comfortably inside the 50 GB free tier.

## DOI Assignment

The Zenodo record receives a DOI on first publication. The DOI follows the Zenodo convention:

```
10.5281/zenodo.{record_id}
```

The future Commit 5 author requests a DOI reservation via the Zenodo API before uploading the files, then publishes after upload. The DOI is recorded in `competitions/glioblastoma-1min-trial/releases/v3.9.1/zenodo_doi.txt` and in every Zenodo pointer JSON file under `competitions/glioblastoma-1min-trial/data/`.

## SHA-256 Manifest Contract

Every Zenodo deposition includes a `manifest.json` file at the root of the deposition. The manifest has the following schema:

```
{
  "schema_version": "1.0",
  "release_version": "v3.9.1",
  "deposited_at": "2026-05-10T20:00:00Z",
  "git_sha": "PLACEHOLDER_PATCHED_AT_DEPOSITION",
  "files": [
    {
      "filename": "run_00001_L0_raw_4arm.parquet",
      "sha256": "PLACEHOLDER_COMPUTED_AT_DEPOSITION",
      "byte_size": 26000000,
      "iteration_id": "run_00001",
      "scope": "per_iteration_l0_raw"
    }
  ]
}
```

The future Commit 5 author computes the SHA-256 of every file before upload and writes the values into the manifest before publishing the Zenodo record. The same SHA-256 values are written into the per-iteration Zenodo pointer JSON files under `competitions/glioblastoma-1min-trial/data/`.

## Zenodo Pointer JSON Files

The 1-minute variant authors several Zenodo pointer JSON files that live in the GitHub repository and reference Zenodo deposition contents by DOI and SHA-256. The pointer files are tiny (1 KB each) and serve as the bridge between the committed L1 to L3 aggregates and the Zenodo-archived L0 raw.

The full list of pointer files committed in the 1-minute variant output tree:

| Pointer file | Authored at commit | Patched at commit | Points to |
|--------------|---------------------|---------------------|-----------|
| `data/sensor_l0_raw_4arm.zenodo_pointer.json` | Commit 2 | Commit 5 | Release-aggregate L0 raw across 16 iterations |
| `data/xyz_trace_4arm.zenodo_pointer.json` | Commit 3 | Commit 5 | Per-arm xyz command trace, single canonical iteration |
| `data/iterations/run_00001_L0_raw.zenodo_pointer.json` through `run_00016_L0_raw.zenodo_pointer.json` | Commit 4 | Commit 5 | Per-iteration L0 raw |

Each pointer file uses the schema defined in `commit_02_sensors_1min.md` File 8 or in `commit_04_iterations_1min.md` File 10.

## Deposition Workflow (executed at Commit 5)

The future Commit 5 author runs the following workflow after the L1 to L3 aggregates and the L0 raw Parquet files are produced:

```
# 1. Compute SHA-256 for every file to be deposited
python -m src.zenodo.compute_sha256 --input data/iterations/ --output deposition_manifest.json

# 2. Reserve a Zenodo DOI (uses ZENODO_API_TOKEN from environment)
python -m src.zenodo.reserve_doi --metadata zenodo_metadata.json --output zenodo_record.json

# 3. Upload all files to the reserved Zenodo record
python -m src.zenodo.upload_files --record-id $(jq -r .record_id zenodo_record.json) \
  --files data/iterations/run_*_L0_raw_4arm.parquet \
          data/iterations/run_*_xyz_trace_4arm.parquet \
          deposition_manifest.json

# 4. Publish the Zenodo record (triggers DOI registration)
python -m src.zenodo.publish --record-id $(jq -r .record_id zenodo_record.json)

# 5. Patch every pointer JSON file in the GitHub repository
python -m src.zenodo.patch_pointers --record-id $(jq -r .record_id zenodo_record.json) \
  --doi $(jq -r .doi zenodo_record.json) \
  --pointers data/sensor_l0_raw_4arm.zenodo_pointer.json \
             data/xyz_trace_4arm.zenodo_pointer.json \
             data/iterations/run_*_L0_raw.zenodo_pointer.json \
  --manifest deposition_manifest.json

# 6. Commit the patched pointers to the same v3.9.1 PR (this is part of Commit 5)
git add data/*.zenodo_pointer.json data/iterations/*.zenodo_pointer.json
git commit -m "fix: patch zenodo pointers with v3.9.1 DOI and SHA-256 hashes"
```

The Zenodo helper modules (`src/zenodo/compute_sha256.py`, `reserve_doi.py`, `upload_files.py`, `publish.py`, `patch_pointers.py`) are part of the future Commit 5 deliverable; they are small (under 200 lines each) and are not listed as separate items in the Commit 5 file table because they are implementation details of `compare_agent_1min.py` and the `zenodo` optional extras dependency.

## Authentication

The Zenodo API requires a personal access token (`ZENODO_API_TOKEN`). The token is read from `os.environ["ZENODO_API_TOKEN"]` and must never be written to any committed file. The future Commit 5 author must verify that no `.env`, `.env.local`, or similar file containing the token is staged before pushing.

## Verification After Deposition

The future Commit 5 author must run the following verification block after the Zenodo deposition completes and after the pointer patching commit lands:

```
# 1. Verify every pointer file has been patched (no PLACEHOLDER strings remain)
grep -r PLACEHOLDER competitions/glioblastoma-1min-trial/data/ && echo "ERROR: unpatched pointer found" || echo "OK: all pointers patched"

# 2. Verify SHA-256 of a sample local L0 file matches the Zenodo manifest
sha256sum data/iterations/run_00001_L0_raw_4arm.parquet | awk '{print $1}'
jq -r '.files[] | select(.filename == "run_00001_L0_raw_4arm.parquet") | .sha256' deposition_manifest.json

# 3. Verify the Zenodo DOI resolves
curl -sI "https://doi.org/10.5281/zenodo.$(jq -r .record_id zenodo_record.json)" | head -1
```

If any verification step fails, the future Commit 5 author must re-run the patching step and re-commit. The Commit 6 error fix pass also re-runs this verification block.

## Source Files Cited

- `competitions/instructions/one_minute_variant/file_size_pyramid_1min.md`. Source for the L0 raw 26 MB per iteration and 416 MB total figures that drive the Zenodo deposition size.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the per-iteration L0 raw schema that the Zenodo deposition stores.
- `competitions/instructions/one_minute_variant/commit_02_sensors_1min.md`. Source for the release-aggregate Zenodo pointer schema patched by this protocol.
- `competitions/instructions/one_minute_variant/commit_03_xyz_4arm.md`. Source for the per-arm xyz trace Zenodo pointer schema patched by this protocol.
- `competitions/instructions/one_minute_variant/commit_04_iterations_1min.md`. Source for the per-iteration Zenodo pointer schema patched by this protocol.
- `competitions/instructions/one_minute_variant/commit_05_competition_1min.md`. Source for the Commit 5 deposition workflow and the verification block.
- `competitions/instructions/competition_protocol.md`. Source for the per-release snapshot pattern that the `releases/v3.9.1/zenodo_doi.txt` file extends.
- `competitions/instructions/file_format_conventions.md`. Source for the JSON conventions used by the manifest and pointer files.
