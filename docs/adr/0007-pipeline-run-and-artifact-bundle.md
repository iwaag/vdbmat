# ADR-007: Pipeline Run and Artifact Bundle

- **Status:** Proposed
- **Date:** 2026-07-01
- **Decision owners:** VBDMAT maintainers
- **Phase:** 1, Step 1

## Context

Phase 0 produced canonical volumes and Zarr assets, but each proof was a separate
script with no single reproducible run record. Phase 1 must let a user run *one*
deterministic pipeline from either supported input (ADR-006) and obtain an inspectable,
validated, repeatable, exportable bundle: source and effective fields, configuration,
provenance, diagnostics, and checksums.

The bundle must **compose** existing schema 1.0.0 assets, not reinterpret them. It must
publish atomically (a crash must not leave a valid-looking partial run), define
overwrite/resume behaviour, chain provenance from input through mapping and export, and
make "reproducible rerun" precise by separating the deterministic scientific payload
from wall-clock timestamps.

## Decision

### D1. Stage graph

The Phase 1 pipeline is a fixed, typed, deterministic sequence:

```text
load/voxelize -> validate material -> persist material
              -> map optics -> validate optical -> persist optical
              -> summarize -> optional export
```

Each stage has typed inputs and outputs (Python objects), a stable name, and a status
(`ok` / `skipped` / `failed`). `load/voxelize` uses the ADR-006 reader/voxelizer;
`map optics` uses `vbdmat.optics` with an `OpticalMappingConfig`; persistence uses the
existing `vbdmat.io.zarr` API unchanged. Optional `export` (ADR-008/Step 8) consumes the
**persisted/restored** `optical.zarr`, never a hidden in-memory optical volume. No
renderer state enters any stage before `export`.

### D2. Versioned pipeline configuration

Configuration is defined in ADR (see the pipeline-config decision in Step 5) and carried
into the run as `config.json` verbatim. The **configuration digest** is
`sha256:` over the canonical (sorted-key, tight-separator) JSON of the configuration —
the same canonicalization already used by `OpticalMappingConfig.canonical_json()`. The
optical mapping identity inside the run is the existing `OpticalMappingConfig.digest`.

### D3. Run identifier

`run_id` is derived deterministically from the scientific inputs, not from wall-clock
time:

```text
run_id = "run-" + first 16 hex of sha256(
    config_digest + "\n" + input_payload_sha256 + "\n" + mapping_digest )
```

Two runs with identical configuration, identical input payload checksum, and identical
mapping digest therefore share a `run_id` and must produce equal scientific artifacts
(D8). `run_id` contains no timestamp.

### D4. Bundle layout

```text
run/
  run.json                # manifest: stage status, schema versions, links, checksums
  config.json             # exact canonical pipeline configuration
  source/                 # copy of the input (manifest+payload, or mesh + resolved args)
  material.zarr/          # schema 1.0.0 MaterialLabelVolume (ADR-004, unchanged)
  optical.zarr/           # schema 1.0.0 OpticalPropertyVolume (ADR-004, unchanged)
  diagnostics/
    validation.json       # per-asset validation results (stable schema)
    summary.json          # geometry, material counts, field ranges, digests
  exports/                # optional; created only when export stages run
    mitsuba/
    openvdb/
```

`material.zarr` and `optical.zarr` use ADR-004 exactly. `run.json` **links** assets and
records their checksums; it never duplicates or reinterprets canonical metadata that
already lives in the Zarr manifests.

### D5. `run.json` contents

`run.json` is a JSON object with a stable schema `vbdmat.run/1.0.0`:

- `schema`: `{ "name": "vbdmat.run", "version": "1.0.0" }`;
- `run_id`;
- `created_utc`: ISO-8601 UTC timestamp — **isolated** from the deterministic payload
  and excluded from content comparison (D8);
- `config_digest`, `input_payload_sha256`, `mapping_digest`;
- `input`: kind (`direct-voxel` | `mesh`), original relative path, source checksum;
- `stages`: ordered list of `{ name, status, started/…optional }` — status only is
  content-relevant;
- `assets`: for each of `config.json`, `source/*`, `material.zarr`, `optical.zarr`,
  `diagnostics/*`, and any `exports/*`: `{ path (relative), schema, sha256, size_bytes }`.
  Zarr asset checksum is a deterministic digest over the store's array bytes + manifest
  (defined by the Step 6 hashing helper), not a single-file hash;
- `provenance`: the chain in D6;
- `versions`: `vbdmat` version, and adapter/renderer versions when export ran.

### D6. Provenance chaining

Provenance is chained end to end and mirrored into the canonical assets' `Provenance`:

```text
input(source id + payload sha256)
  -> [voxelization params, if mesh]
  -> material.zarr (sources include input checksum + voxelization identity)
  -> mapping (OpticalMappingConfig.digest)
  -> optical.zarr (sources include material provenance + mapping digest)
  -> run.json (config_digest ties the whole run)
  -> exports (adapter + renderer version recorded in run.json)
```

Each canonical volume's `Provenance.configuration_digest` is set to the run
`config_digest`; `Provenance.sources` carries the upstream checksums/identities.

### D7. Atomic, failure-safe publication

1. Build the entire bundle inside a sibling temporary directory
   (`<output>.tmp-<run_id>` on the **same filesystem** as `<output>`).
2. Write and validate every asset there; run validation stages against the *persisted*
   Zarr, not just in memory.
3. Publish only on full success by an atomic directory rename to `<output>`.
4. On any failure or interrupt, the temp directory is removed (or left as
   `*.tmp-*` for debugging) and **no** valid `run/` appears. A partial run never looks
   complete because `run.json` is written last and the publish is a single rename.

### D8. Overwrite, resume, and reproducible rerun

- **Overwrite** requires an explicit flag (ADR-008 `--overwrite`). Even then the new
  bundle is built in the temp directory and validated first; the previous valid run is
  replaced by rename only after the replacement validates, so overwrite never destroys a
  good run before a good run exists.
- **Resume** is **not** supported in Phase 1: a run is atomic and recomputed whole. (A
  content-addressed `run_id` makes an unchanged rerun cheap to detect but it is still
  recomputed and compared.)
- **Reproducible rerun** means: two runs with equal `config_digest`,
  `input_payload_sha256`, and `mapping_digest` produce byte-equal `material.zarr` /
  `optical.zarr` array bytes and equal `diagnostics/summary.json`, and equal recorded
  `assets[*].sha256`. Only `created_utc` (and any human timing note) may differ. Tests
  compare everything except those isolated timestamp fields.

### D9. Forward-compatible manifest behaviour

`run.json` and `diagnostics/*` carry explicit schema versions. A reader accepts a
compatible major and may ignore unknown *optional* keys, but the `assets` list, digests,
and stage statuses are required. Canonical Zarr assets remain governed by ADR-004 and
schema 1.0.0; the run bundle adds **no** new interpretation of volume metadata.

## Worked Example: window-coupon run bundle

Input: the direct-voxel window coupon (`vbdmat.voxels/1.0.0`); mapping:
`phase0-provisional-materials-v1` (digest `sha256:…`); no export requested.

```text
run/
  run.json
  config.json
  source/
    window-coupon.voxels.json
    window-coupon.material_id.npy
  material.zarr/
  optical.zarr/
  diagnostics/
    validation.json
    summary.json
```

`summary.json` (shape independent of the implementation under test):

```json
{
  "schema": { "name": "vbdmat.summary", "version": "1.0.0" },
  "geometry": { "shape_zyx": [nz, ny, nx], "voxel_size_xyz_m": [sx, sy, sz] },
  "material": {
    "counts": { "0": n0, "1": n1, "2": n2, "3": n3 },
    "palette": ["background", "transparent", "white", "black"]
  },
  "optical": {
    "sigma_a_range_per_m": [min, max],
    "sigma_s_range_per_m": [min, max],
    "g_range": [min, max],
    "ior_range": [min, max]
  },
  "digests": {
    "config": "sha256:…",
    "input_payload": "sha256:…",
    "mapping": "sha256:…"
  }
}
```

`run.json` records `run_id`, the isolated `created_utc`, the stage list
(`load … summarize` = `ok`, `export` = `skipped`), and one `assets` entry per file with
its relative path, schema, `sha256`, and `size_bytes`. Re-running the same input and
config reproduces every field except `created_utc`.

The mesh stepped-wedge run bundle is identical except `input.kind = "mesh"`, `source/`
holds the STL plus the resolved `{unit, voxel_size, material_id, placement}`, and the
`stages` list begins with `voxelize`.

## Rejected alternatives

- **Single-file archive (`.zip`/`.tar`).** Rejected for Phase 1: a directory keeps Zarr
  stores directly inspectable and partial-readable (ADR-004) and makes atomic
  publication a simple directory rename.
- **Timestamp in `run_id` or asset payloads.** Rejected: it would make every rerun
  differ and defeat the reproducibility contract; timestamps live only in isolated
  `run.json` fields.
- **Resume/incremental stages in Phase 1.** Rejected: atomic whole-run recomputation is
  simpler to reason about and to prove reproducible; incrementalism is deferred.
- **Duplicating geometry/palette into `run.json`.** Rejected: the Zarr manifests are the
  single source of truth; `run.json` only links and checksums them.
- **Publishing in place.** Rejected: an interrupted in-place write can look complete;
  temp-dir build + atomic rename prevents partial valid-looking bundles.

## Consequences

- The core environment completes load/voxelize → validate → persist → map → validate →
  persist → summarize with **no** renderer installed; export is strictly optional and a
  renderer failure is attributed to the export stage without corrupting canonical assets.
- Reproducibility is testable by digest comparison (Step 9), and the atomic publish is
  testable by fault injection (Step 6/9).
- Provenance is queryable from `run.json` and mirrored in each asset's `Provenance`.

## Compliance checks for Step 6

- Both input paths complete the same typed stage sequence.
- Persisted material/optical assets exactly equal the stage outputs.
- Provenance links input checksum, voxelization (if used), mapping digest, and config.
- An interrupted/failed run publishes no valid-looking bundle.
- Overwrite never destroys a previous valid run before the replacement validates.
- Two identical runs have equal scientific artifacts and declared checksums; only
  `created_utc` differs.
- An export failure does not corrupt canonical artifacts and is attributed to `export`.
