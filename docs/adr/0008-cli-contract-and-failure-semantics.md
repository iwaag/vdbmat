# ADR-008: CLI Contract and Failure Semantics

- **Status:** Accepted
- **Date:** 2026-07-01
- **Decision owners:** VDBMAT maintainers
- **Phase:** 1, Step 1

## Context

Phase 1 must let a user run the whole workflow — import, voxelize, convert, inspect,
validate, run, export — from a command line, from an *installed* package outside the
repository, without writing Python. The CLI must be a thin outer layer over package
APIs (no parallel scientific logic in scripts), must separate machine JSON from human
diagnostics, must fail with documented exit codes instead of tracebacks for expected
user errors, and must never silently "fix" input (no inferred units, transposes, casts,
ID remaps, or unauthorized overwrites).

CLI stability here is scoped to Phase 1 examples and docs; it is not yet the long-term
public compatibility promise (Phase 7).

## Decision

### D1. Entry point and command set

A console entry point `vdbmat` (declared in `pyproject.toml`, backed by
`vdbmat.cli.main`) provides:

```text
vdbmat import-voxels MANIFEST OUTPUT
vdbmat voxelize MESH OUTPUT --unit U --voxel-size SX[,SY,SZ] --material-id ID [--placement FILE]
vdbmat convert MATERIAL_ZARR OUTPUT [--mapping NAME|FILE]
vdbmat inspect ASSET [--json]
vdbmat validate ASSET [--json]
vdbmat run CONFIG
vdbmat export {mitsuba|openvdb} OPTICAL_ZARR OUTPUT
```

Exact names/flags may be refined during implementation, but none of these capabilities
may disappear. Every command calls package APIs (`vdbmat.io`, `vdbmat.voxelize`,
`vdbmat.optics`, `vdbmat.pipeline`, `vdbmat.exporters`); the CLI contains no conversion
or scientific logic of its own.

- `import-voxels` — ADR-006 direct-voxel manifest → `material.zarr`.
- `voxelize` — ADR-006 mesh + explicit `--unit/--voxel-size/--material-id` →
  `material.zarr`. Missing required flag ⇒ usage error (never a default).
- `convert` — `material.zarr` → `optical.zarr` via a mapping (default
  `phase0-provisional-materials-v1`).
- `inspect` / `validate` — any canonical asset or run bundle; `--json` for machine
  output.
- `run` — execute the full ADR-007 pipeline from a config file, producing a run bundle.
- `export` — ADR-007/Step 8 export of a restored `optical.zarr` to a renderer target.

### D2. Output discipline

- **stdout** carries machine output only: with `--json`, exactly one JSON document and
  no human log text; without `--json`, a concise human summary. Never mix them.
- **stderr** carries all human-oriented diagnostics, progress, warnings, and error
  messages.
- This lets `vdbmat inspect asset --json > out.json` be parsed without contamination.

### D3. Exit codes

| Code | Category | Examples |
| ---: | --- | --- |
| `0` | success | command completed and validated |
| `2` | usage error | missing/invalid argument, unknown command, missing required `--unit` |
| `3` | validation error | schema/geometry/palette/mesh-topology/manifest violation |
| `4` | I/O error | missing file, unreadable path, checksum mismatch, path traversal |
| `5` | conversion/pipeline error | mapping/stage failure on otherwise valid input |
| `6` | optional-dependency error | requested export needs Mitsuba/OpenVDB/Blender that is absent |

Exit code `1` is reserved for unexpected internal errors (bug); those may print a
traceback. Categories `2–6` are *expected* outcomes: they print a clear, field-oriented
message to stderr and **no traceback**, unless `--debug` (or `VDBMAT_DEBUG=1`) is set,
which re-raises the full traceback for developers.

### D4. Overwrite policy

Commands that write an `OUTPUT` refuse to overwrite an existing path unless
`--overwrite` is passed; otherwise they exit `2` with a message naming the existing
path. Pipeline `run` overwrite additionally honours ADR-007 D8 (build-and-validate in a
temp dir, replace by rename only after the replacement validates). No command ever
destroys existing output without explicit authorization.

### D5. Inspection surface

`inspect` (and `--json`) exposes, as available for the asset type: schema name/version;
geometry (`shape_zyx`, `voxel_size_xyz_m`, `local_to_world`); units and semantic
axis/basis names; provenance; per-field ranges (optical) and per-material counts
(material); and, for `export`, renderer/adapter **capability diagnostics**. `validate`
runs the canonical validators and reports pass/fail per check. JSON output has a stable,
documented shape.

### D6. Optional-dependency and capability diagnostics

Optional renderer/native dependencies are never required for the canonical pipeline. If
an `export` target's dependency is missing, the command exits `6` with actionable
install/usage instructions (which environment/group provides it) and does **not** fail
any canonical stage. Capability reports from the existing adapters are surfaced in CLI
output and copied into the run bundle (ADR-007).

### D7. Installed-package and path robustness

- The CLI must work from an installed wheel run in an arbitrary working directory
  outside the repository root; it must not rely on editable installs or
  repository-relative imports.
- Paths containing spaces and other shell-legal characters must work.
- `--help` for every command documents units, defaults, the provisional/uncalibrated
  nature of optical coefficients, and Phase 1 non-goals (no physical-accuracy claim).

### D8. API equivalence

For the same inputs, a CLI command and the corresponding package API produce identical
canonical results (same Zarr bytes, same validation outcome, same summary). The CLI adds
argument parsing, output formatting, and exit-code mapping only.

## Worked Example: commands and outputs

Direct-voxel path (core environment, no renderer):

```text
vdbmat import-voxels window-coupon.voxels.json out/material.zarr
vdbmat convert out/material.zarr out/optical.zarr
vdbmat inspect out/optical.zarr --json
vdbmat validate out/optical.zarr --json
```

Mesh path (explicit units required):

```text
vdbmat voxelize stepped-wedge.stl out/wedge.material.zarr \
    --unit mm --voxel-size 0.001 --material-id 1
```

Full pipeline from a config file (produces an ADR-007 bundle):

```text
vdbmat run window-coupon.run.json      # -> run/ bundle
vdbmat inspect run/ --json             # summarizes run.json + assets
```

Optional export (may exit 6 if the dependency is absent):

```text
vdbmat export mitsuba run/optical.zarr run/exports/mitsuba
```

Expected failure behaviours (no traceback, documented code):

```text
vdbmat voxelize m.stl out.zarr --voxel-size 0.001 --material-id 1
  # exit 2: missing required --unit

vdbmat import-voxels tampered.voxels.json out.zarr
  # exit 4: payload SHA-256 mismatch: <path>

vdbmat import-voxels out-of-tree.voxels.json out.zarr
  # exit 4: payload path escapes manifest directory: <path>

vdbmat import-voxels wrong-dtype.voxels.json out.zarr
  # exit 3: arrays.material_id must be uint16, got int32

vdbmat convert out/material.zarr out/optical.zarr   # (out/optical.zarr exists)
  # exit 2: refusing to overwrite existing path: out/optical.zarr (use --overwrite)

vdbmat export mitsuba run/optical.zarr run/exports/mitsuba   # (mitsuba absent)
  # exit 6: mitsuba is not installed; install the 'mitsuba' dependency group
```

`--json` on any of the above still writes structured output to stdout only; the
human-readable error text is on stderr.

## Rejected alternatives

- **Single `1`/`0` exit scheme.** Rejected: subprocess tests and scripts must
  distinguish usage vs validation vs I/O vs conversion vs optional-dependency failures.
- **Tracebacks for user errors.** Rejected: they are noise for expected input mistakes;
  `--debug`/`VDBMAT_DEBUG` preserves them for developers.
- **Mixing JSON and logs on stdout.** Rejected: it breaks machine parsing; diagnostics
  go to stderr.
- **Silent overwrite / inferred units / auto-transpose.** Rejected: each is a named stop
  condition; the CLI must fail loudly instead.
- **CLI implementing pipeline logic directly.** Rejected: the CLI must call package APIs
  so `examples/phase1/` is never the only supported entry point and API/CLI stay
  equivalent.

## Consequences

- Subprocess tests can assert each exit-code category and stdout/stderr separation
  (Step 7/9).
- An installed wheel runs the full direct-voxel pipeline from a temporary directory
  (Step 11).
- Optional exports degrade to a single actionable `6` without touching canonical output.

## Compliance checks for Step 7

- Subprocess tests cover success and each of exit codes `2–6`.
- `--json` output is stable, parseable, and free of human log text.
- Paths with spaces work.
- No command overwrites without `--overwrite`.
- `--help` documents units, defaults, provisional coefficients, and non-goals.
- An installed wheel runs the full direct-voxel pipeline from another directory.
- CLI results equal direct package-API results exactly.
