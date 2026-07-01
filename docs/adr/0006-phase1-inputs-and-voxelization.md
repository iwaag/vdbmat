# ADR-006: Phase 1 Inputs and Voxelization

- **Status:** Proposed
- **Date:** 2026-07-01
- **Decision owners:** VBDMAT maintainers
- **Phase:** 1, Step 1

## Context

Phase 0 froze the canonical volume contracts (ADR-001 through ADR-005): metre world
space, semantic XYZ over ZYX storage, cell-centred sampling, immutable
`MaterialLabelVolume` / `MaterialMixtureVolume` / `OpticalPropertyVolume`, and exact
Zarr v3 persistence. Phase 0 built those volumes only from Python fixture code
(`vbdmat.fixtures.synthetic`).

Phase 1 must let a user supply material placement and simple geometry *without writing
Python*, and turn that input into a validated canonical `MaterialLabelVolume`. Two
questions must be answered before any reader or voxelizer is implemented:

1. Which external input formats does Phase 1 accept, and with what exact unit, axis,
   transform, palette, and provenance meaning?
2. How is a triangle mesh converted to cell-centred occupancy deterministically, with
   analytically checkable results and explicit rejection of unsupported topology?

An implicit transpose, silent unit guess, silent ID remap, or ambiguous boundary rule
would reintroduce exactly the failure modes ADR-001 was written to prevent. The
convenience layer must therefore *adapt* input into the existing canonical types, never
weaken them.

### Data inventory

No printer-vendor voxel file and no representative mesh were available when Step 1
closed. A repository search found no `.stl`, `.vox`, `.ply`, or non-test `.npy` payload
under version control, and no mesh library (`trimesh`, `numpy-stl`, `meshio`,
`pyvista`, `open3d`) in `uv.lock`. Section 4 of the Phase 1 plan therefore applies: the
default JSON + `.npy` material-label interchange and a watertight single-solid STL
baseline are adopted. If a real printer/material voxel format is supplied before
implementation begins, a narrow reader for that format is preferred and this manifest
format is retained as the canonical test fixture.

## Decision

### D1. Direct material-voxel interchange: `vbdmat.voxels/1.0.0`

The direct-voxel input is a UTF-8 JSON manifest plus one NumPy `.npy` label payload:

```text
sample.voxels.json      # manifest
sample.material_id.npy  # uint16[z, y, x] label payload
```

The manifest is a JSON object with these required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | string | Must equal `"vbdmat.voxels"`. |
| `format_version` | string | `MAJOR.MINOR.PATCH`; Phase 1 accepts major `1`. |
| `asset_type` | string | Must equal `"material-label"`. |
| `payload.path` | string | Relative POSIX path to the `.npy`, resolved only beneath the manifest directory. |
| `payload.sha256` | string | 64 lowercase hex digits of the payload bytes. |
| `payload.dtype` | string | Must equal `"uint16"`. |
| `payload.dimensions` | array | Must equal `["z", "y", "x"]`. |
| `shape_zyx` | array of 3 int | `(nz, ny, nx)`, each `> 0`; must equal the loaded array shape. |
| `voxel_size_xyz_m` **or** `voxel_size` | see D3 | Physical cell size, with explicit unit. |
| `local_to_world` | 4×4 array | Row-major rigid transform per ADR-001. |
| `materials` | array | Ordered palette (D4). |
| `source` | object | Provenance identity (D5). |

Unknown top-level keys are rejected in Phase 1 (strict parse) so that a future field is
never silently ignored; forward compatibility is handled by the major-version gate, not
by leniency. The `.npy` is loaded with `allow_pickle=False`. The manifest never causes
a transpose, cast, unit inference, or ID remap: every one of those is a validation
failure, not a repair.

### D2. Mesh interchange: watertight single-solid STL, repository-owned reader

Phase 1 accepts one watertight, consistently oriented, single-solid triangle STL
(binary or ASCII). STL carries neither trustworthy units nor material identity, so the
following are **required** CLI/config values and their absence is a hard error, never a
silent default:

- `--unit` — source length unit of the STL coordinates (`m` or `mm` for Phase 1);
- `--voxel-size` — target isotropic or anisotropic voxel size in metres;
- `--material-id` — the single interior material ID (must be a declared non-background
  palette entry);
- optional `--placement` — a rigid `local_to_world` (defaults to identity);
- inside/outside policy is fixed by D7 (not user-selectable in Phase 1).

**Dependency policy.** Phase 1 uses a **repository-owned** narrow STL reader and owns
all topology inspection and voxelization semantics. No third-party mesh library is
added. Rationale: binary/ASCII STL parsing is small and fully specified; the plan's
stop conditions treat mesh-dependency licensing/ABI/core-install cost as a gate; and
voxelization correctness must be VBDMAT-owned and analytically tested regardless of any
library. If a future solid class makes an external parser demonstrably safer than owned
code, the addition and its uv-lock impact will be recorded in a superseding decision.
This choice adds **zero** new runtime dependencies to the core environment.

### D3. Units

The canonical unit is metres (ADR-001). Convenience input may declare another unit at
the boundary and must convert explicitly:

- Direct-voxel manifest: either `voxel_size_xyz_m: [sx, sy, sz]` (already metres) **or**
  `voxel_size: {"value": [sx, sy, sz], "unit": "mm"}`. Exactly one form is present.
  Supported units: `m`, `mm`. `mm` is multiplied by `1e-3`. No other unit is inferred.
- Mesh: `--unit` gives the STL coordinate unit; vertices are multiplied to metres
  before domain construction. `--voxel-size` is always metres.

A missing or unsupported unit is a field-oriented error. This satisfies the stop
condition against silent unit inference.

### D4. Material palette

`materials` is an ordered JSON array mapping directly onto `MaterialPalette` /
`MaterialDefinition`:

```json
{ "material_id": 0, "name": "background", "role": "background" }
```

Rules (enforced by the existing `MaterialPalette`): background ID `0` with role
`background` must be present; only ID `0` may be `background`; IDs are unique and in
`[0, 65535]`; optional `external_id` and JSON-compatible `metadata` are preserved. Every
label value in the payload must reference a declared ID (checked by
`MaterialLabelVolume`). For meshes, the palette is `{0: background, <material-id>:
declared}` and the interior is filled with `--material-id`.

### D5. Provenance and checksums

`source` populates the canonical `Provenance`:

```json
{
  "generator": "vbdmat.voxels",
  "generator_version": "1.0.0",
  "identity": "window-coupon",
  "notes": "provisional research interchange"
}
```

The reader records the *format identity and payload checksum* in provenance so the
source is chainable downstream: `provenance.sources` includes
`"vbdmat.voxels/1.0.0"` and `"sha256:<payload digest>"`, and `notes` may carry the
declared `identity`. `created_utc` and `configuration_digest` are set later by the
pipeline (ADR-007), not by the reader. Payload integrity is verified by recomputing
SHA-256 over the raw `.npy` bytes and comparing to `payload.sha256` **before** the array
is loaded.

### D6. Path safety

`payload.path` must be relative, must not be absolute, and must not escape the manifest
directory. Resolution rule: `resolved = realpath(manifest_dir / payload.path)`; reject
unless `resolved` is inside `realpath(manifest_dir)`. Reject any component equal to `..`,
any absolute path, any drive/UNC prefix, and any symlink that resolves outside the
manifest directory. This prevents a manifest from reading arbitrary files.

### D7. Voxelization domain, sampling, and inside/outside rule

Mesh voxelization is a **dense, cell-centred** reference method (correctness over
speed):

1. **Metric conversion.** Multiply STL vertices by the D3 unit factor to metres, then
   apply `--placement` if the placement is expressed in the mesh's own local frame; by
   default the mesh already lives in local grid space and `--placement` becomes the
   volume's `local_to_world`.
2. **Domain.** Compute the axis-aligned bounding box `[min_xyz, max_xyz]` of the
   metre-space vertices. Choose grid origin at `floor(min / s) * s` per axis and extent
   `n_a = ceil((max_a - origin_a) / s_a)` cells, then add **one cell of padding on each
   side** per axis so no surface cell is clipped. `shape_zyx = (nz, ny, nx)`. The volume
   `local_to_world` translation is set to the padded minimum corner (ADR-001: transform
   translation is the world position of the minimum local grid corner).
3. **Classification.** A cell is *inside* iff its **centre** (D of ADR-001,
   `c_local = ((x+0.5)s_x, (y+0.5)s_y, (z+0.5)s_z)`) is inside the closed mesh, decided
   by a deterministic ray-crossing parity test: cast a ray along `+X` from the centre
   and count boundary crossings; odd ⇒ inside.
4. **Assignment.** Inside cells receive `--material-id`; all others receive background
   `0`.

### D8. Boundary and tie-breaking

The parity test uses an explicit tolerance `EPS = 1e-9` m (matching ADR-001 geometry
tolerance) and these deterministic rules so the result is independent of triangle
ordering:

- A ray that passes exactly through a shared edge or vertex is perturbed by re-casting
  along a fixed alternate direction sequence (`+X`, then `+Y`, then `+Z`) until no
  crossing is within `EPS` of an edge/vertex; the first clean axis decides. The axis
  sequence is fixed, so the outcome is deterministic.
- A centre lying within `EPS` of the surface is classified **inside** (closed-solid
  convention). This makes occupancy for an axis-aligned box whose faces fall exactly on
  cell-centre planes fully predictable (see worked example M).
- Counting is by signed parity only; self-consistent winding is assumed and verified in
  D9.

### D9. Topology checks and rejection cases

Before voxelizing, the reader inspects topology and **rejects** (clear, field-oriented
error; no repair) any of:

- non-triangular or zero-area (degenerate) faces;
- an empty mesh (no triangles);
- an **open / non-watertight** surface — every undirected edge must be shared by exactly
  two triangles;
- a **non-manifold** edge — shared by more than two triangles;
- inconsistent orientation — the two triangles on a shared edge must traverse it in
  opposite directions;
- multiple disconnected solids (Phase 1 is single-solid, single-material);
- self-intersection **when detectable** by the owned checks (not guaranteed exhaustive
  in Phase 1; documented as a limitation).

Non-finite vertex coordinates and non-rigid placements are rejected via the existing
geometry/transform validators.

### D10. Deterministic numeric tolerances

- Geometry/transform validation: absolute `1e-9` (inherited from ADR-001).
- Mesh classification edge/vertex tolerance `EPS = 1e-9` m.
- Unit conversion is exact rational scaling (`×1e-3` for mm) applied in `float64`.
- Voxelization is deterministic and order-independent: identical solids (up to triangle
  reordering and equivalent unit expression) yield byte-identical label arrays.

## Worked Example V: Direct multi-material voxel input

A deliberately tiny, axis-asymmetric grid used to exercise axis reversal. Semantic
extents: `nx = 4`, `ny = 3`, `nz = 2`, so `shape_zyx = (2, 3, 4)`.

```text
voxel_size_xyz_m = (0.00004, 0.00005, 0.00003)   # anisotropic, from ADR-001
local_to_world   = translation (0.010, 0.020, 0.030) m
```

Palette: `0 background`, `1 transparent`, `2 white`, `3 black`.

Label payload `material_id[z, y, x]` (ZYX storage; each row is one Y line of X values):

```text
z = 0:
  y=0: [1, 1, 1, 1]
  y=1: [1, 2, 2, 1]
  y=2: [1, 1, 1, 3]
z = 1:
  y=0: [1, 1, 1, 1]
  y=1: [1, 1, 1, 1]
  y=2: [1, 1, 1, 1]
```

Analytic expectations (implementation-independent):

- material counts: `background 0`, `transparent 1 → 21`, `white 2 → 2`, `black 3 → 1`
  (24 cells total);
- the single black marker is at semantic `(x=3, y=2, z=0)`, i.e. `material_id[0, 2, 3]`.
  Because it is unique across all three axes, any accidental X↔Z or Y↔Z transpose moves
  it to a different array cell and fails the check;
- the white pair is at `material_id[0, 1, 1]` and `material_id[0, 1, 2]`;
- world centre of the black marker: `(0.010 + 3.5·4e-5, 0.020 + 2.5·5e-5, 0.030 +
  0.5·3e-5) = (0.01014, 0.020125, 0.030015)` m.

`payload.sha256` is the SHA-256 of the exact `.npy` bytes produced by `numpy.save` of
this `uint16` array (recorded with the fixture in Step 4). The full Phase 1 window
coupon (Section "Representative objects") scales this asymmetric idea up to a
render-visible size.

## Worked Example M: Mesh-to-grid with known occupancy

An axis-aligned cube solid, declared in millimetres, voxelized at `s = 1 mm = 0.001 m`.
Cube corners span `[0, 3] mm` on each axis (a 3 mm cube), i.e. metre bounds
`[0, 0.003]³`.

Domain construction (D7): origin `= floor(0 / 0.001)·0.001 = 0`; `n_a = ceil((0.003 −
0)/0.001) = 3`; add one padding cell each side ⇒ `n_a = 5` per axis; the padded minimum
corner shifts to `−0.001` m, so `local_to_world` translation `= (−0.001, −0.001,
−0.001)` m and `shape_zyx = (5, 5, 5)`.

Cell centres along an axis are at local `−0.0005, 0.0005, 0.0015, 0.0025, 0.0035` m
(indices 0..4). The cube occupies the closed metre interval `[0, 0.003]` **in world
space**, i.e. local `[0.001, 0.004]` after the padding shift — equivalently cell centres
at padded indices `1, 2, 3` fall inside on each axis (world centres `0.0005, 0.0015,
0.0025` m ∈ `[0, 0.003]`), while padded indices `0` and `4` (world `−0.0005`, `0.0035`)
are outside.

Analytic occupancy: the interior is the `3×3×3` block of cells with padded indices
`(x, y, z) ∈ {1,2,3}³` ⇒ **27 occupied cells**, all assigned `--material-id`; the
remaining `5³ − 27 = 98` cells are background. This is exact under D8 because no cube
face coincides with a cell-centre plane. A `+1 mm` translation of the same cube shifts
the occupied block by one index and preserves the count (metamorphic check for Step 3).

The stepped-wedge representative mesh (below) provides the graded, per-layer analytic
occupancy case.

## Representative objects (Phase 1)

### Multi-material window coupon (direct voxel)

- transparent matrix, one white inclusion, one asymmetric black marker, background;
- features must **not** be symmetric across X, Y, or Z so a render exposes axis
  reversal;
- imported through the `vbdmat.voxels/1.0.0` manifest (no Python fixture construction);
- analytic expectations recorded with the Step 4 fixture: `shape_zyx`, per-material cell
  counts, marker/inclusion cell coordinates, bounds, transform, and payload SHA-256.

### Single-material stepped wedge (mesh)

- a watertight single-solid staircase whose thickness increases in fixed steps along one
  axis, giving analytically predictable per-layer occupied-cell counts and overall
  bounds;
- voxelized through D7 with explicit `--unit`, `--voxel-size`, `--material-id`;
- expected bounds and per-step occupancy recorded with the Step 4 fixture.

## Maximum Phase 1 reference size

A safety bound (not a performance claim): **no axis exceeds 128 cells** and **total
cells do not exceed 2,000,000**. Voxelization and pipeline runs assert this bound;
exceeding it is a usage error directing the user to a coarser voxel size. Peak memory
and runtime at this bound are recorded in Step 3 / Step 11.

## Rejected alternatives

- **Adopt a printer-vendor voxel format now.** Rejected: no representative file or
  stable format contract is available (plan stop condition). The transparent JSON+`.npy`
  interchange is used and clearly labelled as a research format, not a printer format.
- **Add a mesh library (trimesh/numpy-stl) to core.** Rejected for Phase 1: STL parsing
  is small, and owning topology/voxelization semantics is required regardless; a new core
  dependency carries licensing/ABI/install cost the stop conditions guard against.
- **Voxelize by triangle rasterization / surface-only marking.** Rejected: cell-centre
  inside/outside gives unambiguous, analytically checkable solid occupancy; surface
  marking leaves interior fill and boundary ownership ambiguous.
- **Corner sampling or subvoxel coverage.** Rejected: inconsistent with ADR-001
  cell-centred piecewise-constant sampling and with schema 1.0.0 label volumes.
- **Silent unit/axis/ID defaults for convenience.** Rejected: every such default is a
  named stop condition; missing unit/material must fail loudly.

## Consequences

- Input parsing depends only on `core` and NumPy; no Zarr or renderer import is needed
  to import or voxelize (verified in Steps 2–3).
- `MaterialLabelVolume`, `GridGeometry`, `MaterialPalette`, and `Provenance` are reused
  unchanged; schema 1.0.0 is not modified.
- Mixture input is **not** a new external format in Phase 1; canonical mixture volumes
  remain internal/Zarr-only unless a real sample forces the question.
- Boundary determinism (D8) and topology rejection (D9) are testable against analytic
  solids in Step 3, satisfying the plan's boundary stop condition.

## Compliance checks for Steps 2–3

- Valid direct input reproduces exact selected cells, geometry, palette, and checksum;
  malformed JSON, wrong major version, wrong dtype/shape, undeclared IDs, missing
  payload, checksum mismatch, path traversal, non-finite geometry, and invalid transform
  each fail with a field-oriented error.
- Import/voxelize require no Zarr or renderer dependency; repeated reads are structurally
  equal.
- Worked Example V imports byte-for-byte; the black marker stays at `material_id[0,2,3]`.
- Worked Example M yields exactly 27 occupied cells; a translated cube preserves the
  count; triangle reordering and mm/m expressions of one solid give equal volumes.
- Open, non-manifold, degenerate, empty, multi-solid, and (when detectable)
  self-intersecting meshes are rejected clearly.
- Peak memory and runtime at the 128-cell / 2M-cell bound are recorded.
