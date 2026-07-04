# VDBMAT Logical Volume Schema 1.0

This document is the implementation-oriented summary of ADR-001 and ADR-002. It defines logical assets, not their Zarr paths or compression. ADR-004 will define persistence layout.

## Common Manifest

```text
schema:
  name: "vdbmat.volume"
  version: "1.0.0"
asset_type: "material-label" | "material-mixture" | "optical-property"
geometry:
  shape_zyx: [nz, ny, nx]
  voxel_size_xyz_m: [sx, sy, sz]
  local_to_world: 4x4 float64 rigid transform
arrays:
  <field name>:
    dtype: canonical dtype string
    dimensions: ordered dimension names
    shape: integer extents
provenance:
  generator: string
  generator_version: string
  created_utc: optional RFC 3339 string
  configuration_digest: optional "sha256:<lowercase hex>"
  sources: optional ordered string list
  notes: optional string
```

## Common Geometry Invariants

- `shape_zyx` contains three positive integers.
- `voxel_size_xyz_m` contains three finite positive numbers.
- `local_to_world` is finite, rigid, right-handed, and has homogeneous last row `[0, 0, 0, 1]`.
- Spatial arrays begin with dimensions `z`, `y`, `x` and shape `shape_zyx`.
- Values are cell-centred.
- Geometry uses metres.

## Asset Tables

### `material-label`

```text
material_id: uint16[z, y, x]
palette: ordered material definitions
```

Required checks: every ID is declared; ID 0 is the sole background entry.

### `material-mixture`

```text
fractions: float32[z, y, x, material]
material_ids: uint16[material]
palette: ordered material definitions
```

Required checks: IDs match palette order; each fraction is in `[0, 1]`; each cell sums to `1 +/- 1e-6`.

### `optical-property`

```text
sigma_a: float32[z, y, x, basis]  # m^-1, >= 0
sigma_s: float32[z, y, x, basis]  # m^-1, >= 0
g:       float32[z, y, x]         # dimensionless, -1 <= g <= 1
ior:     float32[z, y, x]         # dimensionless, > 0
optical_basis: basis definition
```

Phase 0 basis:

```text
kind: "rgb"
identifier: "linear-srgb-effective-v1"
coordinates: ["R", "G", "B"]
reference_white: "D65"
observer: "CIE-1931-2deg"
transfer: "linear"
```

## Canonical Dtypes

| Logical value | Dtype |
| --- | --- |
| Material identifier | `uint16` |
| Material fraction | `float32` |
| Optical coefficient | `float32` |
| Anisotropy | `float32` |
| Refractive index | `float32` |
| Geometry metadata and transforms | `float64` |

Implementations may compute in greater precision but must perform explicit, validated conversion to these canonical dtypes.

## Version Handling

- Reject unsupported schema major versions.
- Never infer missing required arrays.
- Never silently transpose, cast, clip, normalize, or discard fields.
- Unknown-field and minor-version persistence behavior is deferred to ADR-004.

## Normative Sources

- [ADR-001](../adr/0001-coordinates-axes-units-and-sampling.md)
- [ADR-002](../adr/0002-canonical-volume-schemas.md)
