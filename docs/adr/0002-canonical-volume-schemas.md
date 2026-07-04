# ADR-002: Canonical Volume Schemas

- **Status:** Accepted
- **Date:** 2026-06-28
- **Decision owners:** VDBMAT maintainers
- **Phase:** 0, Step 2

## Context

VDBMAT needs a renderer-neutral contract between intended material placement, future print-process models, optical mapping, persistence, and exporters. Material identifiers alone cannot describe effective appearance. Conversely, renderer-native density or shader parameters would couple the core to one consumer.

Phase 0 needs three logical asset types:

1. a discrete material-label volume;
2. a continuous material-mixture volume;
3. an effective optical-property volume.

The schema must make array dimensions, dtypes, physical units, color meaning, background behavior, versions, and provenance explicit. It must also allow a future spectral basis without changing the meaning of Phase 0 RGB data.

This ADR defines the logical data model. ADR-004 will define how it is laid out in Zarr.

## Decision

### Schema identity and versioning

Every asset declares:

```text
schema_name = "vdbmat.volume"
schema_version = "1.0.0"
asset_type = one of:
  "material-label"
  "material-mixture"
  "optical-property"
```

Schema versions use semantic versioning:

- **MAJOR** changes may reinterpret existing fields or remove compatibility;
- **MINOR** changes may add optional fields without changing existing meaning;
- **PATCH** changes clarify documentation or constraints without changing the logical data model.

Readers must reject unsupported major versions. Forward handling of later minor versions and unknown optional fields will be finalized with persistence behavior in ADR-004.

### Common metadata

Every asset contains:

- schema name, schema version, and asset type;
- geometry as defined by ADR-001;
- explicit array descriptors with name, dtype, dimensions, and shape;
- provenance;
- asset-type-specific metadata.

Required provenance fields are:

- `generator`: stable tool or library name;
- `generator_version`: version string for that generator.

Optional provenance fields include:

- `created_utc`: RFC 3339 timestamp;
- `configuration_digest`: lowercase SHA-256 digest prefixed with `sha256:`;
- `sources`: ordered source identifiers, URIs, or digests;
- `notes`: non-normative text.

Timestamps are optional so deterministic fixtures do not acquire meaningless differences. Provenance does not replace the schema version.

### Common array rules

- All array shapes must agree with `geometry.shape_zyx` for their spatial dimensions.
- Spatial dimension names are exactly `z`, `y`, and `x`, in that order.
- Additional dimensions are last and explicitly named.
- Numeric values must be finite; NaN and infinity are invalid.
- Phase 0 arrays are dense logical arrays even if a later storage layer uses chunks, compression, or sparsity.
- No reader or writer silently clips, normalizes, transposes, or casts invalid input.

## Material Palette

Material-label and material-mixture assets contain an ordered palette. Each entry has:

- `material_id`: unsigned integer in `0..65535`;
- `name`: non-empty human-readable string;
- `role`: `background` or `material`;
- optional `external_id`: printer or laboratory identifier;
- optional non-normative metadata.

Rules:

- IDs are unique.
- Material ID `0` is reserved for the background medium.
- Exactly one entry has ID `0` and role `background`.
- Palette order has no semantic effect for label volumes.
- Palette order defines the material axis for mixture volumes and is therefore normative there.
- Palette entries describe identity, not calibrated optical properties. Optical lookup data belongs to a mapping configuration or material library.

## Material-Label Volume

The required array is:

| Name | Dtype | Dimensions | Shape | Meaning |
| --- | --- | --- | --- | --- |
| `material_id` | `uint16` | `(z, y, x)` | `(nz, ny, nx)` | Palette ID occupying each cell |

Invariants:

- Every stored ID exists in the palette.
- ID `0` denotes the background medium, not missing data.
- Unused palette entries are permitted to support consistent palettes across related assets.
- There is no null or masked cell in schema 1.0.

## Material-Mixture Volume

The required arrays are:

| Name | Dtype | Dimensions | Shape | Meaning |
| --- | --- | --- | --- | --- |
| `fractions` | `float32` | `(z, y, x, material)` | `(nz, ny, nx, nm)` | Volume fraction of every palette component |
| `material_ids` | `uint16` | `(material)` | `(nm)` | Palette ID at each material-axis position |

Invariants:

- `nm >= 1`.
- `material_ids` contains every palette ID exactly once and in the same order as the palette.
- Every fraction lies in closed interval `[0, 1]`.
- Fractions at each voxel sum to `1` within absolute tolerance `1e-6`.
- Stored values must already satisfy normalization; validation does not repair them.
- Background is an explicit mixture component, so a fully empty cell has background fraction `1` and all other fractions `0`.

The schema models volume fractions. Mass fractions, droplet counts, and printer command weights require separate asset semantics and must not be mislabeled as this schema.

## Optical-Property Volume

The required arrays are:

| Name | Dtype | Dimensions | Shape | Unit | Meaning |
| --- | --- | --- | --- | --- | --- |
| `sigma_a` | `float32` | `(z, y, x, basis)` | `(nz, ny, nx, nb)` | `m^-1` | Absorption coefficient |
| `sigma_s` | `float32` | `(z, y, x, basis)` | `(nz, ny, nx, nb)` | `m^-1` | Scattering coefficient |
| `g` | `float32` | `(z, y, x)` | `(nz, ny, nx)` | `1` | Mean cosine anisotropy parameter |
| `ior` | `float32` | `(z, y, x)` | `(nz, ny, nx)` | `1` | Real refractive index |

Invariants:

- `sigma_a >= 0` componentwise.
- `sigma_s >= 0` componentwise.
- `-1 <= g <= 1`.
- `ior > 0`.
- `nb` equals the number of coordinates declared by `optical_basis`.
- Absorption and scattering use the same optical basis in schema 1.0.
- `g` and `ior` are scalar spatial fields in schema 1.0.

The endpoint values `g=-1` and `g=1` are valid canonical values but may be unsupported by a renderer's phase function. Exporters must report or reject that limitation rather than silently perturbing the field.

There are no implicit defaults for missing required fields. Background cells must contain explicit optical properties, such as zero absorption, zero scattering, zero anisotropy, and the selected ambient refractive index. The exact ambient values come from mapping configuration, not from this schema.

### Phase 0 RGB optical basis

Phase 0 uses:

```text
kind = "rgb"
identifier = "linear-srgb-effective-v1"
coordinates = ["R", "G", "B"]
reference_white = "D65"
observer = "CIE-1931-2deg"
transfer = "linear"
```

This is an effective three-channel transport approximation:

- Values are linear coefficients, not gamma-encoded sRGB color values.
- The sRGB transfer function is never applied to `sigma_a` or `sigma_s`.
- Coefficients are consumed and mixed componentwise unless a mapping model explicitly says otherwise.
- The three coefficients do not constitute a sampled physical spectrum.
- Renderers must document how this basis maps to their color-management and spectral systems.

The identifier ends in `v1` so a future improved RGB fitting convention can coexist without reinterpreting existing assets.

### Future spectral basis

Future spectral assets will preserve the spatial dimensions and use the same last `basis` dimension:

```text
kind = "spectral"
identifier = "wavelength-nm"
coordinates = [wavelength_0_nm, ..., wavelength_n_nm]
```

Wavelength coordinates must be finite, strictly increasing, and expressed in nanometres. A spectral asset will have `nb = len(coordinates)`. Adding this basis kind does not change the definition of an asset whose basis kind is `rgb`.

The spectral basis is reserved, not implemented, in Phase 0. Its final wavelength range, sampling, interpolation, and version compatibility require a later ADR.

## Interfaces and Refractive-Index Boundaries

Schema 1.0 records cell-centred `ior`, but this ADR does not assert that all consumers can reconstruct sharp boundary behavior from it. ADR-003 will decide whether a derived interface asset or boundary geometry is also required.

Until ADR-003 is accepted:

- `ior` remains required and must be preserved;
- exporters must not silently discard it;
- no interface array is part of schema 1.0;
- any experimental boundary representation is derived data, not a canonical schema extension.

If ADR-003 requires new canonical data, schema 1.0 must be revised before Phase 0 exits.

## Worked Examples

### RGB coefficient at one voxel

For array location `[z=1, y=2, x=3]`:

```text
sigma_a[1, 2, 3, :] = [12.0, 8.0, 4.0] m^-1
sigma_s[1, 2, 3, :] = [150.0, 160.0, 170.0] m^-1
g[1, 2, 3] = 0.25
ior[1, 2, 3] = 1.49
```

The coefficient `12.0` is the effective linear R-basis absorption coefficient. It is neither an encoded R color value nor a wavelength sample. The extinction coefficient for each basis component can be derived as `sigma_t = sigma_a + sigma_s`, giving `[162.0, 168.0, 174.0] m^-1` for this voxel. `sigma_t` is derived and is not stored in schema 1.0.

### Two-material mixture

For ordered palette IDs `[0, 7]`, a voxel containing 25% background and 75% material 7 stores:

```text
material_ids[:] = [0, 7]
fractions[z, y, x, :] = [0.25, 0.75]
```

The sum is exactly `1`. Reversing `material_ids` without also reversing the last axis of `fractions` changes the asset's meaning and is invalid if it disagrees with palette order.

## Rejected Alternatives

### Use material IDs as renderer output

Rejected because material identity does not encode absorption, scattering, anisotropy, refractive index, or effective process changes.

### Store all optical fields in one interleaved array

Rejected because the fields have different units, ranks, and likely access patterns. Separate named arrays are easier to validate and export.

### Store RGB on the leading axis

Rejected because spatial indexing and chunk selection should share the same leading dimensions across fields. The trailing basis axis also extends directly to spectral samples.

### Treat RGB coefficients as ordinary sRGB colors

Rejected because gamma encoding is not meaningful for physical coefficients and would make mixing and unit conversion invalid.

### Omit background from mixture fractions

Rejected because partial fill would otherwise have ambiguous normalization and empty space would require a separate mask.

### Use NaN for missing or outside cells

Rejected because background is a physical transport medium and NaN propagates unpredictably through processing and rendering.

### Permit arbitrary numeric dtypes

Rejected for schema 1.0 because dtype variation complicates cross-tool reproducibility. Computation may use higher precision internally, but canonical arrays have fixed storage dtypes.

## Consequences

- Every optical cell is self-contained with explicit coefficients, including background.
- A material palette is required even for simple label data.
- Mixture fields can be large because the material axis is dense; sparse mixture representation is deferred until measured need.
- RGB results are explicitly approximate but cannot be confused with gamma-encoded image data.
- Future wavelength samples fit the same array-rank convention.
- Step 3 and Step 4 can implement types and validators directly from the listed invariants.

## Compliance Checks for Steps 3 and 4

- Reject unknown material IDs and duplicate palette IDs.
- Require exactly one background palette entry with ID `0`.
- Reject spatial shape or dimension-order mismatches.
- Reject non-finite fields and values outside field ranges.
- Reject mixture normalization errors over `1e-6` without repairing them.
- Check basis coordinate count against array shape.
- Preserve all required metadata through copies and conversions.
- Include field paths and summary counts in validation errors.
