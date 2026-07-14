# Cross-Consumer Conformance Check

## Purpose

The cross-consumer conformance check compares contracts shared by the Mitsuba 3 and
OpenVDB/Blender Cycles adapters. It does not compare final pixels or claim that the
renderers implement the same transport model.

The command maps all six deterministic fixtures, round-trips each optical volume
through Zarr, and runs both adapters' pure conversion paths without importing Mitsuba,
OpenVDB, or Blender:

```bash
uv run python examples/native_fixtures/check_cross_consumer_conformance.py \
  .local/conformance.json
```

Optional image sanity checks can inspect already generated proof outputs:

```bash
uv run python examples/native_fixtures/check_cross_consumer_conformance.py \
  .local/conformance-with-images.json \
  --mitsuba-renders .local/mitsuba-fixtures \
  --cycles-renders .local/cycles-native \
  --cycles-width 16 --cycles-height 16
```

The explicit 16 x 16 Cycles size above describes the low-sample native smoke outputs
stored locally. Use the default 64 x 64 expectation for the documented reference
render configuration.

## Checked contracts

Every fixture records these checks independently:

- canonical ZYX storage, metre geometry, `m^-1` coefficients, and dimensionless scalar
  fields;
- exact Zarr round-trip of geometry, schema, basis, provenance, and arrays;
- exact OpenVDB XYZ-to-ZYX reconstruction of `sigma_a`, `sigma_s`, `g`, and `ior`;
- exact documented Mitsuba `sigma_t` and albedo conversion;
- equality of all eight canonical world-space domain corners through both transforms;
- canonical region values and IOR boundary face locations;
- preservation of zero-extinction background cells;
- metre scene units and numeric coefficient scale 1;
- the same scattering-weighted global `g` reduction;
- complete capability fields and explicit IOR/interface dispositions.

Failures carry one of four origin labels: `canonical`, `serialization`,
`adapter_conversion`, or `image_sanity`. This prevents a renderer conversion error from
being reported as a canonical schema or storage failure.

Optional PNG checks reject missing fixture records, unreadable/empty output, unexpected
dimensions, unsupported PNG layouts, and spatially flat images. Orientation is checked
from transforms and selected field locations before rendering; image checks are only a
gross regression signal.

## Expected adapter differences

These differences are intentional and are emitted in the JSON report rather than
hidden with fixture-specific corrections:

| Semantic | Mitsuba 3 | OpenVDB / Cycles |
| --- | --- | --- |
| RGB coefficients | RGB `sigma_t` and albedo | RGB grids retained; equal-weight scalar densities consumed |
| Spatial `g` | scattering-weighted global scalar | same global scalar; spatial grid retained in VDB |
| Spatial `ior` | unsupported as a medium grid | retained in VDB but unsupported by Cycles volume shading |
| Derived interfaces | dielectric/null PLY surfaces | internal interfaces unsupported |
| Scene and pixels | Mitsuba proof camera/light/transport | Blender proof camera/light/transport |

Final pixels are therefore not compared across consumers. The image sanity check also
does not impose equal hashes or equal brightness between renderers. Machine-readable
reports are generated under `.local/` and are not committed.
