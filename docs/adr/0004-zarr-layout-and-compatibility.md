# ADR-004: Zarr Layout and Compatibility

- Status: Accepted
- Date: 2026-06-28

## Context

The canonical Python volume objects need a persistent, chunked representation that
retains their axis, unit, schema, palette, optical-basis, and provenance semantics.
The representation must also support spatial reads without loading an entire optical
volume.

## Decision

Phase 0 assets use Zarr format 3 directory stores. The root group contains a
`vdbmat` attribute holding the common manifest and an `arrays` child group holding
the canonical arrays.

```text
<asset>.zarr/
  zarr.json                         # root group and vdbmat manifest
  arrays/
    zarr.json
    material_id/                    # material-label only
    fractions/ and material_ids/    # material-mixture only
    sigma_a/, sigma_s/, g/, ior/    # optical-property only
```

The root manifest contains:

- schema name and semantic version;
- asset type;
- geometry (`shape_zyx`, `voxel_size_xyz_m`, and `local_to_world`);
- length unit, fixed to `m`;
- provenance;
- palette for material assets, or optical basis for optical assets;
- an array declaration for every required field.

Each array declaration and matching Zarr-array attributes repeat its dtype,
dimension names, and unit. Readers require both declarations to agree with the
logical schema and with the stored array. This intentional redundancy makes
corruption and accidental reinterpretation visible.

Canonical dtypes are stored without conversion: `uint16` for identifiers and
`float32` for fractions and optical fields. Zarr chunks use up to `2 x 2 x 2`
spatial cells for Phase 0 fixtures and include the complete material or RGB basis
axis. This deliberately small proof layout demonstrates partial reads; production
chunk sizing is deferred. Arrays use Zstandard through Zarr's Blosc codec at level
5 with bit shuffle.

Writers create a uniquely named temporary sibling directory, fully write it, reopen
and inspect its required structure, and then rename it into place. Creation at a
previously absent target is atomic on filesystems where same-directory rename is
atomic. Replacement first renames the old target to a backup, installs the new
target, and restores the backup if installation fails. Replacement is failure-safe
but has a short interval in which the final path is absent; cross-filesystem,
object-store, and multi-writer guarantees are outside Phase 0.

## Compatibility

- A schema name mismatch or major version other than `1` is rejected explicitly.
- The Phase 0 object model reads schema `1.0.x`. A newer minor version is rejected
  until its required-field changes are understood; metadata inspection still
  reports it.
- Unknown root attributes, manifest keys, array attributes, and arrays are ignored.
  This permits non-semantic optional additions without changing version meaning.
- Missing required attributes or arrays, dtype/dimension/unit mismatches, and shape
  mismatches are errors. Readers never infer, cast, transpose, clamp, or normalize.
- Patch releases retain the schema 1.0 contract and are accepted, then represented
  by the current canonical `1.0.0` runtime identity after validation.

## Partial reads

`read_optical_region` accepts three unit-stride half-open slices in `(z, y, x)`
order. It reads only the intersecting chunks of the four optical arrays. The returned
volume has the sliced shape and a translated `local_to_world` transform so local
cell `(0, 0, 0)` has the same world centre as the requested source start cell.

Material region reads are not needed for the Phase 0 proof and are deferred.

## Consequences

- Metadata can be inspected without reading array payloads.
- The same persisted optical asset can feed later renderer adapters.
- Tiny chunks add overhead and are not a production recommendation.
- Directory-store rename guarantees depend on the local filesystem; remote stores
  will need a commit-marker or versioned-prefix protocol.

