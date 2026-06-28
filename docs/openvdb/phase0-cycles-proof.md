# Phase 0 OpenVDB / Blender Cycles Consumer Proof

**Date:** 2026-06-29  
**Adapter:** `vbdmat.exporters.openvdb` 1.0.0

## Environment and reproduction

The core project intentionally does not install OpenVDB or Blender. OpenVDB Python
bindings are ABI-coupled native packages and are not available from the selected uv
index for this host; the `openvdb` dependency group therefore remains an isolation
point for a compatible system or Blender-provided binding rather than declaring a
non-reproducible wheel. Use a Python environment in which `import openvdb` (or
`import pyopenvdb`) succeeds, then run:

```bash
uv run python examples/phase0/export_openvdb_fixtures.py \
  .local/phase0/openvdb-step10

uv run python examples/phase0/render_blender_fixtures.py \
  .local/phase0/openvdb-step10 \
  .local/phase0/cycles-step10 \
  --blender /path/to/blender
```

The first command writes one `.vdb`, `openvdb-manifest.json`, and
`capabilities.json` per fixture. The second invokes Blender in background mode for
every manifest and writes a PNG, `.blend`, and hash report. No scene edits are needed.

This development host has neither OpenVDB Python bindings nor Blender installed.
Consequently, the native grid inspection and headless render tests are present but
were skipped here. Pure conversion, fake-binding serialization, diagnostics, and all
core tests pass. Native output hashes cannot honestly be recorded until the optional
runtime job is executed.

## Grid contract

The file contains ten named `FloatGrid` grids:

| Grid | Shape/order | Unit | Purpose |
| --- | --- | --- | --- |
| `sigma_a_r`, `sigma_a_g`, `sigma_a_b` | `(x, y, z)` | `m^-1` | Canonical absorption components |
| `sigma_s_r`, `sigma_s_g`, `sigma_s_b` | `(x, y, z)` | `m^-1` | Canonical scattering components |
| `g` | `(x, y, z)` | `1` | Canonical anisotropy |
| `ior` | `(x, y, z)` | `1` | Canonical refractive index |
| `cycles_absorption` | `(x, y, z)` | `m^-1` | Equal-weight RGB reduction |
| `cycles_scattering` | `(x, y, z)` | `m^-1` | Equal-weight RGB reduction |

OpenVDB maps NumPy array index `(0, 0, 0)` to VDB index `(i, j, k)=(0, 0, 0)`, so
canonical ZYX arrays are explicitly transposed to XYZ before `copyFromArray`. Each
grid receives the same affine transform. Integer VDB indices identify cell centres:

```text
world(i,j,k) = local_to_world(
    ((i + 0.5) * sx, (j + 0.5) * sy, (k + 0.5) * sz)
)
```

This retains anisotropic voxel size, rigid rotation, translation, and metre units.
The manifest stores the equivalent column-vector matrix; OpenVDB receives its
transpose because its `Mat4` convention right-multiplies row vectors. The optional
native test inspects names, `FloatGrid` types, bounds, selected axis-marker values,
and `indexToWorld` results. This follows the official
[OpenVDB Python array and I/O contract](https://www.openvdb.org/documentation/doxygen/python.html)
and [cell-centred transform guidance](https://www.openvdb.org/documentation/doxygen/transformsAndMaps.html).

## Cycles mapping

The fixed script creates one Blender Volume object, two Attribute nodes, a Volume
Absorption node, a Volume Scatter node, and an Add Shader connected to Material
Output. The derived scalar grids drive the two densities. The scattering-weighted
global mean of `g` drives the Volume Scatter anisotropy input. Camera, area light,
Cycles CPU engine, samples, seed, bounce limit, resolution, world unit scale, and
output format are fixed by the manifest.

| Canonical semantic | OpenVDB / Cycles mapping | Disposition |
| --- | --- | --- |
| Geometry | XYZ grids plus cell-centred affine in metres | Transformed |
| `sigma_a` | RGB grids retained; equal-weight scalar drives absorption | Approximated |
| `sigma_s` | RGB grids retained; equal-weight scalar drives scattering | Approximated |
| `g` | Grid retained; scattering-weighted global node value | Approximated |
| Spatial `ior` | Grid retained but not connected | Unsupported |
| Derived IOR interfaces | No internal dielectric surfaces | Unsupported |
| RGB basis | Components retained, scalar Cycles proof transport | Approximated |

The node construction uses Blender's documented volume density attributes and
[volume shader nodes](https://docs.blender.org/manual/en/latest/render/materials/components/volume.html).
The reductions are adapter-only. Canonical arrays are neither mutated nor relabeled.

## Limitations and decision

- The Cycles proof does not reproduce spatial RGB extinction one-to-one; it uses a
  declared scalar reduction.
- Cycles does not consume heterogeneous IOR through this volume path. Internal IOR
  interfaces are not approximated silently; both are reported unsupported.
- Sparse background voxels use each FloatGrid's zero background. This is exact for
  zero coefficients and `g`; positive IOR cells remain active.
- Actual Blender/OpenVDB compatibility and render outputs remain an optional-job gate
  on a host with both native runtimes. The absence of those runtimes does not add a
  core dependency.

The selected contract is suitable for Step 11 field/transform conformance work, but
the native runtime gate must pass before Phase 0 can claim the complete Step 10 render
verification.
