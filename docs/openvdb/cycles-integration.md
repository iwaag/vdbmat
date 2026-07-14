# OpenVDB / Blender Cycles Consumer

**Adapter:** `vdbmat.exporters.openvdb`
**OpenVDB:** 10.0.1 (`python3-openvdb`, module `pyopenvdb`)
**Blender:** 4.5.11 LTS official Linux build, Cycles CPU

## Environment and reproduction

The core project intentionally does not install OpenVDB or Blender. OpenVDB Python
bindings are ABI-coupled native packages and are not available from the selected uv
index. The reproducible optional environment is isolated in Docker:

```bash
docker build -t vdbmat-openvdb-cycles:blender4.5.11 \
  -f tools/Dockerfile.openvdb-cycles .

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH=/work/src -v "$PWD:/work" -w /work \
  vdbmat-openvdb-cycles:blender4.5.11 \
  python3 examples/native_fixtures/export_openvdb_fixtures.py \
  .local/openvdb-native

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work vdbmat-openvdb-cycles:blender4.5.11 \
  python3 examples/native_fixtures/render_blender_fixtures.py \
  .local/openvdb-native \
  .local/cycles-native --blender blender
```

The first command writes one `.vdb`, `openvdb-manifest.json`, and `capabilities.json`
per fixture. The second invokes Blender in background mode for every manifest and
writes a PNG, `.blend`, and hash report. No scene edits are needed.

Only the official Blender 4.5.11 LTS build is supported; Ubuntu's packaged Blender
4.0.2 crashes in Cycles with a minimal VDB. Denoising is disabled for deterministic
smoke renders and compatibility with CPU-only builds.

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
transpose because its `Mat4` convention right-multiplies row vectors. The native test
inspects names, `FloatGrid` types, bounds, selected axis-marker values, and
`indexToWorld` results. This follows the official
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

## Known limitations

- The Cycles proof does not reproduce spatial RGB extinction one-to-one; it uses a
  declared scalar reduction.
- Cycles does not consume heterogeneous IOR through this volume path. Internal IOR
  interfaces are not approximated silently; both are reported unsupported.
- Sparse background voxels use each FloatGrid's zero background. This is exact for
  zero coefficients and `g`; positive IOR cells remain active.
- Native smoke renders are intentionally low-cost load/render evidence, not an
  appearance baseline; orientation is verified through selected-value and
  `indexToWorld` inspection rather than image content alone.
