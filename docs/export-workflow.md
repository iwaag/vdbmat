# Optional Export Workflow

Optional exports consume a canonical optical volume restored from `optical.zarr`.
They do not participate in import, voxelization, optical mapping, validation, or
canonical persistence. Optical coefficients remain provisional and uncalibrated.

## Mitsuba

Install and invoke the locked optional group:

```bash
uv sync --locked --group mitsuba
uv run --group mitsuba vdbmat export mitsuba \
  .local/window-coupon/optical.zarr \
  .local/window-coupon/exports/mitsuba --json
```

The default command prepares a loadable scene, PLY domain/interface meshes,
`scene-summary.json`, and `capabilities.json`. Add `--render` to also produce the fixed
EXR, display PNG, and attenuation-diagnostic PNG. Scene preparation and rendering use
the same lazy Mitsuba adapter; neither changes the source Zarr asset.

## OpenVDB and Blender Cycles

OpenVDB is ABI-coupled and remains in the pinned native container:

```bash
docker build -t vdbmat-openvdb-cycles:blender4.5.11 \
  -f tools/Dockerfile.openvdb-cycles .

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH=/work/src -v "$PWD:/work" -w /work \
  vdbmat-openvdb-cycles:blender4.5.11 \
  python3 -m vdbmat.cli.main export openvdb \
  .local/window-coupon/optical.zarr \
  .local/window-coupon/exports/openvdb --json
```

The image combines Ubuntu's NumPy 1.26/OpenVDB 10.0.1 ABI with Zarr 3.0.10. Newer
Zarr releases require NumPy 2 and cannot share this OpenVDB 10 binding safely.

This writes `volume.vdb`, `openvdb-manifest.json`, and `capabilities.json`. Cycles
rendering is deliberately a separate native follow-up because Blender is not a Python
package dependency:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work vdbmat-openvdb-cycles:blender4.5.11 \
  blender --background --python examples/native_fixtures/blender_cycles_volume.py -- \
  .local/window-coupon/exports/openvdb/openvdb-manifest.json \
  .local/window-coupon/exports/openvdb/cycles.png
```

## Pipeline behavior and diagnostics

A config can request `mitsuba` or `openvdb` under `stages.exports`. The pipeline first
publishes the complete canonical bundle, restores its `optical.zarr`, and only then
runs adapters under `exports/<target>/`. Successful export files and checksums,
capability reports, adapter versions, and renderer versions are recorded in
`run.json`.

If an optional runtime is missing, standalone `vdbmat export` exits with code 6 and an
actionable environment instruction. A config-driven export is marked `failed` in
`run.json`, while the already-published material/optical Zarr assets remain valid.
Unsupported and approximated fields are explicit in `capabilities.json` and in JSON
CLI output. In particular, Cycles uses scalar RGB reductions and omits internal IOR
interfaces; Mitsuba reduces spatial `g` to one scattering-weighted value and derives
IOR interface meshes.
