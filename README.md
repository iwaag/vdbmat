# VBDMAT

VBDMAT is a renderer-independent preprocessing backend for voxel-based
material-jetting appearance research. The project will convert material voxel
data into explicit optical-property volumes for downstream renderers.

Phase 0 is complete: it establishes volume schemas, physical conventions,
persistence, and renderer interoperability proofs. The
[feasibility report](docs/phase0-feasibility-report.md) recommends proceeding to
Phase 1 without revising those foundations.

## Architecture

```text
material labels/mixtures -> optical mapping -> canonical optical volume -> Zarr
                                                   |                 |
                                                   v                 v
                                           Mitsuba adapter    OpenVDB/Cycles adapter
```

Core volume types define renderer-independent arrays, geometry, units, basis metadata,
and provenance. Optical mapping, storage, boundary derivation, and renderer adapters
are separate modules. Optional renderer dependencies are loaded only within exporters.

## Non-goals

The current project does not provide calibrated appearance prediction, production CAD
voxelization, printer input support, droplet/curing/process simulation, spectral
transport, large-volume optimization, GPU acceleration, production renderer plugins,
or a GUI. Proof images validate integration and gross trends, not physical equivalence
between renderers.

## Requirements

- [uv](https://docs.astral.sh/uv/)

uv manages the Python installation, virtual environment, dependencies,
lockfile, and project commands. No separate `pip`, Poetry, or Conda workflow is
supported.

## Development setup

Install the Python version selected by the repository and synchronize the
locked default environment:

```bash
uv python install
uv sync --locked
```

Run all checks through uv:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Use `uv add <package>` for runtime dependencies and
`uv add --group dev <package>` for development-only dependencies. Commit both
`pyproject.toml` and `uv.lock` when dependencies change.

## Optional integration environments

Renderer proofs are kept out of the default environment:

```bash
uv sync --locked --group mitsuba
uv sync --locked --group openvdb
```

The OpenVDB group is empty because compatible Python bindings and Blender are
provided by the isolated integration container documented in
[the OpenVDB/Cycles proof](docs/openvdb/phase0-cycles-proof.md).

## Fixture demo

Run the complete renderer-independent fixture path:

```bash
uv run python examples/phase0/inspect_synthetic_fixtures.py
uv run python examples/phase0/map_synthetic_fixtures.py
uv run python examples/phase0/zarr_fixture_report.py
uv run python examples/phase0/check_cross_consumer_conformance.py \
  .local/phase0/conformance.json
```

The six deterministic fixtures cover homogeneous media, a sharp interface, layered
materials, a mixture ramp, and anisotropic XYZ axis markers.

## Current package

The package currently exposes the Phase 0 core foundation:

```text
src/vbdmat/
  core/
    axes.py
    errors.py
    geometry.py
    materials.py
    metadata.py
    optical_basis.py
    transforms.py
    validation.py
    volumes.py
  fixtures/
    synthetic.py
  optics/
    config.py
    mapping.py
  io/
    zarr.py
  boundaries/
    interfaces.py
    policies.py
  exporters/
    diagnostics.py
    mitsuba.py
    openvdb.py
  conformance.py
tests/
```

The public core API includes `GridGeometry`, `OpticalBasis`, material palette
types, schema and provenance metadata, `MaterialLabelVolume`,
`MaterialMixtureVolume`, `OpticalPropertyVolume`, and structured volume
validation errors. Zarr v3 persistence supports failure-safe writes, validated
full reads, metadata-only inspection, and spatial optical-field reads. Optional
renderer adapters remain isolated from core imports and dependencies.

Generate and inspect the small deterministic Phase 0 fixtures with:

```bash
uv run python examples/phase0/inspect_synthetic_fixtures.py
```

The fixture set covers homogeneous transparent and white commands, a sharp
transparent/opaque interface, a layered slab, a two-material mixture ramp, and
an anisotropic axis marker.

Apply the explicit provisional and uncalibrated Phase 0 optical mapping with:

```bash
uv run python examples/phase0/map_synthetic_fixtures.py
```

The reference mapping uses direct label lookup and linear volume-fraction
mixing. Its assumptions and provisional values are documented in
[Phase 0 Reference Optical Mapping v1](docs/optics/reference-mapping-v1.md).

Inspect a persisted volume without loading array payloads, or generate the
fixture size and partial-read report:

```bash
uv run python examples/phase0/inspect_zarr.py path/to/asset.zarr
uv run python examples/phase0/zarr_fixture_report.py
```

Derive and inspect sharp refractive-index interfaces from every mapped fixture:

```bash
uv run python examples/phase0/inspect_ior_interfaces.py
```

Reproduce the optional Mitsuba IOR API probe with the locked renderer group:

```bash
uv run --group mitsuba python examples/phase0/probe_mitsuba_ior.py
```

The probe confirms that heterogeneous-medium spatial IOR is rejected while scalar
dielectric `int_ior`/`ext_ior` is accepted.

Render all canonical fixture proofs with fixed Mitsuba settings:

```bash
uv run --group mitsuba python examples/phase0/render_mitsuba_fixtures.py \
  .local/phase0/mitsuba-step9
```

Each fixture directory contains EXR/PNG output, oriented boundary PLY files, a scene
summary, and a machine-readable capability report.

Build and verify the isolated OpenVDB/Blender Cycles proof:

```bash
docker build -t vbdmat-phase0-step10:blender4.5.11 \
  -f tools/phase0/Dockerfile.openvdb-cycles .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH=/work/src -v "$PWD:/work" -w /work \
  vbdmat-phase0-step10:blender4.5.11 \
  python3 -m pytest -q tests/integration/test_openvdb.py \
  tests/integration/test_blender_cycles.py
```

Export and headless-render commands are in the
[OpenVDB/Cycles proof](docs/openvdb/phase0-cycles-proof.md).

Run the renderer-independent cross-consumer contract check for every fixture:

```bash
uv run python examples/phase0/check_cross_consumer_conformance.py \
  .local/phase0/conformance-step11.json
```

The command includes an exact Zarr round-trip and both pure adapter conversions. It
does not require renderer bindings. Optional `--mitsuba-renders` and
`--cycles-renders` arguments add gross PNG sanity checks without comparing renderer
pixels for physical equality. See
[Phase 0 cross-consumer conformance](docs/conformance/phase0-cross-consumer.md).

## Phase 0 design contracts

- [ADR-001: coordinates, axes, units, and sampling](docs/adr/0001-coordinates-axes-units-and-sampling.md)
- [ADR-002: canonical volume schemas](docs/adr/0002-canonical-volume-schemas.md)
- [ADR-003: boundaries and refractive index](docs/adr/0003-boundaries-and-refractive-index.md)
- [ADR-004: Zarr layout and compatibility](docs/adr/0004-zarr-layout-and-compatibility.md)
- [ADR-005: exporter boundary](docs/adr/0005-exporter-boundary.md)
- [Logical volume schema 1.0](docs/schemas/volume-schema-v1.md)
- [Worked schema examples](docs/schemas/examples/)
- [Phase 0 Zarr fixture report](docs/zarr/phase0-fixture-report.md)
- [Phase 0 Mitsuba consumer proof](docs/mitsuba/phase0-proof.md)
- [Phase 0 OpenVDB/Cycles consumer proof](docs/openvdb/phase0-cycles-proof.md)
- [Phase 0 cross-consumer conformance](docs/conformance/phase0-cross-consumer.md)
- [Phase 0 feasibility report](docs/phase0-feasibility-report.md)

## License

The repository is dedicated to the public domain under CC0 1.0 Universal. See
`LICENSE`.
