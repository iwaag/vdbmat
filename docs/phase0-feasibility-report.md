# Phase 0 Feasibility Report

**Review date:** 2026-06-29  
**Schema:** `vbdmat.volume` 1.0.0  
**Package:** `vbdmat` 0.1.0  
**Decision:** Proceed to Phase 1 without revising the Phase 0 foundation

## Executive conclusion

Phase 0 answered all four feasibility questions positively:

1. material placement, units, axes, transforms, and optical properties have one
   validated canonical interpretation;
2. all three volume assets round-trip through Zarr v3, and optical subregions preserve
   exact values and world placement;
3. the same canonical optical volume drives substantially different Mitsuba 3 and
   OpenVDB/Blender Cycles consumers without renderer state entering the core model;
4. schemas, storage, boundaries, mapping, and exporters have explicit module and ADR
   boundaries.

The foundation is technically credible for a research MVP. This is not evidence of
calibrated appearance prediction, production-scale performance, or cross-renderer
pixel equivalence. Phase 1 should retain the schema and adapter boundaries while
replacing provisional material data and adding measured validation.

## Scope and architecture

Phase 0 proves this data path:

```text
material-label / material-mixture
                |
                v
    provisional optical mapping
                |
                v
 canonical optical-property volume -----> Zarr v3
                |
          +-----+----------------+
          |                      |
          v                      v
 Mitsuba RGB medium      OpenVDB named grids
 + interface meshes      + Blender Cycles nodes
```

Canonical modules do not import renderer bindings. Exporters accept validated
`OpticalPropertyVolume` objects, perform explicit conversions, and emit capability
reports. Renderer-specific tensors, meshes, grids, nodes, cameras, and lighting remain
below the exporter boundary.

Phase 0 explicitly excludes production CAD voxelization, printer formats, process
physics, calibrated materials, spectral rendering, sparse/large-volume optimization,
GPU acceleration, production renderer plugins, and a GUI.

## Final canonical contracts

All spatial arrays use NumPy order `(z, y, x)` while semantic coordinates and geometry
use `(x, y, z)`. World space is right-handed, lengths are metres, samples are
cell-centred and piecewise constant, bounds are half-open, voxel size may be
anisotropic, and `local_to_world` is a right-handed rigid transform.

| Asset | Required canonical arrays | Key invariants |
| --- | --- | --- |
| Material label | `material_id: uint16[z,y,x]` | every ID declared; ID 0 is background |
| Material mixture | `fractions: float32[z,y,x,material]`, `material_ids: uint16[material]` | ordered palette; fractions in `[0,1]`; sum within `1e-6` |
| Optical property | `sigma_a`, `sigma_s: float32[z,y,x,basis]`; `g`, `ior: float32[z,y,x]` | coefficients non-negative in `m^-1`; `g` in `[-1,1]`; `ior > 0` |

The Phase 0 optical basis is effective linear RGB:
`linear-srgb-effective-v1`, D65, CIE 1931 2-degree observer, linear transfer. A future
spectral basis must add explicit wavelength coordinates; it may not reinterpret RGB
arrays.

Every asset carries schema identity, geometry, provenance, and either an ordered
material palette or optical-basis metadata. Constructors reject invalid shape, dtype,
range, normalization, transform, and non-finite values without silent casting,
clamping, transposition, or repair.

The normative definitions are the
[logical schema](schemas/volume-schema-v1.md),
[ADR-001](adr/0001-coordinates-axes-units-and-sampling.md), and
[ADR-002](adr/0002-canonical-volume-schemas.md).

## ADR outcomes

| ADR | Outcome | Phase 0 evidence |
| --- | --- | --- |
| 001 — coordinates, axes, units | Accepted: metres, semantic XYZ, storage ZYX, cell centres, anisotropic spacing, rigid placement | geometry round-trip and axis-marker tests |
| 002 — canonical schemas | Accepted: three versioned assets with explicit dtypes, dimensions, ranges, basis, palette, and provenance | constructor invariant tests and generated fixtures |
| 003 — boundaries and IOR | Accepted: retain cell-centred IOR; derive oriented faces at discontinuities; keep renderer mesh policy outside canonical data | interface derivation tests and Mitsuba PLY artifacts |
| 004 — Zarr layout | Accepted: Zarr v3 directory store, root manifest, `arrays/`, Blosc/Zstd, failure-safe sibling publication | round-trip, corruption, partial-read, and failure recovery tests |
| 005 — exporter boundary | Accepted: validated optical input, lazy optional dependencies, explicit capability report for every semantic | two adapters and cross-consumer conformance report |

No renderer finding required a schema or coordinate ADR revision.

## Tested environments

### Core and Mitsuba

| Component | Tested version |
| --- | --- |
| Host OS/time zone | Linux, Asia/Tokyo |
| uv | 0.10.3 |
| CPython baseline | 3.11; tested 3.11.14 |
| NumPy | 2.4.6 |
| Zarr | 3.1.6 |
| pytest / Ruff / mypy | 9.1.1 / 0.15.20 / 2.1.0 |
| Mitsuba / Dr.Jit | 3.9.0 / 1.4.0 |
| Mitsuba variant | `llvm_ad_rgb` |

Versions above are resolved by the committed `uv.lock`. The default environment
passes 241 tests with three optional integrations skipped. The locked Mitsuba group
passes 251 tests with only the two OpenVDB/Blender tests skipped.

### OpenVDB and Blender

The optional native stack is isolated in Docker image
`vbdmat-phase0-step10:blender4.5.11`:

| Component | Tested version |
| --- | --- |
| Base OS | Ubuntu 24.04 |
| Python / NumPy / pytest | 3.12.3 / 1.26.4 / 7.4.4 |
| OpenVDB Python package | 10.0.1-2.1build5 (`pyopenvdb`) |
| Blender | official 4.5.11 LTS Linux build |
| Renderer | Cycles CPU, denoising disabled |

Native OpenVDB readback and Blender headless rendering pass 2/2 integration tests.
Ubuntu's Blender 4.0.2 package crashed in Cycles with a minimal VDB; it is not a
supported proof environment.

## Zarr result

The selected layout is:

```text
asset.zarr/
  zarr.json              # root group and vbdmat manifest
  arrays/
    zarr.json
    <canonical fields>/  # one array per canonical field
```

Arrays retain canonical dtypes and dimension names. Spatial chunks are at most
`(2,2,2)` for the proof fixtures, with a complete RGB/material trailing axis. Blosc
with Zstandard level 5 and bit shuffle is used. Writes validate a temporary sibling
before rename publication; incompatible schema majors and corrupted required metadata
or arrays fail explicitly. Unknown optional fields are ignored.

| Fixture | Material asset | Optical asset | Partial shape ZYX | Exact partial read |
| --- | ---: | ---: | --- | --- |
| homogeneous-transparent | 3,116 B | 7,176 B | `(1,1,2)` | Yes |
| homogeneous-scattering-white | 3,121 B | 7,821 B | `(1,1,2)` | Yes |
| transparent-opaque-interface | 3,185 B | 8,141 B | `(1,2,3)` | Yes |
| layered-material-slab | 3,370 B | 10,374 B | `(2,1,2)` | Yes |
| two-material-mixture-ramp | 4,620 B | 8,458 B | `(1,1,2)` | Yes |
| anisotropic-axis-marker | 3,091 B | 7,176 B | `(1,1,2)` | Yes |

These are directory-size observations for tiny fixtures, not compression or scaling
benchmarks. Partial `sigma_a`, `sigma_s`, `g`, and `ior` values are bit-exact, and
cropped geometry preserves world placement, including under rotation. Production
chunking, cloud/object stores, concurrent writers, and crash-atomic remote publication
remain untested. See the [Zarr fixture report](zarr/phase0-fixture-report.md).

## Renderer capability matrix

Material label and mixture arrays are intentionally not renderer inputs; the shared
mapping first produces the optical asset below.

| Canonical semantic | Mitsuba 3 | OpenVDB artifact | Blender Cycles consumer |
| --- | --- | --- | --- |
| Geometry/transform | ZYX tensor and metric `to_world`; transformed | XYZ grids and cell-centred affine; transformed | VDB grid transform in metre scene; transformed |
| Coefficient unit | scene metres, medium scale 1; represented | `m^-1` grid metadata; represented | metre scale, no numeric scaling; represented |
| RGB optical basis | raw RGB transport tensor; approximated | RGB component grids retained | equal-weight scalar density reduction; approximated |
| `sigma_a` | `sigma_t` + albedo algebra; transformed | three RGB FloatGrids retained | scalar `cycles_absorption`; approximated |
| `sigma_s` | `sigma_t` + albedo algebra; transformed | three RGB FloatGrids retained | scalar `cycles_scattering`; approximated |
| `g` | scattering-weighted global HG value; approximated | spatial grid retained | same global node value; approximated |
| `ior` | heterogeneous field unsupported | spatial grid retained | not connected; unsupported |
| Derived IOR boundaries | oriented dielectric/null PLY patches; transformed | no interface asset | internal interfaces unsupported |
| Provenance/schema | capability JSON and scene summary; represented | file metadata, manifest, capability JSON; represented | consumed through matching VDB manifest |
| Background | zero extinction remains zero; complete exterior containment | zero FloatGrid background, active positive IOR | zero absorption/scattering density |

The shared Step 11 command passed all 60 field/transform/diagnostic checks. Adding
existing proof images passed 12/12 gross PNG sanity checks. Expected adapter differences
are recorded in the machine-readable report; pixels are not compared as physically
equivalent. See [cross-consumer conformance](conformance/phase0-cross-consumer.md).

## Reference proof checksums

Mitsuba reference images are 64 × 64, 32 spp, seed 20260628. Cycles hashes below are
the current 16 × 16, one-sample native load/render smoke outputs, not appearance
goldens.

| Fixture | Mitsuba display PNG SHA-256 | Cycles smoke PNG SHA-256 |
| --- | --- | --- |
| homogeneous-transparent | `3436eabe928df2adec6b4d54d1764cc24d074b22865d9583296844145a028f90` | `675113bd618ab59670de06002deec9195d2ce7693139dfe0af22a301ed43ff00` |
| homogeneous-scattering-white | `534ad7b3ea877a9be685b7ecc34e1dad159b072f3636523688c31bd5bdf3cbf9` | `675113bd618ab59670de06002deec9195d2ce7693139dfe0af22a301ed43ff00` |
| transparent-opaque-interface | `1e999e5d22d33018cb5459882240397b72042ffdb23712176c80af7ebc5f1ef6` | `80919b71244b8fd70e9453cb7af4c779cfd9973b57af617bfd317a2666efd1c0` |
| layered-material-slab | `cdaa81a3d6fcbdfcf8697763f6d6ca2f88ab8b2cf09c90d9a43de3df062c7e90` | `675113bd618ab59670de06002deec9195d2ce7693139dfe0af22a301ed43ff00` |
| two-material-mixture-ramp | `107afd1fe687b01afbc94a6cc7c7ac578f616ab63578fed280ec699ceb62ecb4` | `80919b71244b8fd70e9453cb7af4c779cfd9973b57af617bfd317a2666efd1c0` |
| anisotropic-axis-marker | `7d958bc7a4494bfc110ce5a411983fe0189079b49b0034e1d74604399548347d` | `675113bd618ab59670de06002deec9195d2ce7693139dfe0af22a301ed43ff00` |

Mitsuba attenuation-image hashes and linear means are recorded in the
[Mitsuba proof](mitsuba/phase0-proof.md). Repeated Cycles hashes reflect the low-sample,
dark smoke scene and must not be interpreted as fixture equivalence.

## Exit-criterion traceability

| Phase 0 exit criterion | Evidence |
| --- | --- |
| Coordinates, units, axes, transform, sampling, and color are explicit | ADR-001/002, schema docs, geometry and metadata tests |
| Three versioned canonical assets validate all invariants | `core.volumes`; focused valid/invalid unit tests |
| Optical schema includes `sigma_a`, `sigma_s`, `g`, and `ior`; boundaries decided | ADR-003; interface derivation tests |
| Synthetic material data maps deterministically to optical data | six fixture generators; mapping digest and regression tests |
| Zarr preserves values/semantics and supports partial reads | exact round-trip, corruption, inspection, partial-read tests and size report |
| Both consumers use the same canonical asset | adapter tests, six-scene Mitsuba proof, six-file OpenVDB/Cycles proof |
| Every approximation and unsupported semantic is reported | ADR-005 and capability-report completeness tests |
| Core runs without renderer dependencies | default suite: 241 passed, 3 optional skips; lazy-import tests |
| Locked environment is reproducible | `.python-version`, `pyproject.toml`, `uv.lock`, CI `uv sync --locked`, `uv lock --check` |
| Cross-consumer contracts agree | six fixtures, 60/60 contract checks; 72/72 with image sanity |

## Unresolved risks and Phase 1 decisions

None of these blocks starting Phase 1, but each must remain explicit.

| Risk | Owner | Consequence if ignored | Recommended Phase 1 decision |
| --- | --- | --- | --- |
| Optical coefficients are provisional and uncalibrated | Materials/research lead | rendered appearance has no predictive validity | define measurement protocol, uncertainty, versioned calibration datasets, and acceptance targets before appearance claims |
| Linear mixture rule omits droplet/process physics | Process-model lead | mixed voxels may be physically misleading | introduce a versioned process/mixing model behind the existing mapping boundary; retain the reference mapper as a test oracle |
| RGB effective coefficients are not spectral | Optics lead | metamerism and illuminant-dependent effects cannot be predicted | decide measured spectral sampling and basis metadata before adding spectral arrays; do not reinterpret schema 1.0 RGB |
| Mitsuba unit-cell interface patches are unmerged; complex nesting is approximate | Rendering lead | seams, duplicate edges, or incorrect nested dielectric behavior | prototype region extraction and watertight merged interfaces; validate normals/nesting on adversarial fixtures |
| Cycles uses scalar RGB reductions and omits internal IOR interfaces | Blender adapter owner | Cycles cannot be a fidelity reference for colored extinction/refraction | keep Cycles as interoperability/smoke consumer unless a measured error budget justifies a richer node/mesh adapter |
| Dense tiny-volume implementation has no scaling evidence | Storage/performance owner | research datasets may exceed memory or practical I/O time | benchmark representative volumes, select chunks from access patterns, then evaluate sparse processing and LOD |
| Directory-store publication is only local-filesystem failure-safe | Storage owner | object-store readers may observe partial assets | define store-specific commit markers/versioned prefixes before remote or concurrent writes |
| Native OpenVDB/Blender proof depends on a Docker image, not uv alone | Build/release owner | optional proof may drift or be unavailable on another architecture | publish or CI-build the pinned image and add a scheduled native integration job |

## Roadmap changes justified by the proofs

1. Keep `vbdmat.volume` 1.0.0 and the five accepted ADRs as the Phase 1 foundation;
   no migration is needed.
2. Treat calibration and measured validation as the first Phase 1 scientific gate,
   ahead of renderer polish.
3. Add process/mixing models behind `vbdmat.optics`; do not add printer physics to
   core volume types.
4. Develop merged region boundaries as a derived asset, not a required canonical
   field.
5. Retain capability reports and field-level conformance as mandatory for every new
   consumer.
6. Keep Cycles as an interoperability target until its scalar/IOR limitations have a
   quantified acceptable error; use Mitsuba as the stronger Phase 1 volume proof.
7. Require representative scale benchmarks before selecting sparse storage, GPU, or
   LOD work.

## Reproduction

Core foundation and conformance:

```bash
uv python install
uv sync --locked
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run python examples/phase0/zarr_fixture_report.py
uv run python examples/phase0/check_cross_consumer_conformance.py \
  .local/phase0/conformance-step11.json
```

Mitsuba proof:

```bash
uv sync --locked --group mitsuba
uv run --group mitsuba pytest tests/integration/test_mitsuba.py
uv run --group mitsuba python examples/phase0/render_mitsuba_fixtures.py \
  .local/phase0/mitsuba-step9
```

OpenVDB/Cycles proof:

```bash
docker build -t vbdmat-phase0-step10:blender4.5.11 \
  -f tools/phase0/Dockerfile.openvdb-cycles .

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH=/work/src -v "$PWD:/work" -w /work \
  vbdmat-phase0-step10:blender4.5.11 \
  python3 -m pytest -q tests/integration/test_openvdb.py \
  tests/integration/test_blender_cycles.py
```

Artifact export and rendering commands are documented in the
[OpenVDB/Cycles proof](openvdb/phase0-cycles-proof.md). A new contributor needs uv for
the core/Mitsuba path and Docker for the isolated OpenVDB/Blender native path.

## Final recommendation

Proceed to Phase 1. The canonical representation, Zarr persistence, boundary policy,
and exporter contract survived both consumer proofs and cross-consumer checks without
semantic ambiguity or renderer leakage. Phase 1 success must be measured against
calibrated physical data and representative scale; Phase 0 images alone are not an
appearance-validation result.
