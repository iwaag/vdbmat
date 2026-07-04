# Phase 1 Research MVP Report

- **Date:** 2026-07-02
- **Status:** Complete
- **Recommendation:** Proceed to Phase 2 without changing schema 1.0.0 or ADR-001
  through ADR-008. Begin process-model implementation only after the input metadata
  gates in the Phase 2 handoff below are resolved.

## Executive summary

Phase 1 turns the Phase 0 contracts into an installed, reproducible command-line
workflow. A user can import an explicit material-label voxel asset or voxelize one
supported watertight STL, map the resulting material volume through the provisional
optical mapping, persist and validate both canonical volumes, inspect provenance and
checksums, and optionally export restored optical assets to Mitsuba or OpenVDB/Cycles.

The research MVP is technically reproducible for the small supported objects. It is
not a printer workflow, production voxelizer, calibrated material model, or physical
print predictor. Optical coefficients are provisional and uncalibrated.

## Supported inputs and limits

### Direct material voxels

`vdbmat.voxels/1.x` consists of one UTF-8 JSON manifest and one NumPy `.npy` payload.
The payload is exact `uint16[z,y,x]`; the manifest explicitly declares shape, metre or
millimetre voxel size, rigid placement, ordered palette including background ID 0,
source identity, relative payload path, and SHA-256. The reader disables pickle and
rejects path traversal, checksum mismatch, undeclared material IDs, implicit casts,
transposes, and unit inference.

Phase 1 does not claim compatibility with a printer-vendor format and does not expose
external mixture-volume interchange. Canonical `MaterialMixtureVolume` remains
supported internally and in Zarr.

### Triangle mesh

The mesh boundary accepts binary or ASCII STL only when it represents one watertight,
consistently oriented, non-degenerate, connected triangle solid. Unit (`m` or `mm`),
target voxel size in metres, and one non-background material ID are mandatory.
Placement is rigid; the default is identity. The dense reference voxelizer uses
cell-centre classification, a closed-solid surface rule, deterministic ray tie-breaking,
one padding cell by default, and a `1e-9 m` geometry tolerance.

Non-watertight, non-manifold, degenerate, empty, inconsistently oriented, and
multi-solid meshes are rejected. Self-intersection detection is not exhaustive.
Textures, per-face materials, scene graphs, B-reps, and non-rigid transforms are out
of scope. No axis may exceed 128 cells and no run may exceed 2,000,000 cells; this is a
safety bound, not an object-scale performance claim.

## Accepted Phase 1 decisions

- [ADR-006](adr/0006-phase1-inputs-and-voxelization.md) fixes the two input contracts,
  path/checksum rules, explicit units and palette semantics, dense domain construction,
  closed cell-centre occupancy, deterministic tolerances, and topology rejection.
- [ADR-007](adr/0007-pipeline-run-and-artifact-bundle.md) fixes the typed stage order,
  versioned configuration and digest, deterministic run ID, provenance chain, bundle
  layout, atomic publication, explicit overwrite, and no-resume policy.
- [ADR-008](adr/0008-cli-contract-and-failure-semantics.md) fixes the installed
  command set, stdout/stderr discipline, exit categories, overwrite behavior,
  optional-dependency failures, and API/CLI equivalence.

All three decisions are Accepted. Phase 1 composes the accepted Phase 0 schemas and
does not amend their coordinate, unit, sampling, persistence, boundary, or exporter
semantics. No migration note is required.

## Pipeline and artifact contracts

The deterministic stage sequence is:

```text
load direct voxels / voxelize mesh
  -> validate material -> persist material.zarr
  -> map provisional optics -> validate optical -> persist optical.zarr
  -> summarize -> optional export from restored optical.zarr
```

`vdbmat.pipeline-config/1.0.0` identifies the input kind and path, explicit mesh
settings when applicable, optical mapping name and digest, validation/export stages,
output path, overwrite policy, and execution seed. Its canonical JSON SHA-256 is the
configuration digest. The deterministic run ID combines that digest with the source
payload and mapping digests; timestamps do not affect scientific identity.

```text
run/
  run.json
  config.json
  source/
  material.zarr/
  optical.zarr/
  diagnostics/validation.json
  diagnostics/summary.json
  exports/{mitsuba,openvdb}/       # only when requested
```

`run.json` is `vdbmat.run/1.0.0`. It records stage status, versions, chained
provenance, relative artifact paths, schemas, sizes, and SHA-256 values. The Zarr
assets remain `vdbmat.volume/1.0.0`; the run manifest links rather than reinterprets
them. A sibling temporary directory is completely written and validated before atomic
publication. Resume is not supported; overwrite must be explicit and preserves the
previous valid bundle until its replacement validates.

## CLI reference and failure semantics

| Command | Successful result | Principal required input / expected failures |
| --- | --- | --- |
| `import-voxels MANIFEST OUTPUT` | canonical material-label Zarr | Manifest/palette/schema errors are code 3; missing, unsafe, or checksum-failed payload is code 4. |
| `voxelize MESH OUTPUT --unit U --voxel-size M[,M,M] --material-id ID` | canonical material-label Zarr plus occupancy diagnostics | Missing flags are code 2; invalid topology/placement is code 3; missing mesh is code 4. |
| `convert MATERIAL_ZARR OUTPUT` | canonical optical-property Zarr | Wrong asset or mapping failure is code 5; invalid volume is code 3. |
| `inspect ASSET` | metadata, geometry, provenance, counts/ranges, or run summary | Invalid manifest is code 3; missing asset is code 4. Inspection does not verify every payload checksum. |
| `validate ASSET` | full canonical validation and run checksum verification | Corruption is code 3; a missing declared asset is code 4. |
| `run CONFIG` | complete atomically published run bundle | Invalid config is code 3; stage/conversion failure is code 5; existing output without authorization is code 2. |
| `export {mitsuba,openvdb} OPTICAL_ZARR OUTPUT` | renderer artifacts and capability report; Mitsuba supports `--render` | Missing optional runtime is code 6; invalid source is code 3/4; adapter failure is code 5. |

All writing commands refuse an existing output unless `--overwrite` is supplied.
`--json` emits one machine-readable document on stdout; diagnostics remain on stderr.
Expected errors have no traceback unless `--debug` or `VDBMAT_DEBUG=1` is used.

| Exit | Meaning |
| ---: | --- |
| 0 | success |
| 1 | unexpected internal error |
| 2 | CLI usage or overwrite error |
| 3 | validation error |
| 4 | I/O, missing file, checksum, or unsafe path error |
| 5 | conversion, pipeline, or adapter error |
| 6 | optional dependency unavailable |

Run both supported paths with the [README quickstart](../README.md#phase-1-quickstart).
Optional renderer commands and limitations are in the
[export workflow](phase1-export-workflow.md).

## Representative objects and provenance

The direct-voxel window coupon is an axis-asymmetric multi-material regression object:
shape `(12,16,20)`, anisotropic voxel size `(0.0005,0.0004,0.0003) m`, placement
translation `(0.01,0.02,0.03) m`, and counts `{0:0, 1:3750, 2:72, 3:18}`. Its white
inclusion and separate black marker detect material loss and XYZ/ZYX reversal. Payload
SHA-256 is `fc9c3b362d692f4066f158f2b43dc1c44ffc84bb1eb9f3d31a5593be3b372e0b`.

The stepped wedge is a 36-triangle, four-step, single-material STL declared in
millimetres and voxelized at 1 mm. Its canonical shape is `(10,8,18)` with 480 occupied
cells split by step as 48, 96, 144, and 192; 960 cells are background. Source SHA-256
is `68c03e6e937d1ba39bc5b4c25feda5c4e1de7c923dbc18dbb19afecf064c818d`.

Both run bundles copy their source, record source/configuration/mapping digests, stamp
those identities into material and optical provenance, and checksum every declared
artifact.

## Verification and reproducibility evidence

The final clean-installed reproduction produced these results:

```text
uv lock --check                              pass
ruff format check / lint                     pass
mypy --strict src/vdbmat                     pass (44 source files)
default tests                                374 passed, 2 native-only skipped
locked Mitsuba tests                         10 passed
pinned OpenVDB/Blender integration tests     2 passed
cross-consumer conformance                   6 fixtures passed, 0 failures
installed-wheel reproduction                 pass outside repository root
```

Focused tests cover exact material conservation and mixture fractions, homogeneous
coefficients, closed mesh boundaries, translations/rotations/anisotropic grids,
equivalent units, triangle ordering, direct-input relocation, Zarr round trips and
chunk-independent reads, sharp IOR interfaces, atomic failures, artifact corruption,
deterministic reruns, API/CLI equality, and cross-consumer field/transform/unit/
background/capability conformance. The default suite imports no renderer or native DCC
dependency.

## Reference renders and capability diagnostics

Mitsuba 3.9.0 is the primary visual regression consumer: 256 x 256, 64 spp, seed
`20260628`. Both images are non-empty and non-flat and reproduce byte-identically.

| Object | Run ID | Material / optical Zarr SHA-256 | Mitsuba PNG SHA-256 |
| --- | --- | --- | --- |
| window coupon | `run-39e618b0049e5ef6` | `765ffd52…42cc81` / `969b152b…5995c5` | `f8a810cf…06dfdc` |
| stepped wedge | `run-044b29275258b6bf` | `b7c487ad…e9f649` / `b80bebb2…5ff693` | `a1aa8fa9…6b242b` |

![Window coupon Mitsuba baseline](../.local/phase1/step10/runs/window_coupon/exports/mitsuba/baseline.png)

![Stepped wedge Mitsuba baseline](../.local/phase1/step10/runs/stepped_wedge/exports/mitsuba/baseline.png)

The screenshots are generated evidence under `.local/`, not committed binary goldens.
Regeneration and complete hashes are documented in
[Phase 1 reference baselines](phase1-reference-baselines.md).

Mitsuba retains RGB extinction/albedo and derives IOR interface meshes, but reduces
spatial `g` to one scattering-weighted value and cannot consume heterogeneous medium
IOR directly. OpenVDB preserves component grids; Cycles reduces RGB coefficients to
scalar absorption/scattering and omits internal IOR interfaces. Its 64 x 64 result is
an interoperability smoke output, not a visual baseline. Both stable decoded-pixel
hashes are `25ef30d0…06a1a0`; PNG file bytes, OpenVDB, and `.blend` bytes are not claimed
stable. Capability reports make every transformed, approximated, and unsupported
semantic machine-readable. Pixels are never compared across renderers as physical
equivalents.

## Runtime, memory, and artifact observations

Measurements are environment-specific and are not scaling claims.

| Workload | Runtime | Peak memory | Artifact size |
| --- | ---: | ---: | ---: |
| installed coupon pipeline | 1.02 s | 51,132 KiB RSS | 119,976 B bundle |
| installed wedge pipeline | 0.48 s | 50,976 KiB RSS | 48,684 B bundle |
| both Mitsuba reference baselines | 2.32 s | 319,396 KiB RSS | included below |
| full two-object baseline evidence | — | — | 2,028,209 B |
| near-limit dense voxelization, `125^3` | ~1.6 s | ~76 MB RSS / ~29 MB traced | 1,953,125 cells |

The clean wheel is 96,370 bytes. A `128^3` grid is correctly rejected because it
exceeds the 2,000,000-cell total bound.

## Exit-criteria review

| Phase 1 criterion | Evidence |
| --- | --- |
| Direct material input without Python | Installed `import-voxels` and complete coupon config; manifest security tests. |
| Explicit supported mesh voxelization | Installed `voxelize` and wedge config; analytic/topology tests. |
| Both inputs produce valid schema 1.0.0 material volumes | Full CLI and bundle validation in Steps 4, 9, and 11. |
| One deterministic optical pipeline | Mapping digest, exact coefficient tests, deterministic reruns. |
| Failure-safe inspectable run bundle | ADR-007, fault injection, checksum corruption/missing-asset tests. |
| Installed CLI works outside repository | Clean wheel reproduction in Step 11. |
| Analytic coverage | Conservation, transforms, homogeneous fields, boundaries, persistence, and reruns pass. |
| Reproducible coupon and wedge outputs | Step 10 clean rerun and Step 11 installed rerun hashes match. |
| Renderers consume restored optical assets | Export runner and both optional environments pass independently. |
| Approximation is machine-readable | Mitsuba/OpenVDB `capabilities.json` and CLI JSON. |
| Default environment has no renderer/DCC requirement | 374-test host suite; only two explicitly native tests skip. |
| Technical reproduction is separated from prediction | README, CLI help, baseline docs, and this report state the limitation. |
| Phase 2 can insert process stages without schema change | Handoff below uses existing label/mixture contracts before optical mapping. |

All Phase 1 exit criteria have passing evidence or an accepted design decision.

## Known risks and required actions

| Risk / missing input | Owner | Consequence | Recommended action |
| --- | --- | --- | --- |
| No representative printer-vendor command file or stable format contract | Phase 2 input-contract owner | No printer compatibility or command fidelity claim is possible. | Obtain a legally usable representative file, specification, units, axes, palette, and checksum semantics before adding a vendor reader. |
| Printer resolution, layer thickness, build orientation, and anisotropy are absent | Process-model owner with printer-domain owner | Spread/blur kernels and layer interactions are underdetermined. | Freeze these fields in a Phase 2 config/ADR before fitting or validating a process stage. |
| Material-pair interaction and ordering metadata are absent | Materials/process owner | Mixing and interface behavior cannot be parameterized defensibly. | Define pair IDs, directional/order dependence, and provenance for measured or provisional parameters. |
| Dense implementation is capped at 2M cells | Performance owner | Representative printer-scale objects may exceed memory/runtime bounds. | Preserve the cap; specify halo-aware chunking, then benchmark sparse/out-of-core processing before raising it. |
| Owned STL self-intersection detection is incomplete | Geometry owner | Some invalid meshes may yield misleading occupancy. | Keep the narrow acceptance policy; evaluate a robust licensed topology checker when real Phase 2 geometry requires it. |
| Optical coefficients are provisional and RGB-effective | Phase 3 calibration owner | Renders cannot support quantitative or physical appearance claims. | Acquire measurement protocol and fit versioned coefficients; do not tune coefficients per fixture. |
| Cycles drops internal IOR and reduces RGB fields | Exporter owner | Cycles images cannot validate full canonical semantics. | Retain it as smoke-only; use field conformance and Mitsuba for stronger regression coverage. |

## Phase 2 process-model handoff

1. **Commanded input.** The commanded material field is the validated
   `MaterialLabelVolume` in Phase 1 `material.zarr`, before optical mapping. It retains
   the exact commanded IDs, grid, placement, palette, and source/configuration
   provenance. A future explicitly supported command mixture may enter through the
   same canonical boundary, but Phase 1 external inputs are labels only.
2. **Process-stage output.** Insert process modeling between material persistence and
   optical mapping. Neighborhood effects should produce a validated
   `MaterialMixtureVolume` with `float32[z,y,x,material]` fractions and ordered
   `uint16[material]` IDs. Fractions must remain in `[0,1]` and sum to one within
   `1e-6`. If fill/void is modeled separately, either represent void/background as an
   explicit palette fraction or define a new versioned intermediate contract; do not
   overload optical coefficients or silently renormalize lost mass.
3. **Neighborhood and halo.** Every process operator must declare its support in
   physical metres and derived cells per XYZ axis, boundary condition, layer-direction
   dependence, and deterministic halo width. Chunked and dense evaluation must agree
   on the interior after halo cropping; no operator may read an undocumented neighbor
   or depend on chunk partitioning.
4. **Mass conservation.** Track per-material commanded mass/volume, deposited amount,
   boundary loss, and any intentional reaction/void term. Conservation must hold
   globally and locally to an explicit numeric tolerance. Clipping or fraction
   normalization cannot hide a deficit; all loss/transfer terms belong in diagnostics
   and provenance.
5. **Metadata gate.** Before selecting blur/spread/mixing kernels, obtain printer
   native XY resolution, layer thickness, build orientation, voxel-to-layer mapping,
   droplet/deposition order, material identities, and material-pair interaction
   metadata. The process configuration and parameter source need stable identities and
   digests just like the Phase 1 mapping.
6. **Required fixtures.** Extend the coupon set with: a single isolated voxel/line for
   point/line spread; a sharp planar boundary in X, Y, and Z for anisotropic blur; two
   nearby features for overlap; alternating layers for vertical bleeding; unequal
   material-pair interfaces for directional mixing; a boundary-adjacent feature for
   loss behavior; and a mixture ramp for conservation. Each fixture needs analytic
   totals, symmetry/asymmetry expectations, and chunk-boundary variants.

Phase 2 should preserve `vdbmat.volume/1.0.0`, ADR-001 geometry, ADR-004 persistence,
and ADR-007 provenance/publication. A new process configuration and intermediate
diagnostic schema are expected; a canonical schema revision is not justified by the
current evidence.

## Final recommendation

Proceed to Phase 2 as a bounded process-model research phase. Do not expand Phase 1
claims or lift the dense safety bound. The first Phase 2 decision gate is metadata and
representative-data acquisition, followed by a versioned process-stage contract with
explicit halos and conservation accounting. Stop and revise that new contract if the
real printer input cannot supply unambiguous units, orientation, palette, or deposition
semantics.
