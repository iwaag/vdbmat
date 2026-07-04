# ADR-005: Exporter Boundary and Capability Diagnostics

- **Status:** Accepted
- **Date:** 2026-06-28
- **Decision owners:** VDBMAT maintainers
- **Phase:** 0, Step 9

## Context

The two Phase 0 rendering consumers do not share parameterizations, interpolation,
boundary models, dependency environments, or artifact formats. Renderer-specific state
must not enter the canonical volume model, and an adapter must not silently discard a
canonical property merely because its target cannot represent it directly.

## Decision

An exporter accepts one already validated `OpticalPropertyVolume`, explicit adapter
configuration, and an output location when it produces files. It may derive temporary
fields and boundary assets, but does not mutate the canonical volume.

Every export returns a machine-readable `CapabilityReport` containing:

- consumer and adapter identity/version;
- source schema name/version;
- one diagnostic entry for geometry, units, optical basis, `sigma_a`, `sigma_s`, `g`,
  `ior`, derived interfaces, and provenance;
- for every entry, one disposition and a concrete target mapping.

Allowed dispositions are:

- `represented`: meaning and values are consumed directly;
- `transformed`: a deterministic, documented, independently testable conversion is
  used without intentional information loss for the target contract;
- `approximated`: the adapter consumes a reduced or renderer-specific approximation;
- `unsupported`: the target cannot consume the property with the selected path.

An `unsupported` entry is not permission to omit diagnostics. If an approximation is
disabled or cannot be built, export must either fail clearly or retain the unsupported
entry in the report.

Renderer imports are lazy and remain below `vdbmat.exporters`. Importing `vdbmat`,
`vdbmat.core`, or running the core test suite must not require optional renderer
bindings. Renderer scenes, tensors, meshes, node graphs, color conversion, coefficient
conversion, interpolation, and integrator settings are adapter state.

## Mitsuba 3 mapping selected for Phase 0

- Scene coordinates are metres, so canonical `m^-1` coefficients use medium scale `1`.
- Grid data retains canonical `(z, y, x, basis)` storage; Mitsuba documents X-fastest
  tensor indexing and accepts this shape.
- `sigma_t = sigma_a + sigma_s` componentwise.
- `albedo = sigma_s / sigma_t` componentwise, with exact zero where `sigma_t == 0`.
- Both grids use nearest filtering to retain cell discontinuities.
- RGB coefficient tensors use raw grid data and are passed to Mitsuba's RGB transport
  variant; this is diagnosed as an effective-RGB approximation, not spectral data.
- Mitsuba's Henyey-Greenstein plugin accepts one scalar `g`, not a volume. The adapter
  uses a global scattering-weighted mean and reports `approximated`.
- Canonical spatial `ior` is unsupported as a medium volume. Exterior domain patches
  and derived internal interfaces use dielectric surfaces with explicit scalar side
  IORs. Interface meshes are reported separately from the unsupported spatial field.
- The complete domain boundary is emitted even when index matched, because medium
  containment is required independently of optical refraction.

## Failure behavior

Export rejects invalid configuration, unavailable bindings, an unsupported Mitsuba
variant, non-finite conversion results, or a scene that Mitsuba cannot load. It never
clips coefficients or canonical IOR. The sole phase reduction is the documented global
`g` approximation; canonical endpoints remain reported even if a future fixture needs a
separate renderer-safe policy.

Artifacts are written beneath a caller-provided directory. Reports use stable JSON.
Reference renders use a fixed perspective sensor, area backlight, volumetric path
integrator, resolution, SPP, and seed from explicit configuration. EXR stores the linear
transport output. PNG stores its display conversion. A second PNG applies the fixed
visualization `clip((1 - radiance) * 128, 0, 1)` to expose small attenuation differences;
this diagnostic image is not a replacement for the linear result and does not alter the
renderer inputs.

## OpenVDB / Blender Cycles mapping selected for Phase 0

- Canonical ZYX arrays are transposed to XYZ before OpenVDB `copyFromArray`; all
  exported grids are named `FloatGrid` values with a shared cell-centred affine.
- The affine retains anisotropic voxel dimensions, canonical rigid transform, and
  metre units. OpenVDB's row-vector matrix is the transpose of the canonical
  column-vector representation.
- The six RGB coefficient components, spatial `g`, and spatial `ior` are preserved as
  separate grids. Two additional scalar grids contain the explicit equal-weight RGB
  reductions used by the Cycles proof.
- Cycles Volume Absorption and Volume Scatter density inputs consume those derived
  scalar grids. This is reported as an approximation rather than a reinterpretation
  of the canonical RGB values.
- Cycles receives one scattering-weighted global anisotropy value. The spatial `g`
  grid remains in the artifact and is reported approximated.
- Spatial `ior` remains in the artifact but is unsupported by the selected Cycles
  volume path. Internal derived IOR interfaces are also unsupported; the adapter does
  not silently create index-matched or misleading surfaces.
- The Blender scene script fixes engine, CPU device, camera, light, resolution,
  samples, random seed, bounce limit, unit scale, and output format.

## Consequences

- Core code and persistence stay renderer independent.
- Adapter differences are reviewable and suitable for Step 11 conformance checks.
- A successful render does not imply full semantic support; the report remains the
  authoritative mapping record.
- New consumers can share diagnostic types without sharing scene construction.
