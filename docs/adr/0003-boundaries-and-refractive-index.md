# ADR-003: Boundaries and Refractive Index

- **Status:** Accepted
- **Date:** 2026-06-28
- **Decision owners:** VDBMAT maintainers
- **Phase:** 0, Step 8

## Context

The canonical optical volume stores a real, scalar, cell-centred `ior` value at every
voxel. Refraction, however, occurs at an interface between two media. The Phase 0
consumers expose substantially different volume APIs, and neither API establishes that
an arbitrary spatial `ior` grid creates physically meaningful sharp transitions.

This decision must preserve canonical information while identifying the minimum common
derived representation needed by the Mitsuba 3 and OpenVDB/Blender Cycles proofs.

## Consumer findings

### Mitsuba 3

Mitsuba's heterogeneous medium accepts spatial volume inputs for extinction `sigma_t`
and single-scattering albedo. Its grid-volume input supports trilinear or nearest-neighbor
filtering. The medium API does not accept a spatial IOR volume. Index mismatch is instead
described by a dielectric surface with scalar `int_ior` and `ext_ior`, with the exterior
defined as the side containing the surface normal.

Consequently, loading canonical `ior` as an ordinary `gridvolume` would not create
refraction. The proof must use nearest filtering for fields whose cell discontinuities
must remain sharp and use explicitly oriented dielectric boundary geometry for IOR.

Official references:

- [Mitsuba heterogeneous media](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html)
- [Mitsuba grid volumes](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html)
- [Mitsuba dielectric BSDF](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-dielectric-material-dielectric)

### Blender Cycles and OpenVDB

Blender loads named OpenVDB grids as volume data and exposes them to volume shading.
The Blender 4.5 Principled Volume inputs cover density, scattering color, absorption,
anisotropy, emission, and temperature, but not bulk-medium IOR. Blender's development
Volume Coefficients node has an input named IOR for the Fournier-Forand particle phase
model; it is the refractive index of scattering particles relative to water, not the
cell's bulk refractive index and cannot represent canonical `ior`.

Bulk refraction remains a surface operation through Glass or transmissive Principled
BSDFs. OpenVDB may preserve an `ior` grid as data, but a preserved grid is not evidence
that Cycles consumes it as refractive-index transport.

Official references:

- [Blender volume properties and OpenVDB grids](https://docs.blender.org/manual/en/4.5/modeling/volumes/properties.html)
- [Blender 4.5 Principled Volume](https://docs.blender.org/manual/en/4.5/render/shader_nodes/shader/volume_principled.html)
- [Blender Volume Coefficients](https://docs.blender.org/manual/en/dev/render/shader_nodes/shader/volume_coefficients.html)
- [Blender Glass BSDF](https://docs.blender.org/manual/en/4.5/render/shader_nodes/shader/glass.html)

## Candidate representations

### 1. Cell-centred spatial IOR only

This remains the canonical representation. It preserves the mapped effective property
at every cell and supports future consumers capable of gradient-index transport.

It is insufficient as the only renderer input in Phase 0:

- Mitsuba has no spatial-IOR parameter on its heterogeneous medium;
- Cycles volume shaders do not interpret an OpenVDB grid as bulk IOR;
- trilinear interpolation would smear a declared cell transition and still would not
  make either consumer refract through that field.

### 2. Explicit derived interface asset

An interface is emitted for every adjacent cell pair whose IOR difference exceeds an
explicit absolute tolerance. Grid exterior cells are compared with an explicitly
configured ambient IOR. Each face records:

- its semantic normal axis and minimum continuous XYZ corner index;
- the cell on each side, with `None` denoting exterior;
- negative-side and positive-side IOR;
- orientation from the negative semantic-axis side to the positive side.

Geometry and source schema/provenance remain attached to the interface set. World
corners and normals are calculated through the canonical rigid transform. This asset is
fully derived: deleting it loses no canonical data, and changing its threshold or
ambient requires recording a different derivation configuration.

This is the selected shared Phase 0 boundary representation.

### 3. Region or material boundary meshes

Closed region meshes are suitable renderer artifacts. They permit a dielectric surface
and medium assignment in Mitsuba and a Glass/transmissive surface plus volume material
in Cycles. They are not canonical because mesh partitioning, triangulation, coincident
surface handling, normal conventions, and renderer nesting rules are adapter concerns.

The common interface face set supplies the information needed to build these meshes.
Mesh construction is deferred to Steps 9 and 10 so it can satisfy each consumer's
specific correctness constraints without entering the core schema.

## Decision

1. `OpticalPropertyVolume.ior` remains required canonical data. Schema 1.0 does not
   gain a boundary array or mesh.
2. Phase 0 reconstructs IOR as piecewise constant per cell. It never interpolates IOR.
3. A sharp interface exists when `abs(ior_positive - ior_negative) > 1e-6` by default.
   The absolute threshold is explicit and configurable.
4. Exterior faces compare boundary cells with explicit `ambient_ior`, default `1.0`.
   If the values match within tolerance, no optical interface exists.
5. Material identity alone never creates an optical interface. Adjacent materials with
   equal IOR have no IOR face; a mixture ramp creates stair-step faces wherever its
   mapped cell values differ.
6. The derived interface set is renderer-neutral. Closed meshes, triangulation, surface
   shaders, medium nesting, and deduplication are exporter outputs.
7. Both adapters must issue a diagnostic for canonical `ior`, even when no faces are
   produced. An empty face set means index matched under the selected ambient/tolerance,
   not that the field was silently ignored.

## Interpolation and sampling policy

The canonical samples are cell-centred. For the Phase 0 proofs:

- optical coefficient volume sampling uses nearest-cell reconstruction at declared
  discontinuities unless an adapter explicitly diagnoses another choice;
- `ior` is never sent through trilinear volume interpolation;
- each derived face lies on the shared cell-corner plane, not halfway between cell
  centres;
- the face carries the exact float32 values from both cells as Python floats;
- a difference at or below the derivation tolerance is index matched and suppressed.

For the transparent/opaque interface fixture, the internal plane is continuous X index
`3`, at world X `0.01012 m`. It has eight internal faces, each oriented from IOR `1.48`
to `1.52`. For the layered fixture, the two materials at Z layers 1 and 2 both map to
IOR `1.52`, so their material boundary correctly creates no IOR interface.

## Adapter policies and mandatory diagnostics

The policies are machine-readable in `vdbmat.boundaries.policies`.

| Consumer | Canonical spatial `ior` | Derived interfaces | Required behavior |
| --- | --- | --- | --- |
| Mitsuba 3 | `unsupported` as a medium grid | `transformed` | Map oriented sides to dielectric `ext_ior`/`int_ior`; construct mutually compatible closed region meshes. |
| Blender Cycles | `unsupported` as bulk OpenVDB refraction | `approximated` | Build closed region surfaces using Glass/transmissive surface IOR; diagnose arbitrary adjacent-medium/nesting limitations. |

No exporter may omit the canonical field without reporting the `unsupported` spatial
mapping and the disposition of the derived interface asset.

## Background, air, and object boundaries

Background cells are ordinary canonical cells with explicit optical properties. A
background-to-material transition inside the grid is treated exactly like any other
adjacent IOR pair. The grid exterior is not an implicit material cell; its IOR comes
from `BoundaryDerivationConfig.ambient_ior`.

The visible/refractive object boundary is therefore the set of exterior and internal
faces with non-index-matched sides. The finite volume bounds still require closed
containment geometry for participating media even when exterior IOR is matched; an
adapter may create an index-matched containment mesh, but that mesh is a volume-domain
artifact and is not an IOR interface.

## Consequences

- Canonical data remains renderer independent and retains future gradient-index value.
- Both proofs consume the same deterministic interface topology and side values.
- Sharp stair-step boundaries expose voxel resolution instead of inventing smoothness.
- Face count can become large; sparse meshing and coplanar-face merging are deferred.
- The Cycles representation is explicitly approximate, especially for complex nested
  adjacent media.
- Medium containment meshes and IOR interface meshes are distinct concepts even when
  some faces coincide.

## Verification

Automated tests cover:

- exact face counts for all six fixtures;
- the interface fixture plane, orientation, adjacent cells, and side IOR values;
- suppression of equal-IOR material boundaries;
- threshold behavior above and below the default tolerance;
- ambient-index matching and exterior-face suppression;
- world corners, normal, anisotropic scale, translation, and rotation;
- explicit machine-readable policies for Mitsuba and Cycles.

