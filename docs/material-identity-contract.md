# Material Identity Contract

This document expands ADR-009 D4: material identity in VBDMAT has exactly two
layers, and code or data that conflates them is defective.

## Layer 1 — Simulation contract (`material_id` + `name`)

The contract between external voxel generators and the core's optical mapping:

- `material_id` (uint16, 0–65535) is the **join key**. A voxel manifest's palette
  and an optical mapping are matched on it; ID 0 is conventionally the background.
- `name` is the human-readable identity of what the ID means (e.g.
  `transparent-resin`). It is not a free-form comment: when a palette and a
  mapping share an ID, their names **must be equal**, and
  `map_material_volume_to_optical` fails on any mismatch. This turns "ID 2 means
  white here but black there" from a silent wrong-coefficient bug into an error.
- The canonical name set for the Phase 0 provisional mapping is: `air` (0),
  `transparent-resin` (1), `white-resin` (2), `black-opaque-resin` (3), and the
  `axis-{x,y,z}-diagnostic` markers (10/20/30). Fixtures and example manifests use
  these names.

## Layer 2 — Physical catalog (`external_id` + palette `metadata`)

Identifies real printer materials: vendor SKUs, batches, cartridge codes. It lives
only in the voxel manifest's palette entries, as provenance:

- `external_id` never appears in a `vbdmat.optical-mapping` document (the reader
  rejects it) and never participates in coefficient lookup.
- A future calibrated material library (Phase 3) will key measured coefficient
  sets on this layer — scoped to printer, batch, and print mode — and *emit*
  Layer-1 mappings. Until then the two layers must not be mixed.

## Practical rules

1. A generator inventing a new simulated material picks an unused `material_id`
   and a descriptive `name`; the optical mapping supplied to the run must define
   the same ID with the same name.
2. Renaming a material is a breaking change to both the manifests and the mapping
   that mention it; do both together.
3. Swapping coefficients without renaming (e.g. a recalibration) changes the
   mapping digest, which changes the scientific digest and run IDs — as intended.
4. Anything about the *physical* material (vendor, batch, lot, cure profile) goes
   in `external_id`/`metadata`, not in `name`.
