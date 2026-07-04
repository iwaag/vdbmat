# ADR-009: Input Generator Contract and External Material Mappings

- **Status:** Accepted
- **Date:** 2026-07-04
- **Decision owners:** VBDMAT maintainers
- **Phase:** 1-side1

## Context

Phase 1 closed with a working research MVP whose core pipeline accepted two inputs:
the `vbdmat.voxels` direct-voxel manifest (ADR-006) and a watertight STL that the
core itself voxelized. Before Phase 2 deepens print-process fidelity, the input
boundary must be fixed so that geometry/material *generation* and material
*coefficient data* can evolve independently of the core without weakening
reproducibility.

Three coupling problems motivate this ADR:

1. The core owns geometry-to-voxel conversion (`vbdmat.voxelize`), so every new
   generation method (image stacking, generative growth models, richer mesh
   pipelines) would grow the core.
2. The only optical mapping is the hardcoded `phase0-provisional-materials-v1`
   builtin; supplying measured or alternative coefficients requires a code change
   in `optics/`.
3. Material identity is informally split across `material_id`, palette `name`, and
   `external_id`, with no stated rule about which layer means what, inviting a
   silent conflation of simulation palettes with physical printer catalogs.

Backward compatibility with Phase 1 configurations is explicitly **not** required;
this is a research-stage breaking change.

## Decision

### D1 — The voxel manifest is the only core input

The core pipeline accepts exactly one input contract: a `vbdmat.voxels`
material-label manifest plus its checksummed `.npy` payload (ADR-006). The
`input.kind: "mesh"` path, `MeshVoxelizationSettings`, `vbdmat.voxelize`, the STL
reader, and the `vbdmat voxelize` CLI command are removed from the core package.
`vbdmat.pipeline-config` is bumped to **2.0.0**; version 1.x documents are rejected
by the major-version guard and no migration path is provided. The `InputKind` enum
is retained with the single member `direct-voxel` as the explicit extension point.

Mesh voxelization may later return as an external tool that *emits* a conforming
manifest (see D2). Its Phase 1 implementation is preserved in git history; the
design knowledge needed to rebuild it is recorded outside the package
(`.local/memo_stltovoxel.md`).

### D2 — The input-generator contract

Any producer of core input — mesh voxelizer, layered image stacker, generative
formation model, printer-slice converter — is an *input generator*. A generator is
conforming iff it emits a valid `vbdmat.voxels` manifest, which requires it to:

- declare shape, dtype (`uint16[z, y, x]`), voxel size with explicit units, and a
  rigid `local_to_world` transform;
- provide the payload as a pickle-free `.npy` file with a declared SHA-256;
- declare the full material palette (`material_id`, `name`, `role`, optional
  `external_id` and `metadata`);
- identify itself in `source.generator` / `source.generator_version` and, where
  possible, the input identity in `source.identity`.

The core never repairs, transposes, casts, or remaps generator output; violations
are field-oriented failures (ADR-006 semantics, unchanged). The core package
provides a writer helper (`vbdmat.io.voxel_manifest.write_material_label_manifest`)
that generators may depend on; the dependency direction is generator → core, never
the reverse.

### D3 — Optical mappings are supplied as data

The material coefficient mapping becomes swappable without code changes. A new
`vbdmat.optical-mapping` v1 JSON document carries exactly the fields of
`OpticalMappingConfig` (`configuration_id`, `version`, `optical_basis`,
`mixing_rule`, `calibration_status`, `materials[]` with `material_id`, `name`,
`sigma_a_rgb_per_m`, `sigma_s_rgb_per_m`, `g`, `ior`). Its canonical JSON and
digest are computed by the same rules as the builtin, so a mapping's identity is
independent of whether it was compiled in or loaded from a file.

`PipelineConfig` references a mapping by builtin `mapping.name` *or* by
`mapping.path`; the two are mutually exclusive. For a file-based mapping the
recorded `mapping.digest` is **required** and verified against the loaded file, so
canonical results remain a pure function of the scientific digest. The mapping
*path* stays out of the scientific digest (like the input path, ADR-007 D3); the
mapping *digest* is in it.

`phase0-provisional-materials-v1` remains available as a builtin and as a
reference external document whose digest must equal the builtin's.

### D4 — Two-layer material identity

Material identity has exactly two layers, which must never be conflated:

- **Simulation contract** — `material_id` (the join key between a palette and an
  optical mapping) plus `name` (human-readable; a palette/mapping `name` mismatch
  for the same `material_id` is an error, not a warning). This is the contract
  between external generators and the core's optical mapping.
- **Physical catalog** — `external_id` (and palette `metadata`) identifies real
  printer materials, batches, or vendor SKUs. It is provenance only: it never
  appears in an optical-mapping document and never participates in coefficient
  lookup. A future calibrated material library (Phase 3) will key on it, outside
  the scope of this ADR.

## Consequences

- The core pipeline shrinks to: manifest → material volume → optical volume →
  bundle, with no geometry processing.
- Existing v1 pipeline configurations and the `stepped_wedge` mesh input stop
  working; fixtures and examples are regenerated against the manifest contract.
- At least one non-mesh external generator must demonstrate the D2 contract end to
  end before Phase 1-side1 closes.
- ADR-006's mesh-path sections (topology rules, voxelization semantics) are
  superseded for the *core*; they remain the reference design for a future external
  voxelizer tool.
- ADR-007's run-bundle semantics are unchanged; only the config schema major and
  the mapping-resolution rule (D3) are affected.
