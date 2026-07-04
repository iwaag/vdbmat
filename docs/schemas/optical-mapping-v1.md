# `vdbmat.optical-mapping` 1.0.0

An optical mapping supplied as data (ADR-009 D3): a UTF-8 JSON document carrying
exactly the fields of `OpticalMappingConfig`, so a mapping's canonical JSON and
SHA-256 digest are identical whether it is compiled in or loaded from a file.
The reader (`vdbmat.optics.load_optical_mapping`) never repairs or defaults
scientific values; violations are field-oriented failures.

## Document layout

```json
{
  "format": "vdbmat.optical-mapping",
  "format_version": "1.0.0",
  "configuration_id": "phase0-provisional-materials-v1",
  "version": "1.0.0",
  "optical_basis": {
    "kind": "rgb",
    "identifier": "linear-srgb-effective-v1",
    "coordinates": ["R", "G", "B"],
    "reference_white": "D65",
    "observer": "CIE-1931-2deg",
    "transfer": "linear"
  },
  "mixing_rule": "linear-volume-fraction-v1",
  "calibration_status": "provisional-uncalibrated",
  "materials": [
    {
      "material_id": 1,
      "name": "transparent-resin",
      "sigma_a_rgb_per_m": [2.0, 1.0, 0.5],
      "sigma_s_rgb_per_m": [0.0, 0.0, 0.0],
      "g": 0.0,
      "ior": 1.48
    }
  ]
}
```

## Rules

- **All top-level fields are required**; unknown fields anywhere are rejected.
- `format_version` major must be 1. `optical_basis` must be exactly the Phase 0
  RGB basis shown above (spectral bases are reserved for a future major).
- `materials[]` entries carry `material_id` (the join key against a volume's
  palette), `name`, RGB `sigma_a`/`sigma_s` in 1/m, `g` in [-1, 1], and `ior` > 0.
  Duplicate `material_id`s are rejected.
- **`external_id` is forbidden** (ADR-009 D4): physical printer-material catalog
  identifiers belong to the voxel manifest palette layer, never to coefficient
  lookup.
- The mapping's identity is `OpticalMappingConfig.digest` — the SHA-256 of its
  canonical JSON, reported by `vdbmat mapping-digest FILE`. A pipeline
  configuration referencing a mapping by `mapping.path` must declare this digest
  and the run fails if the file no longer hashes to it.

## Reference document

`examples/phase1/mappings/phase0-provisional-materials-v1.optical-mapping.json`
is the builtin Phase 0 mapping in external form; its digest equals the builtin's
by construction (asserted by `tests/optics/test_mapping_document.py`).
