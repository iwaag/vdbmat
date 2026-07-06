# README_EXTEND.md — For External Voxel-Generator Tool Developers

Audience: developers writing tools that *produce* vdbmat's input data (material-labeled
voxel grids) — e.g. STL-to-voxel converters, 2D image-stack importers, generative
formation models. You do not need to read the vdbmat core codebase.
**The contract you must honor is the file format, not the generation algorithm.**

## 1. What you must ultimately emit

A pair of files:

1. `<name>.voxels.json` — metadata (the subject of this document)
2. `<name>.material_id.npy` — the material-ID voxel grid itself (`uint16`, axis order `z,y,x`)

The vdbmat core only reads these two files. Whether your source is an STL mesh, a CT
scan image stack, or a procedural formation model, it is accepted as input as long as
you can reduce it to this shape.

## 2. Required fields in `.voxels.json`

| Field | Value | Notes |
| --- | --- | --- |
| `format` | `"vdbmat.voxels"` | fixed literal |
| `format_version` | `"1.0.0"` | major version must be `1` |
| `asset_type` | `"material-label"` | fixed literal |
| `payload.path` | relative path to the `.npy` (POSIX, no `..`) | must resolve inside the manifest's own directory |
| `payload.sha256` | SHA-256 of the **raw `.npy` file bytes** | includes the numpy header; this is not a hash of the array data alone |
| `payload.dtype` | `"uint16"` | hardcoded — no other dtype is accepted |
| `payload.dimensions` | `["z","y","x"]` | this exact order only; no other ordering is parsed |
| `shape_zyx` | `[nz, ny, nx]` | must match the actual `.npy` shape |
| `voxel_size_xyz_m` **or** `voxel_size` | e.g. `[0.0005,0.0004,0.0003]` <br> or `{"value":[...], "unit":"m"\|"mm"}` | **exactly one** of the two — supplying both is an error. `mm` is auto-converted to `m` |
| `local_to_world` | 4x4 matrix | **rigid transform only** (rotation + translation). Last row must be `[0,0,0,1]`, the rotation block must be orthonormal (no scale/shear), determinant must be `+1` (right-handed) |
| `materials` | array of `{material_id, name, role}` | see "Material palette contract" below |
| `source` | `{generator, generator_version, identity, notes}` | free text is fine, but the object itself is required — record your tool's provenance |

Unknown keys anywhere in the document are rejected (strict schema).

## 3. Material palette (`materials`) contract — the easiest thing to get wrong

- `material_id = 0` must always have `role: "background"`. **No other ID may be `0`**,
  and no ID other than `0` may carry the `background` role.
- `role` accepts exactly two values: `"background"` or `"material"` — nothing else.
- `material_id` must be an integer in `[0, 65535]`, unique within the palette.
- **`name` is free text at this layer, but if it doesn't match a name known to
  vdbmat's optical mapping table, that material's optical coefficients (absorption,
  scattering, refractive index) cannot be resolved.**
  The built-in table currently supports exactly these names
  (`phase0-provisional-materials-v1`):
  `air`, `transparent-resin`, `white-resin`, `black-opaque-resin`,
  and the diagnostic markers `axis-x-diagnostic` / `axis-y-diagnostic` /
  `axis-z-diagnostic`. If you need a material outside this list, coordinate with the
  vdbmat side first — today there is only one built-in table, and external
  substitution is not yet supported (planned under roadmap `Phase 1-side1`).

## 4. Minimal example

```json
{
  "format": "vdbmat.voxels",
  "format_version": "1.0.0",
  "asset_type": "material-label",
  "payload": {
    "path": "sample.material_id.npy",
    "sha256": "<sha256 of sample.material_id.npy>",
    "dtype": "uint16",
    "dimensions": ["z", "y", "x"]
  },
  "shape_zyx": [12, 16, 20],
  "voxel_size_xyz_m": [0.0005, 0.0004, 0.0003],
  "local_to_world": [
    [1, 0, 0, 0.0],
    [0, 1, 0, 0.0],
    [0, 0, 1, 0.0],
    [0, 0, 0, 1]
  ],
  "materials": [
    {"material_id": 0, "name": "background", "role": "background"},
    {"material_id": 1, "name": "transparent-resin", "role": "material"}
  ],
  "source": {
    "generator": "your-tool-name",
    "generator_version": "0.1.0",
    "identity": "sample",
    "notes": "free text"
  }
}
```

See `examples/phase1/inputs/window_coupon.voxels.json` for a real example (its `.npy`
sits alongside it in the same directory).

## 5. Self-validation

```bash
uv run vdbmat import-voxels path/to/your.voxels.json
```

If this succeeds without error, your tool's output satisfies the contract. To run it
through the full pipeline, write a run config (`{"input": {"kind": "direct-voxel",
"path": "your.voxels.json"}, ...}`) and run `uv run vdbmat run <config>.json`.

## 6. Common pitfalls

- Re-saving the `.npy` after computing the checksum invalidates `sha256`. **Always
  recompute the manifest's `sha256` from the final `.npy` you actually ship.**
- Putting scale into `local_to_world` (voxel size belongs in `voxel_size_xyz_m`; the
  transform must stay rigid).
- Writing `dimensions` in a different order, e.g. `["x","y","z"]` (always fixed to
  `["z","y","x"]`).
- A palette `name` that disagrees with the optical mapping's name for the same
  `material_id` (the run fails: see `docs/material-identity-contract.md`).

## 7. Status (ADR-009)

Phase 1-side1 has landed: this contract is now the **sole** core input interface.
STL mesh voxelization was removed from the core (`input.kind: "mesh"` no longer
exists), and the optical coefficient mapping is pluggable — supply a
`vdbmat.optical-mapping` JSON document via the run config's `mapping.path` +
`mapping.digest` or the CLI's `convert --mapping-file` (see
`docs/schemas/optical-mapping-v1.md`; get the digest with
`vdbmat mapping-digest FILE`).

Two conveniences for generator authors:

- `vdbmat.io.write_material_label_manifest(directory, name, volume)` emits a
  conforming manifest + payload from a canonical volume, so you don't hand-roll
  the JSON.
- `vdbmat-utils convert-image-stack` (in the companion `vdbmat-utils`
  distribution) is a complete reference generator (grayscale PGM/PNG slice
  stack → manifest); its `vdbmat_utils.image` package is the API form.

## 8. Further reading (if you need more detail)

- `docs/adr/0009-input-generator-contract-and-external-mappings.md` — the input
  generator contract and mapping externalization decisions
- `docs/adr/0006-phase1-inputs-and-voxelization.md` — the formal ADR for the manifest
- `docs/material-identity-contract.md` — material naming rules and the two identity layers
- `docs/schemas/optical-mapping-v1.md` — the external optical-mapping document format
- `src/vdbmat/io/voxel_manifest.py` — validation implementation (source of truth for behavior)
- `src/vdbmat/core/materials.py` — `MaterialRole`/`MaterialPalette` definitions
- `src/vdbmat/optics/config.py` — built-in material names and their optical coefficients
