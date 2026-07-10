# Mitsuba 3 Consumer

**Adapter:** `vdbmat.exporters.mitsuba`
**Mitsuba:** 3.9.0, variant `llvm_ad_rgb`
**Reference render:** 64 x 64, 32 spp, seed 20260628, `volpath` depth 8

## Reproduction

```bash
uv sync --locked --group mitsuba
uv run --group mitsuba python examples/phase0/render_mitsuba_fixtures.py \
  .local/mitsuba-fixtures
```

Each fixture directory contains:

- linear RGB EXR and display PNG;
- a fixed-gain attenuation diagnostic PNG;
- complete exterior containment and internal IOR-interface PLY meshes;
- `capabilities.json`;
- `scene-summary.json`.

The output root contains `render-report.json` with paths, hashes, extrema, and linear
RGB means.

## Mapping

| Canonical semantic | Mitsuba mapping | Disposition |
| --- | --- | --- |
| Geometry | ZYX tensor plus metric `to_world`; world PLY vertices | Transformed |
| `m^-1` units | Scene metres, medium scale 1 | Represented |
| RGB basis | Raw three-channel RGB transport tensor | Approximated |
| `sigma_a`, `sigma_s` | `sigma_t = sigma_a + sigma_s`; `albedo = sigma_s / sigma_t` | Transformed |
| `g` | Global scattering-weighted HG scalar | Approximated |
| Spatial `ior` | Not accepted by heterogeneous medium | Unsupported |
| Derived IOR interfaces | Oriented dielectric PLY patches | Transformed |

Both grids use nearest filtering. Exterior patches delimit the medium even when their
IOR is index matched. Internal dielectric patches retain the same heterogeneous medium
on both sides while changing refraction according to the two cell values.

## Orientation and scale evidence

Automated checks establish the consumer contract before rendering:

- tensor shape is exactly `(z, y, x, 3)`;
- selected X/Y/Z marker cells retain their distinct coefficient triplets;
- the volume transform maps the unit grid domain to canonical translated, anisotropic
  metric bounds;
- internal interface PLY coordinates use the same world transform;
- every fixture scene loads and renders without edits;
- mean image radiance decreases strictly as `sigma_a` or `sigma_s` increases, checked
  by ordering rather than fixed target values.

This is stronger than inferring orientation from a tiny image alone. The angled fixed
camera and attenuation diagnostic remain gross visual regression aids.

## Known limitations

- Spatial `g` is reduced to one scattering-weighted scalar.
- Effective RGB transport is not a spectral interpretation.
- Unit-cell boundary patches are not merged and can create coincident-edge artifacts.
- Complex nested dielectric correctness remains an approximation.
- Reference coefficients are provisional and uncalibrated.
- Images are regression/sanity artifacts, not cross-renderer physical ground truth.
