# Phase 0 Zarr Fixture Report

**Date:** 2026-06-28

**Environment:** CPython 3.11.14, Zarr 3.1.6, NumPy 2.4.6

**Layout:** Zarr format 3 directory store, Blosc/Zstandard level 5, bit shuffle

The report was generated with:

```bash
uv run python examples/phase0/zarr_fixture_report.py
```

Sizes include Zarr JSON metadata, chunk metadata, and compressed chunk payloads.
They are measurements of tiny proof fixtures, not compression benchmarks.

| Fixture | Canonical asset bytes | Mapped optical bytes | Partial shape ZYX | Exact partial values |
| --- | ---: | ---: | --- | --- |
| homogeneous-transparent | 3,116 | 7,176 | `(1, 1, 2)` | Yes |
| homogeneous-scattering-white | 3,121 | 7,821 | `(1, 1, 2)` | Yes |
| transparent-opaque-interface | 3,185 | 8,141 | `(1, 2, 3)` | Yes |
| layered-material-slab | 3,370 | 10,374 | `(2, 1, 2)` | Yes |
| two-material-mixture-ramp | 4,620 | 8,458 | `(1, 1, 2)` | Yes |
| anisotropic-axis-marker | 3,091 | 7,176 | `(1, 1, 2)` | Yes |

For each fixture, the canonical material asset and its mapped optical asset were
written independently. The partial region selected the first half of every spatial
axis, with a minimum of one cell. `sigma_a`, `sigma_s`, `g`, and `ior` were all
bit-exact against direct NumPy slicing. The returned region geometry preserved the
source cell's world-space location; automated coverage also checks a rotated grid.

All arrays use at most `(2, 2, 2)` spatial chunks and keep the complete RGB basis or
material axis in one chunk. Metadata-only inspection was verified without invoking
array payload access. Production chunk selection and remote-store transactional
behavior remain deferred.
