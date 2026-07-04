# Phase 0 Reference Optical Mapping v1

## Status and Purpose

This mapping proves the canonical material-to-optical data flow. Every value in this document is **provisional and uncalibrated**. The values must not be presented as measurements of a printer or resin.

The implementation is `vdbmat.optics` and the stable rule identifier is `linear-volume-fraction-v1`.

## Inputs and Output

Accepted inputs:

- `MaterialLabelVolume`;
- `MaterialMixtureVolume`.

Output:

- `OpticalPropertyVolume` using `linear-srgb-effective-v1`;
- the same `GridGeometry` and volume schema as the input;
- mapping configuration SHA-256 in output provenance;
- input provenance sources, generator identity, and a complete provenance fingerprint.

Every material declared by the input palette must have a mapping entry, including unused palette entries and background ID `0`. Extra mapping entries are permitted.

## Label Rule

For a label cell containing material ID `m`, every output property is a direct lookup:

```text
sigma_a(x) = sigma_a[m]
sigma_s(x) = sigma_s[m]
g(x)       = g[m]
ior(x)     = ior[m]
```

No interpolation, neighborhood operation, or interface processing is applied.

## Mixture Rule

For material volume fractions `f_m(x)`, every property uses direct componentwise volume-fraction interpolation:

```text
sigma_a(x) = sum_m(f_m(x) * sigma_a[m])
sigma_s(x) = sum_m(f_m(x) * sigma_s[m])
g(x)       = sum_m(f_m(x) * g[m])
ior(x)     = sum_m(f_m(x) * ior[m])
```

The input fraction vector is already validated by `MaterialMixtureVolume`. Mapping computes in float64 and explicitly converts the final fields to canonical float32.

The same linear rule for `g` and `ior` is deliberately simple. It is not asserted to be the correct effective-medium or scattering-weighted model.

## Phase 0 Provisional Material Table

| ID | Name | `sigma_a` RGB (`m^-1`) | `sigma_s` RGB (`m^-1`) | `g` | IOR |
| ---: | --- | --- | --- | ---: | ---: |
| 0 | Air | `(0, 0, 0)` | `(0, 0, 0)` | 0.0 | 1.00 |
| 1 | Transparent resin | `(2, 1, 0.5)` | `(0, 0, 0)` | 0.0 | 1.48 |
| 2 | White resin | `(1, 1, 1)` | `(1000, 1000, 1000)` | 0.2 | 1.52 |
| 3 | Black opaque resin | `(4000, 5000, 6000)` | `(100, 100, 100)` | 0.1 | 1.52 |
| 10 | X diagnostic marker | `(0, 100, 100)` | `(0, 0, 0)` | 0.0 | 1.00 |
| 20 | Y diagnostic marker | `(100, 0, 100)` | `(0, 0, 0)` | 0.0 | 1.00 |
| 30 | Z diagnostic marker | `(100, 100, 0)` | `(0, 0, 0)` | 0.0 | 1.00 |

Diagnostic marker properties are intended only to expose orientation mistakes in later consumer proofs.

## Worked Mixture Example

For a voxel containing 50% transparent resin and 50% white resin:

```text
sigma_a = 0.5 * (2, 1, 0.5) + 0.5 * (1, 1, 1)
        = (1.5, 1.0, 0.75) m^-1

sigma_s = 0.5 * (0, 0, 0) + 0.5 * (1000, 1000, 1000)
        = (500, 500, 500) m^-1

g       = 0.5 * 0.0 + 0.5 * 0.2
        = 0.1

ior     = 0.5 * 1.48 + 0.5 * 1.52
        = 1.50
```

Pure mixture endpoints must equal direct label lookup for the same material.

## Determinism and Provenance

`OpticalMappingConfig` normalizes mapping entries by material ID and serializes a canonical JSON representation. Its SHA-256 digest changes when any material property, version, basis, rule, status, or configuration identifier changes.

Output provenance has no generated timestamp. It contains:

- mapping generator and version;
- exact configuration digest;
- original source identifiers;
- original generator identity;
- SHA-256 fingerprint of the full input provenance;
- configuration identifier and version;
- an explicit provisional/uncalibrated note.

## Known Physical Limitations

- RGB coefficients are effective transport channels, not a measured spectrum.
- Material coefficients have not been measured or fitted.
- Fractions are treated as optical volume fractions without a print-process model.
- Absorption and scattering are linearly interpolated without microstructure effects.
- `g` is linearly interpolated instead of scattering-weighted.
- IOR is linearly interpolated instead of using an effective-medium model.
- Sharp interfaces, Fresnel behavior, surface roughness, droplets, bleeding, curing, and shrinkage are absent.
- Spatially varying IOR boundary semantics remain unresolved until ADR-003.

These limitations are intentional. Later calibration or process models must use new versioned configurations or mapping rules rather than silently changing `linear-volume-fraction-v1`.
