# Reference Baselines

The visual baselines are deterministic software-regression artifacts. Their optical
coefficients are provisional and uncalibrated; they are not predictions of a physical
print.

Generate both canonical run bundles and the primary Mitsuba baselines from a clean
output directory:

```bash
uv run --group mitsuba python examples/phase1/generate_reference_baselines.py \
  .local/baselines
```

This regenerates the inputs, runs both canonical pipelines with portable relative
configuration, renders Mitsuba 3 at 256 x 256, 64 spp, seed `20260628`, and writes
`baseline-manifest.json`. The manifest records source, configuration, mapping, run,
Zarr, scene, and image digests; canonical summaries; renderer settings and versions;
capability reports; and field-level conformance checks.

The coupon view uses the fixed geometry-framed perspective from direction
`(1.6, -2.2, 1.4)`. At baseline resolution its white inclusion and separate dark,
asymmetric marker are visible. The stepped-wedge view exposes all four steps.

OpenVDB/Cycles remains a pinned interoperability smoke consumer. Export and render
each object's restored `optical.zarr` with the commands in
[the export workflow](export-workflow.md), then attach the evidence to the manifest:

```bash
uv run python examples/phase1/generate_reference_baselines.py \
  .local/baselines --record-cycles
```

The pinned smoke settings are 64 x 64, 32 samples, seed `20260629`, and 8 maximum
bounces. Blender's PNG files contain nondeterministic encoder metadata, so their full
file SHA-256 is recorded as an observation but is not a stable baseline. The manifest
also records a SHA-256 over decoded dimensions and pixels; that hash is stable across
clean reruns. OpenVDB and `.blend` bytes are likewise not treated as byte-stable.

Mitsuba and Cycles pixels must not be compared as physical equivalents. Mitsuba uses
RGB extinction/albedo and derived IOR interface meshes. Cycles uses scalar coefficient
reductions and omits internal IOR interfaces.
