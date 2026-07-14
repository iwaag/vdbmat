# VDBMAT developer notes

Working notes for developers and debugging. Product/architecture direction lives in
`.local/roadmap.md`; renderer-facing demo plans live in the `.local/sidemission_*`
files. This file records hard-won, non-obvious findings and the tooling that produced
them.

## Renderer demo & debugging

### Finding (2026-07-02): why the OpenVDB→Cycles proof images render black

The historic `cycles.png` proofs (both Phase 0 and Phase 1) came out essentially black
or a flat uniform colour. Careful reproduction inside the pinned
`vdbmat-openvdb-cycles` container showed this is **not one code bug but a combination of
three factors**, the first of which is fundamental:

1. **Transparent test pieces × Cycles omitting internal IOR interfaces (the real
   cause).** The Phase 1 fixtures are dominated by — or made entirely of — transparent
   resin. Measured with `tools/vdb_inspect/inspect_vdb_grids.py`, `stepped_wedge` has an
   **all-zero `cycles_scattering` grid** and only a uniform `cycles_absorption` of
   `1.167 m⁻¹`; `window_coupon` is mostly transparent with small white/black inclusions.
   A non-scattering, weakly-absorbing medium is nearly invisible on its own, and the
   OpenVDB/Cycles adapter deliberately drops the internal IOR interfaces (documented in
   `docs/export-workflow.md`). So the object has no refractive surface to catch
   light and no medium to speak of → it disappears into the background. Mitsuba shows
   the shape/steps only because it builds the IOR interface meshes.
2. **Camera near-clip left at the 0.1 m default.** The objects are centimetre-scale
   (camera sits ~25 mm away), so the entire object falls inside the near-clip plane and
   is culled. `clip_start` must be scaled to the object size (a real bug in
   `examples/native_fixtures/blender_cycles_volume.py`).
3. **Near-black world (`0.02`) with a single backlight**, so even lit geometry gets
   almost no illumination.

Debugging cross-checks that isolated factor 1: a 2 m fog cube renders fine; a
hand-made uniform-density sphere VDB renders fine; the centimetre-scale *transparent*
VDB does not — at any object scale, with either an Attribute-node or a Principled
Volume shader.

### Fix (proven): hybrid glass surface + lit stage

Lighting/clip fixes alone are not enough because there is no surface to light. The
demonstrated fix is a **hybrid** setup:

* **Surface** = the `exterior-*.ply` mesh that `vdbmat export mitsuba` *already writes*
  (metres, same coordinate frame as the VDB), shaded as a **Glass BSDF at IOR 1.48**.
  There is no need to reconstruct a surface from the VDB — reuse the Mitsuba export.
* **Stage** = patterned backdrop + floor + environment light, so the transparent object
  refracts something recognisable.

With this, both `stepped_wedge` and `window_coupon` render as clear glass blocks that
refract the backdrop and cast a contact shadow — i.e. legible to a non-expert. These
images remain **qualitative and uncalibrated**; they are not physical predictions.

This is the technical basis of the Blender demo side-mission
(`.local/sidemission_fancy_demo_blender.md`, milestones B1/B2).

### Permanent tooling

Both run in the pinned `tools/Dockerfile.openvdb-cycles` image.

**Inspect what is actually inside an exported volume** (grid value ranges; instantly
reveals "this piece is pure transparent resin"):

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/work/src \
  -v "$PWD:/work" -w /work vdbmat-openvdb-cycles \
  python3 tools/vdb_inspect/inspect_vdb_grids.py \
  .local/phase1/step10/runs/stepped_wedge/exports/openvdb/openvdb-manifest.json
```

**Render the lay-person glass demo** from a Mitsuba export directory (needs
`exterior-*.ply`, produced by `vdbmat export mitsuba`):

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work vdbmat-openvdb-cycles \
  blender --background --python examples/pipeline_run/demo/blender_glass_demo.py -- \
  .local/phase1/step11/runs/stepped_wedge/exports/mitsuba \
  .local/demo/stepped_wedge.png --samples 96 --size 400
```

### Headless regression signal: `PIXELSTATS`

`blender_glass_demo.py` prints `PIXELSTATS ... std=<n>` for the rendered pixels. This is
a cheap way to catch "nothing rendered" regressions without eyeballing every image:
**std ≈ 0 means an empty/flat frame** (object invisible), while a healthy render lands
around `std ≈ 0.12–0.15`. During the investigation, identical `PIXELSTATS` across two
*different* objects was the tell that the volume was contributing nothing.
