"""Render an exterior-*.ply asset inside a hand-built Blender template scene.

This is a demo-track helper (see :mod:`blender_glass_demo`); it is not part of the
canonical pipeline and produces qualitative, uncalibrated images.

Why this exists: a from-scratch scene (as built by ``blender_glass_demo.py``) is too
simple for a real appearance check — no interesting backdrop, no reference geometry to
judge scale/reflection against. Instead this script opens a hand-authored template
``.blend`` (e.g. ``.local/local_demo/template_scene/cube_diorama.blend``) that already
has lighting, camera, and a placeholder object at the position/scale you want the
generated model to render at, and swaps that placeholder's mesh data for the new
``exterior-*.ply``. Only the mesh data changes — the placeholder's transform,
materials, name, and collection membership are left untouched, so the template's
manually-tuned framing carries over automatically.

Invoke inside Blender's Python (the pinned ``vdbmat-openvdb-cycles`` image):

    blender --background --python examples/phase1/demo/blender_template_swap.py -- \
        TEMPLATE_BLEND EXTERIOR_PLY OUTPUT_PNG [--target-object exterior-000] \
        [--samples 96]

``TEMPLATE_BLEND`` is the hand-built scene file. ``EXTERIOR_PLY`` is a single
``exterior-*.ply`` (from ``vdbmat export mitsuba``). ``--target-object`` is the name of
the placeholder object in the template whose mesh gets replaced (default
``exterior-000``). The script prints ``PIXELSTATS`` (min/max/mean/std of the rendered
pixels), the same cheap headless regression signal used by ``blender_glass_demo.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def _parse_args() -> argparse.Namespace:
    try:
        extra = sys.argv[sys.argv.index("--") + 1 :]
    except ValueError:
        raise SystemExit("expected: -- TEMPLATE_BLEND EXTERIOR_PLY OUTPUT_PNG [options]")
    parser = argparse.ArgumentParser(prog="blender_template_swap")
    parser.add_argument("template_blend", type=Path)
    parser.add_argument("exterior_ply", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--target-object", default="exterior-000")
    parser.add_argument("--samples", type=int, default=96)
    return parser.parse_args(extra)


def _import_ply(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    try:
        bpy.ops.wm.ply_import(filepath=str(path))
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=str(path))
    (obj,) = (o for o in bpy.data.objects if o not in before)
    return obj


def _swap_mesh(target: bpy.types.Object, replacement: bpy.types.Object) -> None:
    old_mesh = target.data
    target.data = replacement.data
    bpy.data.objects.remove(replacement, do_unlink=True)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def main() -> None:
    args = _parse_args()
    if not args.exterior_ply.exists():
        raise SystemExit(f"no such file: {args.exterior_ply}")

    bpy.ops.wm.open_mainfile(filepath=str(args.template_blend))

    target = bpy.data.objects.get(args.target_object)
    if target is None:
        available = ", ".join(sorted(o.name for o in bpy.data.objects)) or "(none)"
        raise SystemExit(
            f"no object named {args.target_object!r} in {args.template_blend}; "
            f"available objects: {available}"
        )

    replacement = _import_ply(args.exterior_ply)
    _swap_mesh(target, replacement)

    scene = bpy.context.scene
    scene.cycles.samples = args.samples
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output_png)
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(str(args.output_png))
    pixels = image.pixels[:]
    channels = [pixels[i] for i in range(len(pixels)) if i % 4 != 3]
    lo = min(channels)
    hi = max(channels)
    mean = sum(channels) / len(channels)
    var = sum((c - mean) ** 2 for c in channels) / len(channels)
    print(
        f"PIXELSTATS target={args.target_object} "
        f"min={lo:.4f} max={hi:.4f} mean={mean:.4f} std={var ** 0.5:.4f}"
    )


if __name__ == "__main__":
    main()
