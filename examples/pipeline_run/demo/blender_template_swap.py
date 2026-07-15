"""Render an exterior-*.ply asset inside a hand-built Blender template scene.

This is a demo-track helper (see :mod:`blender_glass_demo`); it is not part of the
canonical pipeline and produces qualitative, uncalibrated images.

Why this exists: a from-scratch scene (as built by ``blender_glass_demo.py``) is too
simple for a real appearance check — no interesting backdrop, no reference geometry to
judge scale/reflection against. Instead this script opens a hand-authored template
``.blend`` (e.g. ``.local/local_demo/template_scene/cube_diorama.blend``) that already
has lighting, camera, and a placeholder object at the position/scale you want the
generated model to render at, and swaps that placeholder's mesh data for the new
``exterior-*.ply``. The placeholder's materials, name, scale/rotation, and collection
membership are retained. Its translation is adjusted so the replacement's world-space
bounds centre matches the placeholder's former centre; source-coordinate offsets
therefore do not move a different model out of the manually tuned framing.

Optionally, ``--interior-ply`` adds one or more ``interior-*.ply`` meshes (also from
``vdbmat export mitsuba``) as opaque solids anchored to the exact same transform the
exterior ended up with. ``vdbmat export mitsuba`` writes these for internal material
interfaces (e.g. an opaque core nested inside a transparent shell) as dielectric
patches meant for Mitsuba's internal-IOR transport; Cycles has no equivalent, so here
they are painted with a flat dark Diffuse BSDF instead of refracting light. This is a
qualitative approximation, not a physically matched interior boundary.

Invoke inside Blender's Python (the pinned ``vdbmat-openvdb-cycles`` image):

    blender --background --python \
        examples/pipeline_run/demo/blender_template_swap.py -- \
        TEMPLATE_BLEND EXTERIOR_PLY OUTPUT_PNG [--target-object exterior-000] \
        [--samples 96] [--interior-ply INTERIOR_PLY ...]

``TEMPLATE_BLEND`` is the hand-built scene file. ``EXTERIOR_PLY`` is a single
``exterior-*.ply`` (from ``vdbmat export mitsuba``). ``--target-object`` is the name of
the placeholder object in the template whose mesh gets replaced (default
``exterior-000``). ``--interior-ply`` may be passed multiple times; omitting it
reproduces the original exterior-only behaviour exactly. The script prints
``PIXELSTATS`` (min/max/mean/std of the rendered pixels), the same cheap headless
regression signal used by ``blender_glass_demo.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    try:
        extra = sys.argv[sys.argv.index("--") + 1 :]
    except ValueError:
        raise SystemExit(
            "expected: -- TEMPLATE_BLEND EXTERIOR_PLY OUTPUT_PNG [options]"
        ) from None
    parser = argparse.ArgumentParser(prog="blender_template_swap")
    parser.add_argument("template_blend", type=Path)
    parser.add_argument("exterior_ply", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--target-object", default="exterior-000")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--interior-ply",
        type=Path,
        action="append",
        default=None,
        help=(
            "interior-*.ply mesh (from vdbmat export mitsuba) to render as an "
            "opaque solid anchored to the exterior's final transform; may be "
            "passed multiple times"
        ),
    )
    return parser.parse_args(extra)


def _import_ply(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    try:
        bpy.ops.wm.ply_import(filepath=str(path))
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=str(path))
    (obj,) = (o for o in bpy.data.objects if o not in before)
    return obj


def _bounds_center(obj: bpy.types.Object) -> Vector:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return Vector(
        tuple(
            (
                min(corner[axis] for corner in corners)
                + max(corner[axis] for corner in corners)
            )
            * 0.5
            for axis in range(3)
        )
    )


def _swap_mesh_centered(
    target: bpy.types.Object, replacement: bpy.types.Object
) -> Vector:
    if target.type != "MESH":
        raise SystemExit(f"target object {target.name!r} is not a mesh")
    desired_center = _bounds_center(target)
    local_corners = [Vector(corner) for corner in replacement.bound_box]
    replacement_local_center = Vector(
        tuple(
            (
                min(corner[axis] for corner in local_corners)
                + max(corner[axis] for corner in local_corners)
            )
            * 0.5
            for axis in range(3)
        )
    )
    old_mesh = target.data
    materials = tuple(
        material for material in old_mesh.materials if material is not None
    )
    target.data = replacement.data
    target.data.materials.clear()
    for material in materials:
        target.data.materials.append(material)

    matrix = target.matrix_world.copy()
    matrix.translation = desired_center - matrix.to_3x3() @ replacement_local_center
    target.matrix_world = matrix
    bpy.data.objects.remove(replacement, do_unlink=True)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    return desired_center


def _interior_opaque_material() -> bpy.types.Material:
    name = "vdbmat-interior-opaque"
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    nodes = tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def _place_interior_mesh(
    path: Path, target: bpy.types.Object, material: bpy.types.Material
) -> bpy.types.Object:
    obj = _import_ply(path)
    obj.matrix_world = target.matrix_world.copy()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    for collection in target.users_collection:
        collection.objects.link(obj)
    obj.name = f"vdbmat-interior-{path.stem}"
    return obj


def _enabled_view_layers(scene: bpy.types.Scene) -> tuple[str, ...]:
    enabled = tuple(layer.name for layer in scene.view_layers if layer.use)
    if not enabled:
        raise SystemExit(
            "template has no View Layer enabled for rendering; enable at least one"
        )
    return enabled


def main() -> None:
    args = _parse_args()
    template_blend = args.template_blend.resolve()
    exterior_ply = args.exterior_ply.resolve()
    output_png = args.output_png.resolve()
    if not template_blend.is_file():
        raise SystemExit(f"no such Blender template: {template_blend}")
    if not exterior_ply.is_file():
        raise SystemExit(f"no such file: {exterior_ply}")
    if args.samples <= 0:
        raise SystemExit("--samples must be greater than zero")
    interior_plys = [path.resolve() for path in args.interior_ply or ()]
    for interior_ply in interior_plys:
        if not interior_ply.is_file():
            raise SystemExit(f"no such file: {interior_ply}")

    bpy.ops.wm.open_mainfile(filepath=str(template_blend))

    target = bpy.data.objects.get(args.target_object)
    if target is None:
        available = ", ".join(sorted(o.name for o in bpy.data.objects)) or "(none)"
        raise SystemExit(
            f"no object named {args.target_object!r} in {template_blend}; "
            f"available objects: {available}"
        )

    replacement = _import_ply(exterior_ply)
    center = _swap_mesh_centered(target, replacement)

    if interior_plys:
        interior_material = _interior_opaque_material()
        for interior_ply in interior_plys:
            _place_interior_mesh(interior_ply, target, interior_material)

    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        raise SystemExit(
            f"template render engine must be CYCLES, got {scene.render.engine!r}"
        )
    layers = _enabled_view_layers(scene)
    scene.cycles.samples = args.samples
    output_png.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_png)
    scene.render.image_settings.file_format = "PNG"

    print(
        f"SWAP target={target.name} exterior={exterior_ply.name} "
        f"center={tuple(round(value, 6) for value in center)} "
        f"materials={len(target.data.materials)} layers={','.join(layers)} "
        f"interior_count={len(interior_plys)}"
    )
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(str(output_png))
    pixels = image.pixels[:]
    channels = [pixels[i] for i in range(len(pixels)) if i % 4 != 3]
    lo = min(channels)
    hi = max(channels)
    mean = sum(channels) / len(channels)
    var = sum((c - mean) ** 2 for c in channels) / len(channels)
    print(
        f"PIXELSTATS target={args.target_object} "
        f"min={lo:.4f} max={hi:.4f} mean={mean:.4f} std={var**0.5:.4f}"
    )


if __name__ == "__main__":
    main()
