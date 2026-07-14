"""Render a lay-person-legible "glass" demo of a Phase 1 optical asset.

This is a demo-track helper (see ``.local/sidemission_fancy_demo_blender.md``); it is
not part of the canonical pipeline and produces qualitative, uncalibrated images.

Why this exists: the Phase 1 test pieces are dominated by transparent resin. Cycles
omits internal IOR interfaces, so rendering them as a bare participating medium makes
them physically near-invisible (this is the root cause of the historic all-black
``cycles.png`` proofs). The fix demonstrated on 2026-07-02 is a *hybrid* setup:

  * the refractive **surface** is the ``exterior-*.ply`` mesh that ``export mitsuba``
    already writes (metres, same coordinate frame as the VDB), shaded as a Glass BSDF;
  * a lit **stage** (patterned backdrop + floor + environment) so the transparent
    object refracts something recognisable and reads as glass.

Invoke inside Blender's Python (the pinned ``vdbmat-openvdb-cycles`` image):

    blender --background --python examples/pipeline_run/demo/blender_glass_demo.py -- \
        MITSUBA_EXPORT_DIR OUTPUT_PNG [--ior 1.48] [--samples 96] [--size 400]

``MITSUBA_EXPORT_DIR`` is a directory containing ``exterior-*.ply`` (from
``vdbmat export mitsuba``). The script prints ``PIXELSTATS`` (min/max/mean/std of the
rendered pixels), which is a cheap, headless regression signal: a std near zero means
nothing rendered (empty frame), a healthy std means the object is actually visible.
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
        raise SystemExit("expected: -- MITSUBA_EXPORT_DIR OUTPUT_PNG [options]")
    parser = argparse.ArgumentParser(prog="blender_glass_demo")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--ior", type=float, default=1.48)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--size", type=int, default=400, help="square resolution")
    parser.add_argument("--exposure", type=float, default=-1.3)
    parser.add_argument(
        "--save-blend",
        type=Path,
        default=None,
        help="also write the built scene as a .blend next to the PNG (openable in Blender's GUI)",
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


def _glass_material(ior: float) -> bpy.types.Material:
    material = bpy.data.materials.new("vdbmat-glass")
    material.use_nodes = True
    tree = material.node_tree
    nodes = tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = ior
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["Color"].default_value = (0.95, 0.97, 1.0, 1.0)
    tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for obj in objects for c in obj.bound_box]
    lower = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
    upper = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
    return lower, upper


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _patterned_backdrop(center: Vector, radius: float) -> None:
    bpy.ops.mesh.primitive_plane_add(size=radius * 20, location=center + Vector((0, radius * 6, 0)))
    plane = bpy.context.active_object
    plane.rotation_euler = (1.5708, 0.0, 0.0)
    material = bpy.data.materials.new("vdbmat-backdrop")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 8.0
    checker.inputs["Color1"].default_value = (0.9, 0.2, 0.1, 1.0)
    checker.inputs["Color2"].default_value = (0.1, 0.3, 0.9, 1.0)
    material.node_tree.links.new(checker.outputs["Color"], principled.inputs["Base Color"])
    plane.data.materials.append(material)


def _floor(center: Vector, radius: float, floor_z: float) -> None:
    bpy.ops.mesh.primitive_plane_add(size=radius * 30, location=(center.x, center.y, floor_z))
    floor = bpy.context.active_object
    material = bpy.data.materials.new("vdbmat-floor")
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (
        0.35,
        0.35,
        0.38,
        1.0,
    )
    floor.data.materials.append(material)


def main() -> None:
    args = _parse_args()
    exteriors = sorted(args.export_dir.glob("exterior-*.ply"))
    if not exteriors:
        raise SystemExit(f"no exterior-*.ply in {args.export_dir}; run `vdbmat export mitsuba` first")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.max_bounces = 16
    scene.cycles.transmission_bounces = 16
    scene.cycles.transparent_max_bounces = 16
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.output_png)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.exposure = args.exposure

    glass = _glass_material(args.ior)
    surfaces: list[bpy.types.Object] = []
    for path in exteriors:
        obj = _import_ply(path)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.materials.append(glass)
        surfaces.append(obj)

    lower, upper = _world_bounds(surfaces)
    center = (lower + upper) * 0.5
    radius = max(upper - lower) * 0.5

    _patterned_backdrop(center, radius)
    _floor(center, radius, lower.z)

    scene.world = bpy.data.worlds.new("vdbmat-world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.28, 0.32, 0.4, 1.0)
    background.inputs["Strength"].default_value = 1.0

    light_data = bpy.data.lights.new("key", "AREA")
    light_data.energy = (radius * 10) ** 2 * 3.14159 * 9000.0
    light_data.size = radius * 4
    light = bpy.data.objects.new("key", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((-1.2, -1.5, 2.0)) * radius * 3
    _look_at(light, center)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 50.0
    camera_data.clip_start = radius * 0.001
    camera_data.clip_end = radius * 2000.0
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((2.4, -3.2, 1.6)) * radius
    _look_at(camera, center)
    scene.camera = camera

    bpy.context.view_layer.update()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))

    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(str(args.output_png))
    pixels = image.pixels[:]
    channels = [pixels[i] for i in range(len(pixels)) if i % 4 != 3]
    lo = min(channels)
    hi = max(channels)
    mean = sum(channels) / len(channels)
    var = sum((c - mean) ** 2 for c in channels) / len(channels)
    print(
        f"PIXELSTATS surfaces={len(surfaces)} radius={radius:.4f} "
        f"min={lo:.4f} max={hi:.4f} mean={mean:.4f} std={var ** 0.5:.4f}"
    )


if __name__ == "__main__":
    main()
