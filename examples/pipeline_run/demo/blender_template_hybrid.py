"""Render an exterior PLY and OpenVDB medium in a hand-built Blender scene.

This is a qualitative, uncalibrated demo helper.  It combines the glass exterior
written by ``vdbmat export mitsuba`` with the absorption/scattering grids written by
``vdbmat export openvdb``.  Internal IOR interfaces are intentionally not represented.
The template material slots and scale/rotation are preserved, while translation is
adjusted to align replacement and placeholder world-space bounds centres.

Invoke inside Blender's Python (the pinned ``vdbmat-openvdb-cycles`` image):

    blender --background --python \
        examples/pipeline_run/demo/blender_template_hybrid.py -- \
        TEMPLATE_BLEND MITSUBA_EXPORT_DIR OPENVDB_MANIFEST OUTPUT_PNG \
        [--target-object exterior-000] [--samples 96] [--save-blend OUTPUT_BLEND]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

REQUIRED_GRIDS = frozenset({"cycles_absorption", "cycles_scattering"})


def _parse_args() -> argparse.Namespace:
    try:
        extra = sys.argv[sys.argv.index("--") + 1 :]
    except ValueError:
        raise SystemExit(
            "expected: -- TEMPLATE_BLEND MITSUBA_EXPORT_DIR "
            "OPENVDB_MANIFEST OUTPUT_PNG [options]"
        ) from None
    parser = argparse.ArgumentParser(prog="blender_template_hybrid")
    parser.add_argument("template_blend", type=Path)
    parser.add_argument("mitsuba_export_dir", type=Path)
    parser.add_argument("openvdb_manifest", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--target-object", default="exterior-000")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--save-blend", type=Path, default=None)
    return parser.parse_args(extra)


def _load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    if not path.is_file():
        raise SystemExit(f"no such OpenVDB manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        vdb_name = manifest["vdb"]
        phase_g = manifest["phase_g"]
        grid_names = manifest["grid_names"]
        cycles = manifest["cycles"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(f"invalid OpenVDB manifest {path}: {error}") from error
    if not isinstance(vdb_name, str) or Path(vdb_name).name != vdb_name:
        raise SystemExit(f"invalid VDB filename in {path}: {vdb_name!r}")
    if not isinstance(phase_g, (int, float)) or isinstance(phase_g, bool):
        raise SystemExit(f"invalid phase_g in {path}: {phase_g!r}")
    if not math.isfinite(float(phase_g)) or not -1.0 <= float(phase_g) <= 1.0:
        raise SystemExit(f"phase_g must be finite and in [-1, 1]: {phase_g!r}")
    if not isinstance(grid_names, list) or not all(
        isinstance(name, str) for name in grid_names
    ):
        raise SystemExit(f"invalid grid_names in {path}")
    missing = REQUIRED_GRIDS - set(grid_names)
    if missing:
        raise SystemExit(f"OpenVDB manifest is missing grids: {sorted(missing)}")
    if not isinstance(cycles, dict) or cycles.get("engine") != "CYCLES":
        raise SystemExit(f"OpenVDB manifest does not describe a Cycles export: {path}")
    vdb_path = path.parent / vdb_name
    if not vdb_path.is_file():
        raise SystemExit(f"no such VDB file referenced by {path}: {vdb_path}")
    return manifest, vdb_path


def _single_exterior(directory: Path) -> Path:
    exteriors = sorted(directory.glob("exterior-*.ply"))
    if len(exteriors) != 1:
        count = len(exteriors)
        message = f"expected exactly one exterior-*.ply in {directory}, found {count}"
        raise SystemExit(message)
    return exteriors[0]


def _import_ply(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    try:
        bpy.ops.wm.ply_import(filepath=str(path))
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if len(imported) != 1 or imported[0].type != "MESH":
        raise RuntimeError(f"expected one mesh object from {path}, got {len(imported)}")
    return imported[0]


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


def _uniform_scale(matrix: Any, *, tolerance: float = 1e-6) -> float:
    basis = matrix.to_3x3()
    columns = [basis.col[index] for index in range(3)]
    scales = [column.length for column in columns]
    if min(scales) <= tolerance:
        raise SystemExit("target transform has a zero scale axis")
    scale = sum(scales) / 3.0
    if any(abs(value - scale) > tolerance * max(1.0, scale) for value in scales):
        raise SystemExit(f"target transform must have uniform scale, got {scales}")
    unit = [column / value for column, value in zip(columns, scales, strict=True)]
    if any(
        abs(unit[left].dot(unit[right])) > tolerance
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise SystemExit("target transform must not contain shear")
    if basis.determinant() <= 0.0:
        raise SystemExit("target transform must not contain reflection")
    return scale


def _attribute(nodes: bpy.types.Nodes, name: str) -> bpy.types.Node:
    node = nodes.new("ShaderNodeAttribute")
    node.attribute_name = name
    return node


def _volume_material(phase_g: float) -> bpy.types.Material:
    material = bpy.data.materials.new("vdbmat-cycles-volume")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    absorption.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Anisotropy"].default_value = phase_g
    absorption_attribute = _attribute(nodes, "cycles_absorption")
    scattering_attribute = _attribute(nodes, "cycles_scattering")
    links.new(absorption_attribute.outputs["Fac"], absorption.inputs["Density"])
    links.new(scattering_attribute.outputs["Fac"], scatter.inputs["Density"])
    add = nodes.new("ShaderNodeAddShader")
    links.new(absorption.outputs["Volume"], add.inputs[0])
    links.new(scatter.outputs["Volume"], add.inputs[1])
    links.new(add.outputs[0], output.inputs["Volume"])
    return material


def _add_volume(
    vdb_path: Path, target: bpy.types.Object, phase_g: float
) -> tuple[bpy.types.Object, tuple[str, ...]]:
    volume = bpy.data.volumes.new("vdbmat-volume")
    volume.filepath = str(vdb_path)
    volume.is_sequence = False
    if not volume.grids.load():
        raise RuntimeError(f"failed to load VDB grids: {volume.grids.error_message}")
    available = tuple(sorted(grid.name for grid in volume.grids))
    missing = REQUIRED_GRIDS - set(available)
    if missing:
        raise RuntimeError(f"VDB file is missing grids: {sorted(missing)}")

    volume_obj = bpy.data.objects.new("vdbmat-volume", volume)
    collections = tuple(target.users_collection)
    if not collections:
        raise RuntimeError(
            f"target object {target.name!r} is not linked to a collection"
        )
    for collection in collections:
        collection.objects.link(volume_obj)
    volume_obj.matrix_world = target.matrix_world.copy()
    volume.materials.append(_volume_material(phase_g))
    return volume_obj, available


def _pixel_stats(path: Path) -> tuple[float, float, float, float]:
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = image.pixels[:]
    channels = [pixels[index] for index in range(len(pixels)) if index % 4 != 3]
    lo = min(channels)
    hi = max(channels)
    mean = sum(channels) / len(channels)
    variance = sum((channel - mean) ** 2 for channel in channels) / len(channels)
    return lo, hi, mean, variance**0.5


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
    mitsuba_export_dir = args.mitsuba_export_dir.resolve()
    openvdb_manifest = args.openvdb_manifest.resolve()
    output_png = args.output_png.resolve()
    save_blend = args.save_blend.resolve() if args.save_blend is not None else None
    if not template_blend.is_file():
        raise SystemExit(f"no such Blender template: {template_blend}")
    if args.samples is not None and args.samples <= 0:
        raise SystemExit("--samples must be greater than zero")

    exterior = _single_exterior(mitsuba_export_dir)
    manifest, vdb_path = _load_manifest(openvdb_manifest)
    bpy.ops.wm.open_mainfile(filepath=str(template_blend))

    target = bpy.data.objects.get(args.target_object)
    if target is None:
        available = ", ".join(sorted(obj.name for obj in bpy.data.objects)) or "(none)"
        raise SystemExit(
            f"no object named {args.target_object!r} in {template_blend}; "
            f"available objects: {available}"
        )
    scale = _uniform_scale(target.matrix_world)
    replacement = _import_ply(exterior)
    center = _swap_mesh_centered(target, replacement)
    _, grids = _add_volume(vdb_path, target, float(manifest["phase_g"]))

    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        raise SystemExit(
            f"template render engine must be CYCLES, got {scene.render.engine!r}"
        )
    layers = _enabled_view_layers(scene)
    if args.samples is not None:
        scene.cycles.samples = args.samples
    output_png.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_png)
    scene.render.image_settings.file_format = "PNG"

    if save_blend is not None:
        save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))

    print(
        f"HYBRID target={target.name} exterior={exterior.name} vdb={vdb_path.name} "
        f"scale={scale:.6g} center={tuple(round(value, 6) for value in center)} "
        f"materials={len(target.data.materials)} layers={','.join(layers)} "
        f"grids={','.join(grids)} "
        "mode=qualitative-uncalibrated"
    )
    bpy.ops.render.render(write_still=True)

    lo, hi, mean, std = _pixel_stats(output_png)
    print(
        f"PIXELSTATS target={target.name} volume=vdbmat-volume "
        f"min={lo:.4f} max={hi:.4f} mean={mean:.4f} std={std:.4f}"
    )


if __name__ == "__main__":
    main()
